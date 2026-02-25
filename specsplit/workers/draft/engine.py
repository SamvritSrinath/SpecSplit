"""Draft Engine — autoregressive speculative token tree generation.

The ``DraftEngine`` wraps a small, fast language model and generates
speculative token trees of depth *K*. These trees are sent to the Target
Worker for verification.

Architecture Notes:
    - **Session-based KV caching**: Maintains a ``dict[session_id, DraftCacheState]``
      mapping that stores KV state per session.  This ensures thread-safe
      operation when the gRPC server handles concurrent requests.
    - Cross-round KV cache reuse is supported: each call checks whether the
      prompt prefix matches the cached state (by token IDs, not just length)
      and extends incrementally when possible.
    - Token trees are represented as nested lists of ``TokenNode``-like
      dicts for easy protobuf conversion.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specsplit.core.config import DraftWorkerConfig
from specsplit.core.serialization import logits_to_probs
from specsplit.core.telemetry import Stopwatch

logger = logging.getLogger(__name__)


@dataclass
class TokenNode:
    """In-memory representation of a single node in the draft token tree."""

    token_id: int
    log_prob: float
    children: list[TokenNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict (mirrors the protobuf ``TokenNode`` message)."""
        return {
            "token_id": self.token_id,
            "log_prob": self.log_prob,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class DraftCacheState:
    """Per-session KV cache state for the Draft Engine.

    Attributes:
        kv_cache: The HuggingFace ``past_key_values`` from the last call.
        cached_prompt_len: Number of tokens encoded in the KV cache.
        cached_prompt_ids: Actual token IDs encoded in the KV cache,
            used to verify prefix match (not just length).
        cached_last_logits: Logits at the end of the cached prefix.
    """

    kv_cache: Any = None
    cached_prompt_len: int = 0
    cached_prompt_ids: list[int] = field(default_factory=list)
    cached_last_logits: torch.Tensor | None = None


class DraftEngine:
    """Autoregressive generation engine for the Draft Worker.

    This class manages model loading, KV cache state, and speculative
    tree generation using a real HuggingFace ``AutoModelForCausalLM``.

    Supports per-session KV caching for thread-safe concurrent requests.
    When ``session_id`` is provided to ``generate_draft_tree()``, each
    session gets its own isolated cache state and threading lock.

    Args:
        config: Draft worker configuration (model name, device, etc.).
    """

    def __init__(self, config: DraftWorkerConfig | None = None) -> None:
        self.config = config or DraftWorkerConfig()
        self.device = torch.device(self.config.device)
        self._model: Any = None  # transformers.AutoModelForCausalLM
        self._tokenizer: Any = None  # transformers.AutoTokenizer
        self._is_loaded = False

        # Session-based KV cache (Issue 5: thread-safety)
        self._session_caches: dict[str, DraftCacheState] = {}
        self._session_locks: dict[str, threading.Lock] = {}

        # Backwards-compatible singleton cache for session_id=None
        self._default_cache = DraftCacheState()

        logger.info(
            "DraftEngine initialized (model=%s, device=%s)",
            self.config.model_name,
            self.config.device,
        )

    def load_model(self) -> None:
        """Load the draft model and tokenizer via ``AutoModelForCausalLM``.

        Uses ``torch.float16`` precision and places the model on the
        configured device.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # Issue 12: Do NOT mutate tokenizer.pad_token — we manage our own
        # dense tensors and never use HF batch padding.

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
            )
            .to(self.device)
            .eval()
        )

        self._is_loaded = True
        logger.info("Draft model loaded: %s on %s", self.config.model_name, self.device)

    def _get_or_create_session(self, session_id: str) -> DraftCacheState:
        """Get or create a per-session cache state.

        Thread-safe: creates a lock per session on first access.
        """
        if session_id not in self._session_caches:
            self._session_caches[session_id] = DraftCacheState()
            self._session_locks[session_id] = threading.Lock()
        return self._session_caches[session_id]

    def generate_draft_tree(
        self,
        prompt_ids: list[int],
        k: int | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
        session_id: str | None = None,
    ) -> list[TokenNode]:
        """Generate a speculative token tree from the given prompt context.

        Performs *k* steps of autoregressive generation using KV caching.
        Each "beam" is an independent greedy/sampled chain (no beam-search
        coupling).  The result is a list of root ``TokenNode`` objects,
        each heading a flat chain of depth *k*.

        Args:
            prompt_ids: Tokenized prompt (list of vocabulary indices).
            k: Tree depth (defaults to ``config.max_draft_tokens``).
            num_beams: Number of independent chains (defaults to
                ``config.num_beams``).
            temperature: Sampling temperature (defaults to
                ``config.temperature``).
            session_id: Optional session ID for per-session KV cache
                isolation.  Required for thread-safe concurrent access.

        Returns:
            A list of root-level ``TokenNode`` objects forming the draft
            forest.
        """
        k = k or self.config.max_draft_tokens
        num_beams = num_beams or self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature

        sw = Stopwatch()
        sw.start()

        if not self._is_loaded or self._model is None:
            raise RuntimeError("Draft model not loaded. Call load_model() first.")

        # Select the appropriate cache state
        if session_id is not None:
            cache_state = self._get_or_create_session(session_id)
            session_lock = self._session_locks[session_id]
        else:
            cache_state = self._default_cache
            session_lock = None

        if session_lock is not None:
            session_lock.acquire()

        try:
            roots = self._generate_with_cache(prompt_ids, k, num_beams, temperature, cache_state)
        finally:
            if session_lock is not None:
                session_lock.release()

        sw.stop()
        logger.debug(
            "Generated true draft tree: depth=%d, root_branching=%d, elapsed=%.3f ms, session=%s",
            k,
            num_beams,
            sw.elapsed_ms,
            session_id,
        )
        return roots

    def _generate_with_cache(
        self,
        prompt_ids: list[int],
        k: int,
        num_beams: int,
        temperature: float,
        cache_state: DraftCacheState,
    ) -> list[TokenNode]:
        """Core generation logic operating on a specific cache state.

        This method is always called under the session lock (if applicable).
        """
        # -----------------------------------------------------------------
        # Encode the prompt prefix (with cross-round KV cache reuse)
        # -----------------------------------------------------------------
        # Issue 6: Validate cached token IDs match, not just length
        prefix_matches = (
            cache_state.kv_cache is not None
            and len(prompt_ids) >= cache_state.cached_prompt_len
            and cache_state.cached_prompt_len > 0
            and prompt_ids[: cache_state.cached_prompt_len] == cache_state.cached_prompt_ids
        )

        if prefix_matches:
            new_ids = prompt_ids[cache_state.cached_prompt_len :]
            if new_ids:
                # Case B: context grew — extend from cached KV with only the new tokens
                new_input = torch.tensor([new_ids], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    prefix_out = self._model(
                        input_ids=new_input,
                        past_key_values=cache_state.kv_cache,
                        use_cache=True,
                    )
                past_kv = prefix_out.past_key_values
                last_logits = prefix_out.logits[:, -1, :]
            else:
                # Case C: exact match — reuse stored KV and logits, no forward pass needed
                past_kv = cache_state.kv_cache
                last_logits = cache_state.cached_last_logits
        else:
            # Case A: no cache, stale, or mismatched prompt — full prefix recompute
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                prefix_out = self._model(
                    input_ids=input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
            past_kv = prefix_out.past_key_values
            last_logits = prefix_out.logits[:, -1, :]

        # Update cache state for next round
        cache_state.kv_cache = past_kv
        cache_state.cached_prompt_len = len(prompt_ids)
        cache_state.cached_prompt_ids = list(prompt_ids)
        cache_state.cached_last_logits = last_logits

        # -----------------------------------------------------------------
        # Task 2.2: True Level-Order Batching (Bug 4 Fix)
        # -----------------------------------------------------------------
        # Process ALL nodes at each depth level in a single forward pass,
        # instead of one forward pass per parent node. This maximizes GPU
        # utilization and eliminates the O(n) sequential bottleneck.
        roots: list[TokenNode] = []

        # Level queue: list of (parent_node, past_key_values, logits)
        # All entries share the same depth. We start at depth 0.
        current_level = [(None, past_kv, last_logits)]

        for depth in range(k):
            if not current_level:
                break

            # --- Phase 1: For each entry in current_level, compute top-B tokens ---
            next_level_entries: list[tuple[TokenNode, Any, torch.Tensor]] = []

            # Dynamic Branching Topology:
            # Heavy branching at root, reduce deeper to prevent exponential explosion.
            branching_factor = max(1, num_beams - depth)

            # Collect all (parent_node, token_id, prob, past_kv) for this level
            expansion_items: list[tuple[TokenNode | None, int, float, Any]] = []

            with torch.no_grad():
                for parent_node, entry_past_kv, entry_logits in current_level:
                    probs = logits_to_probs(entry_logits, temperature=temperature)

                    if temperature == 0.0:
                        top_probs, top_indices = torch.topk(probs, branching_factor, dim=-1)
                    else:
                        top_indices = torch.multinomial(probs, num_samples=branching_factor)
                        top_probs = torch.gather(probs, -1, top_indices)

                    for b in range(branching_factor):
                        token_id_int = top_indices[0, b].item()
                        prob_val = max(top_probs[0, b].item(), 1e-10)
                        log_prob = math.log(prob_val)
                        expansion_items.append((parent_node, token_id_int, log_prob, entry_past_kv))

            if not expansion_items:
                break

            # --- Phase 2: Batch ALL expansion items into a single forward pass ---
            total_items = len(expansion_items)
            batch_input_ids = torch.tensor(
                [[item[1]] for item in expansion_items],
                dtype=torch.long,
                device=self.device,
            )
            # Shape: [total_items, 1]

            # Stack KV caches along batch dimension
            first_kv = expansion_items[0][3]
            if first_kv is not None:
                # 1. Normalize all incoming KV caches to legacy tuples
                legacy_kvs = []
                for item in expansion_items:
                    kv = item[3]
                    if hasattr(kv, "to_legacy_cache"):
                        legacy_kvs.append(kv.to_legacy_cache())
                    elif hasattr(kv, "layers") and isinstance(kv.layers, (list, tuple)):
                        legacy_kvs.append(tuple((l.keys, l.values) for l in kv.layers))
                    else:
                        legacy_kvs.append(kv)

                # 2. Stack the legacy tuples along the batch dimension
                batch_past_kv = tuple(
                    tuple(
                        torch.cat([
                            legacy_kvs[j][layer_idx][t_idx]
                            if legacy_kvs[j][layer_idx][t_idx].shape[0] == 1
                            else legacy_kvs[j][layer_idx][t_idx][:1]
                            for j in range(total_items)
                        ], dim=0)
                        for t_idx in range(2)  # key, value
                    )
                    for layer_idx in range(len(legacy_kvs[0]))
                )

                # 3. Convert back to DynamicCache for modern HF models
                try:
                    from transformers import DynamicCache
                    batch_past_kv = DynamicCache.from_legacy_cache(batch_past_kv)
                except ImportError:
                    pass
            else:
                batch_past_kv = None

            with torch.no_grad():
                step_out = self._model(
                    input_ids=batch_input_ids,
                    past_key_values=batch_past_kv,
                    use_cache=True,
                )
            # step_out.logits: [total_items, 1, vocab_size]
            # step_out.past_key_values: batched KV cache

            # --- Phase 3: Unpack and create TokenNodes, populate next_level ---
            for i, (parent_node, token_id_int, log_prob, _) in enumerate(expansion_items):
                child_node = TokenNode(token_id=token_id_int, log_prob=log_prob)

                if parent_node is None:
                    roots.append(child_node)
                else:
                    parent_node.children.append(child_node)

                # Extract this item's KV cache from the batched output
                if isinstance(step_out.past_key_values, tuple):
                    branch_past_kv = tuple(
                        tuple(t[i : i + 1] for t in layer)
                        for layer in step_out.past_key_values
                    )
                elif hasattr(step_out.past_key_values, "__getitem__"):
                    branch_past_kv = step_out.past_key_values
                else:
                    branch_past_kv = step_out.past_key_values

                next_level_entries.append((
                    child_node,
                    branch_past_kv,
                    step_out.logits[i : i + 1, -1, :],
                ))

            current_level = next_level_entries

        return roots

    def reset_cache(self, session_id: str | None = None) -> None:
        """Clear the KV cache (e.g., on prompt change or verification failure).

        Args:
            session_id: If provided, clears only that session's cache.
                If None, clears the default (singleton) cache.
        """
        if session_id is not None:
            if session_id in self._session_caches:
                lock = self._session_locks.get(session_id)
                if lock:
                    lock.acquire()
                try:
                    cache = self._session_caches[session_id]
                    cache.kv_cache = None
                    cache.cached_prompt_len = 0
                    cache.cached_prompt_ids = []
                    cache.cached_last_logits = None
                finally:
                    if lock:
                        lock.release()
                logger.debug("Session KV cache cleared: %s", session_id)
        else:
            self._default_cache.kv_cache = None
            self._default_cache.cached_prompt_len = 0
            self._default_cache.cached_prompt_ids = []
            self._default_cache.cached_last_logits = None
            logger.debug("Default KV cache cleared")

    def end_session(self, session_id: str) -> bool:
        """Terminate a session and free its KV cache.

        Args:
            session_id: The session to terminate.

        Returns:
            True if the session existed and was removed.
        """
        lock = self._session_locks.get(session_id)
        if lock:
            lock.acquire()
        try:
            cache = self._session_caches.pop(session_id, None)
            if cache is not None:
                cache.kv_cache = None
                cache.cached_last_logits = None
                logger.info("Draft session ended: %s", session_id)
                return True
        finally:
            if lock:
                lock.release()
            self._session_locks.pop(session_id, None)
        return False

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._is_loaded
