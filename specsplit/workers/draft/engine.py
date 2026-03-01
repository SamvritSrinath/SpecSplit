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


# Top-K size for draft distribution (covers ~99% mass for residual sampling).
_TOP_K_DRAFT_PROBS = 64


@dataclass
class TokenNode:
    """In-memory representation of a single node in the draft token tree."""

    token_id: int
    log_prob: float
    children: list[TokenNode] = field(default_factory=list)
    top_k_token_ids: list[int] = field(default_factory=list)
    top_k_probs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict (mirrors the protobuf ``TokenNode`` message)."""
        return {
            "token_id": self.token_id,
            "log_prob": self.log_prob,
            "children": [c.to_dict() for c in self.children],
            "top_k_token_ids": self.top_k_token_ids,
            "top_k_probs": self.top_k_probs,
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
        self._eos_token_id: int | None = None

        # Session-based KV cache (Issue 5: thread-safety)
        self._session_caches: dict[str, DraftCacheState] = {}
        self._session_locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

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
        self._eos_token_id = self._tokenizer.eos_token_id
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
        with self._global_lock:
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

        logger.debug(
            "Draft generation: k=%d, num_beams=%d, temperature=%.2f (greedy=%s)",
            k, num_beams, temperature, temperature == 0.0,
        )

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
        Uses longest-common-prefix matching for O(1) rollback on speculation miss.
        """
        # -----------------------------------------------------------------
        # Encode the prompt prefix (LCP matching for O(1) rollback on miss)
        # -----------------------------------------------------------------
        match_len = 0
        if cache_state.kv_cache is not None and cache_state.cached_prompt_len > 0:
            min_len = min(len(prompt_ids), cache_state.cached_prompt_len)
            while match_len < min_len and (
                prompt_ids[match_len] == cache_state.cached_prompt_ids[match_len]
            ):
                match_len += 1

        if match_len == len(prompt_ids) and match_len == cache_state.cached_prompt_len:
            # Case A: Exact match — O(1) reuse, no forward pass needed.
            past_kv = cache_state.kv_cache
            last_logits = cache_state.cached_last_logits
        elif match_len > 0:
            # Case B: Extension or rollback — crop to match_len-1, process tail.
            # One-token forward generates last_logits while reusing prefix cache.
            reuse_len = match_len - 1
            if reuse_len == 0:
                # No prefix to reuse; fall through to full recompute.
                past_kv = None
                new_ids = prompt_ids
            else:
                past_kv = cache_state.kv_cache

                if cache_state.cached_prompt_len > reuse_len:
                    if hasattr(past_kv, "crop"):
                        past_kv.crop(reuse_len)
                    elif isinstance(past_kv, tuple):
                        past_kv = tuple(
                            (k[:, :, :reuse_len, :], v[:, :, :reuse_len, :])
                            for k, v in past_kv
                        )

                new_ids = prompt_ids[reuse_len:]

            if past_kv is None:
                # reuse_len was 0; full recompute.
                new_input = torch.tensor([new_ids], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    prefix_out = self._model(
                        input_ids=new_input,
                        past_key_values=None,
                        use_cache=True,
                    )
            else:
                new_input = torch.tensor([new_ids], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    prefix_out = self._model(
                        input_ids=new_input,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
            past_kv = prefix_out.past_key_values
            last_logits = prefix_out.logits[:, -1, :]
        else:
            # Case C: Complete mismatch or empty cache — full O(N) recompute.
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
            # Greedy (T=0): branching_factor must be 1 — secondary beams have P=0
            # and would always reject on target; branching is mathematically invalid.
            # Stochastic: reduce branching deeper to prevent exponential explosion.
            if temperature == 0.0:
                branching_factor = 1
            else:
                branching_factor = max(1, num_beams - depth)

            # Collect all (parent_node, token_id, prob, past_kv, top_k_ids, top_k_probs)
            expansion_items: list[
                tuple[TokenNode | None, int, float, Any, list[int], list[float]]
            ] = []

            with torch.no_grad():
                for parent_node, entry_past_kv, entry_logits in current_level:
                    # Fix Bug 3: With temp=0, logits_to_probs returns one-hot; topk would
                    # yield invalid non-primary beams. Use topk on raw logits instead.
                    if temperature == 0.0:
                        if branching_factor == 1:
                            top_indices = entry_logits.argmax(dim=-1, keepdim=True)
                            top_probs = torch.ones_like(
                                top_indices, dtype=entry_logits.dtype
                            )
                        else:
                            top_logits, top_indices = torch.topk(
                                entry_logits, branching_factor, dim=-1
                            )
                            top_probs = torch.softmax(top_logits, dim=-1)
                    else:
                        probs = logits_to_probs(entry_logits, temperature=temperature)
                        top_indices = torch.multinomial(
                            probs, num_samples=branching_factor
                        )
                        top_probs = torch.gather(probs, -1, top_indices)

                    # Top-K draft distribution for correct residual sampling.
                    # Use one-hot when greedy; residual = relu(P_target - P_draft) must
                    # reflect the actual sampling distribution (one-hot for greedy).
                    k_probs = min(_TOP_K_DRAFT_PROBS, entry_logits.shape[-1])
                    if temperature == 0.0:
                        probs_for_topk = logits_to_probs(entry_logits, temperature=0.0)
                    else:
                        probs_for_topk = logits_to_probs(
                            entry_logits, temperature=temperature
                        )
                    topk_probs, topk_idx = torch.topk(probs_for_topk, k_probs, dim=-1)
                    topk_ids = topk_idx[0].cpu().tolist()
                    topk_vals = topk_probs[0].cpu().tolist()

                    for b in range(branching_factor):
                        token_id_int = top_indices[0, b].item()
                        prob_val = max(top_probs[0, b].item(), 1e-10)
                        log_prob = math.log(prob_val)
                        
                        if self._eos_token_id is not None and token_id_int == self._eos_token_id:
                            child_node = TokenNode(
                                token_id=token_id_int,
                                log_prob=log_prob,
                                top_k_token_ids=topk_ids,
                                top_k_probs=topk_vals,
                            )
                            if parent_node is None:
                                roots.append(child_node)
                            else:
                                parent_node.children.append(child_node)
                            continue

                        expansion_items.append(
                            (
                                parent_node,
                                token_id_int,
                                log_prob,
                                entry_past_kv,
                                topk_ids,
                                topk_vals,
                            )
                        )

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
                        legacy_kvs.append(
                            tuple((lyr.keys, lyr.values) for lyr in kv.layers)
                        )
                    else:
                        legacy_kvs.append(kv)

                # 2. Vectorized batch: native torch.cat per layer over the batch dimension
                # Replaces O(layers * batch) flat iterations and reshaping with direct layer concatenation
                n_layers = len(legacy_kvs[0])
                batch_past_kv = tuple(
                    (
                        torch.cat([legacy_kvs[j][l][0] for j in range(total_items)], dim=0),
                        torch.cat([legacy_kvs[j][l][1] for j in range(total_items)], dim=0),
                    )
                    for l in range(n_layers)
                )

                # 3. Pass batched legacy tuple — HF models accept it; avoid DynamicCache
                # conversion so test mocks (expecting tuple) and real models both work.
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
            for i, (parent_node, token_id_int, log_prob, _, topk_ids, topk_probs) in enumerate(
                expansion_items
            ):
                child_node = TokenNode(
                    token_id=token_id_int,
                    log_prob=log_prob,
                    top_k_token_ids=topk_ids,
                    top_k_probs=topk_probs,
                )

                if parent_node is None:
                    roots.append(child_node)
                else:
                    parent_node.children.append(child_node)

                # Extract this item's KV cache from the batched output (Fix Bug 1).
                # When model returns DynamicCache, convert to legacy and slice per beam.
                pkv = step_out.past_key_values
                legacy_pkv = None
                if isinstance(pkv, tuple):
                    legacy_pkv = pkv
                elif hasattr(pkv, "to_legacy_cache"):
                    legacy_pkv = pkv.to_legacy_cache()
                elif hasattr(pkv, "layers") and isinstance(pkv.layers, (list, tuple)):
                    legacy_pkv = tuple(
                        (layer.keys, layer.values)
                        for layer in pkv.layers
                        if getattr(layer, "is_initialized", True)
                        and getattr(layer, "keys", None) is not None
                    )
                if legacy_pkv is not None:
                    branch_past_kv = tuple(
                        tuple(t[i : i + 1] for t in layer) for layer in legacy_pkv
                    )
                else:
                    branch_past_kv = pkv

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
