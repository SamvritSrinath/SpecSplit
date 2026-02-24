"""Target Engine — session-based KV-cached tree-attention verification.

The ``TargetEngine`` wraps the large, accurate language model. Given a draft
token tree from the Draft Worker, it performs a forward pass using tree
attention to verify which draft tokens are accepted under the target
distribution.

Architecture Notes:
    - **Session-based KV caching**: Maintains a ``dict[session_id, KVCacheState]``
      mapping that stores HuggingFace ``past_key_values`` per session. This
      eliminates prompt recomputation across verification rounds within the
      same generation session.
    - After verification, ``rollback_cache`` crops the KV cache back to the
      longest accepted prefix so that rejected speculative tokens do not
      pollute future forward passes.
    - Uses greedy verification: for each position in the tree, if
      ``argmax(target_logits)`` matches the drafted token, it is accepted.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specsplit.core.config import TargetWorkerConfig
from specsplit.core.telemetry import Stopwatch
from specsplit.core.verification import verify_greedy_tree, verify_stochastic_tree
from specsplit.workers.target.tree_attn import (
    bool_mask_to_float,
    build_tree_attention,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VerificationResult:
    """Result of verifying a draft tree against the target model."""

    accepted_token_ids: list[int]
    correction_token_id: int | None
    num_accepted: int
    num_draft_tokens: int = 0
    cache_hit: bool = False

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted (0.0-1.0).

        Uses ``num_draft_tokens`` (the total draft candidates presented)
        as the denominator for a meaningful acceptance rate.
        """
        if self.num_draft_tokens > 0:
            return self.num_accepted / self.num_draft_tokens
        return 0.0


@dataclass
class KVCacheState:
    """Per-session KV cache state stored on the Target Worker.

    Wraps the HuggingFace ``past_key_values`` tuple and tracks the
    sequence length so we know where to crop on rollback.

    Attributes:
        past_key_values: The HuggingFace ``past_key_values`` tuple.
            Each element is a ``(key, value)`` pair of tensors with shape
            ``(batch, num_heads, seq_len, head_dim)``.
        seq_len: The current cached sequence length (number of tokens
            whose KV projections are stored).
        next_root_logit: The logit predicting the token directly following the
            accepted sequence. Maintained to perfectly align the target model
            upon a cache hit without duplicate forwarding.
    """

    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
    seq_len: int = 0
    next_root_logit: torch.Tensor | None = None


# =============================================================================
# Helpers
# =============================================================================


def _to_legacy_cache(past_key_values: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    """Convert HuggingFace past_key_values to legacy tuple format for subscript/iteration.

    Handles legacy tuples (returned as-is), Cache API with to_legacy_cache (older
    transformers), and Cache API with .layers (e.g. DynamicCache in newer transformers
    that no longer provide to_legacy_cache).
    """
    if past_key_values is None:
        return None
    # Already legacy: tuple of (key, value) per layer
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            return past_key_values  # type: ignore[return-value]
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    # New Cache API (e.g. DynamicCache): build legacy from .layers
    if hasattr(past_key_values, "layers") and isinstance(past_key_values.layers, (list, tuple)):
        return tuple(
            (layer.keys, layer.values)
            for layer in past_key_values.layers
            if getattr(layer, "is_initialized", True) and getattr(layer, "keys", None) is not None
        )
    return None


def _cache_seq_len(past_key_values: Any) -> int:
    """Return the cached sequence length from HuggingFace past_key_values.

    Supports both the legacy tuple format (tuple of (key, value) per layer)
    and the Cache API (e.g. DynamicCache) used by newer models (Qwen2, etc.).
    """
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    legacy = _to_legacy_cache(past_key_values)
    if legacy is None or len(legacy) == 0:
        return 0
    return int(legacy[0][0].shape[2])


def _flatten_tree(draft_tree: list[dict[str, Any]]) -> tuple[list[int], list[int], list[float]]:
    """Flatten a dict-based draft tree into parallel token-ID and topology arrays.

    Performs a BFS/DFS traversal of the tree and assigns contiguous indices
    to each node.

    Args:
        draft_tree: List of root node dicts (from ``TokenNode.to_dict()``).
            Each dict has keys ``token_id``, ``log_prob``, ``children``.

    Returns:
        A ``(token_ids, topology_map)`` tuple where:
        - ``token_ids[i]`` is the token ID of tree node ``i``.
        - ``topology_map[i]`` is the parent index of node ``i``
          (``-1`` for roots).
    """
    token_ids: list[int] = []
    topology_map: list[int] = []
    log_probs: list[float] = []

    queue: deque[tuple[dict[str, Any], int]] = deque()
    for root in draft_tree:
        queue.append((root, -1))

    while queue:
        node, parent_idx = queue.popleft()
        current_idx = len(token_ids)
        token_ids.append(node["token_id"])
        topology_map.append(parent_idx)
        log_probs.append(node.get("log_prob", 0.0))

        for child in node.get("children", []):
            queue.append((child, current_idx))

    return token_ids, topology_map, log_probs


# =============================================================================
# Target Engine
# =============================================================================


class TargetEngine:
    """Session-aware tree-attention verification engine for the Target Worker.

    Maintains a dictionary of ``session_id → KVCacheState`` to reuse KV
    projections across verification rounds within the same generation
    session.  After each verification, the cache is rolled back to the
    longest accepted prefix via ``rollback_cache()``.

    Args:
        config: Target worker configuration (model name, device, etc.).
    """

    def __init__(self, config: TargetWorkerConfig | None = None) -> None:
        self.config = config or TargetWorkerConfig()
        self.device = torch.device(self.config.device)
        self._model: Any = None  # transformers.AutoModelForCausalLM
        self._tokenizer: Any = None  # transformers.AutoTokenizer
        self._is_loaded = False

        # Session → KV cache mapping (LRU ordered).
        # Concurrency: each session_id has a single lock; concurrent requests
        # for the same session are serialized to avoid cache/next_root_logit races.
        self._session_caches: OrderedDict[str, KVCacheState] = OrderedDict()
        self._session_locks: dict[str, threading.Lock] = {}

        logger.info(
            "TargetEngine initialized (model=%s, device=%s, max_sessions=%d)",
            self.config.model_name,
            self.config.device,
            self.config.max_sessions,
        )

    # --------------------------------------------------------------------- #
    # Model lifecycle
    # --------------------------------------------------------------------- #

    def load_model(self) -> None:
        """Load the target model and tokenizer via ``AutoModelForCausalLM``.

        Uses the HuggingFace ``transformers`` modular auto-class API so that
        any causal-LM architecture is supported out of the box.  The model
        is loaded in ``float16`` precision and placed on the configured
        device.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
            )
            .to(self.device)
            .eval()
        )

        self._is_loaded = True
        logger.info("Target model loaded: %s on %s", self.config.model_name, self.device)

    # --------------------------------------------------------------------- #
    # Session management
    # --------------------------------------------------------------------- #

    def get_or_create_session(self, session_id: str) -> tuple[KVCacheState, bool]:
        """Retrieve an existing session cache or create a new one.

        Args:
            session_id: Unique identifier for the generation session.

        Returns:
            A ``(cache_state, cache_hit)`` tuple.  ``cache_hit`` is True
            if an existing cache was found.
        """
        if session_id in self._session_caches:
            # Move to end to mark as recently used (LRU)
            self._session_caches.move_to_end(session_id)
            logger.debug(
                "Session cache hit: %s (seq_len=%d)",
                session_id,
                self._session_caches[session_id].seq_len,
            )
            return self._session_caches[session_id], True

        # Enforce max-sessions limit with LRU eviction
        if len(self._session_caches) >= self.config.max_sessions:
            evict_id, _ = self._session_caches.popitem(last=False)  # Remove oldest (first)
            self._session_locks.pop(evict_id, None)
            logger.warning(
                "Evicted LRU session %s (max_sessions=%d)", evict_id, self.config.max_sessions
            )

        cache = KVCacheState()
        self._session_caches[session_id] = cache
        self._session_locks[session_id] = threading.Lock()
        logger.debug("Session cache created: %s", session_id)
        return cache, False

    def end_session(self, session_id: str) -> bool:
        """Terminate a session and free its KV cache from memory.

        Args:
            session_id: The session to terminate.

        Returns:
            True if the session existed and was removed, False otherwise.
        """
        cache = self._session_caches.pop(session_id, None)
        self._session_locks.pop(session_id, None)
        if cache is not None:
            # Explicitly delete tensors to free GPU memory
            cache.past_key_values = None
            cache.next_root_logit = None
            logger.info("Session ended and cache released: %s", session_id)
            return True
        logger.debug("Session not found (already ended?): %s", session_id)
        return False

    @property
    def active_sessions(self) -> int:
        """Number of active sessions with cached KV state."""
        return len(self._session_caches)

    # --------------------------------------------------------------------- #
    # KV Cache rollback
    # --------------------------------------------------------------------- #

    def rollback_cache(self, session_id: str, accepted_depth: int) -> None:
        """Crop the KV cache back to the longest accepted prefix.

        After verification, some draft tokens may have been rejected.
        This method truncates the ``past_key_values`` tensors along the
        sequence dimension so only the accepted prefix remains, preventing
        rejected speculative tokens from polluting future forward passes.

        Args:
            session_id: The session whose cache to roll back.
            accepted_depth: The number of tokens in the accepted prefix
                (measured from the start of the full cached sequence).
                The cache will be cropped to exactly this length.

        Raises:
            KeyError: If the session does not exist.
            ValueError: If ``accepted_depth`` exceeds the current cache length.
        """
        if session_id not in self._session_caches:
            raise KeyError(f"No active session with id={session_id!r}")

        cache = self._session_caches[session_id]

        if cache.past_key_values is None:
            logger.debug("Rollback no-op: session %s has no cached KV state", session_id)
            cache.seq_len = 0
            return

        if accepted_depth > cache.seq_len:
            raise ValueError(
                f"accepted_depth={accepted_depth} exceeds cached seq_len={cache.seq_len} "
                f"for session {session_id!r}"
            )

        if accepted_depth == cache.seq_len:
            logger.debug("Rollback no-op: accepted_depth == seq_len (%d)", accepted_depth)
            return

        # Convert to legacy tuple if needed (DynamicCache, etc.), then crop in place.
        # Using slice views WITHOUT .contiguous() avoids O(n) reallocation.
        # The sliced tensors are views into the original buffer — no copy.
        # HuggingFace attention handles non-contiguous past_key_values correctly.
        # For a full zero-copy pre-allocated approach, see StaticKVCache in kv_cache.py.
        rolled_back: list[tuple[torch.Tensor, torch.Tensor]] = []
        past_kv = _to_legacy_cache(cache.past_key_values)
        if past_kv is None:
            cache.past_key_values = None
            cache.seq_len = 0
            return
        for layer_key, layer_value in past_kv:
            rolled_back.append(
                (
                    layer_key[:, :, :accepted_depth, :],
                    layer_value[:, :, :accepted_depth, :],
                )
            )
        cache.past_key_values = tuple(rolled_back)

        old_len = cache.seq_len
        cache.seq_len = accepted_depth

        logger.debug(
            "KV cache rolled back: session=%s, %d → %d tokens (zero-copy slicing)",
            session_id,
            old_len,
            accepted_depth,
        )

    # --------------------------------------------------------------------- #
    # Verification
    # --------------------------------------------------------------------- #

    def verify_draft_tree(
        self,
        prompt_ids: list[int],
        draft_tree: list[dict[str, Any]],
        session_id: str | None = None,
        temperature: float = 0.0,  # <-- Added temperature
    ) -> VerificationResult:
        """Verify a draft token tree against the target model's distribution.

        When a ``session_id`` is provided, the engine reuses (or creates) a
        KV cache for that session.  After verification, the cache is
        automatically rolled back to the accepted prefix.

        The verification pipeline:
            1. Flatten the draft tree into ``(token_ids, topology_map)``.
            2. Build a tree-attention mask via ``build_tree_attention()``.
            3. Forward pass the tree tokens through the target model
               with cached KV state (prefix is not recomputed on cache hit).
            4. Run ``verify_greedy_tree()`` on the tree logits.
            5. Roll back the KV cache to the accepted prefix.

        Args:
            prompt_ids: Original prompt token IDs.
            draft_tree: Draft tree as a list of dicts (from ``TokenNode.to_dict()``).
                Each dict has keys ``token_id``, ``log_prob``, ``children``.
            session_id: Optional session ID for KV cache reuse.  If ``None``,
                verification is fully stateless (no caching).

        Returns:
            A ``VerificationResult`` with accepted tokens, an optional
            correction token, and a ``cache_hit`` flag.
        """
        sw = Stopwatch()
        sw.start()

        if not self._is_loaded or self._model is None:
            raise RuntimeError("Target model not loaded. Call load_model() first.")

        # --- Session cache lookup ---
        cache_hit = False
        cache_state: KVCacheState | None = None
        session_lock: threading.Lock | None = None
        if session_id is not None:
            cache_state, cache_hit = self.get_or_create_session(session_id)
            session_lock = self._session_locks[session_id]
            session_lock.acquire()
        try:
            # --- Handle empty tree (no draft tokens) ---
            if not draft_tree:
                sw.stop()
                return VerificationResult(
                    accepted_token_ids=[],
                    correction_token_id=None,
                    num_accepted=0,
                    num_draft_tokens=0,
                    cache_hit=cache_hit,
                )

            # --- Step 1: Flatten tree → (token_ids, topology_map) ---
            flat_token_ids, topology_map, flat_log_probs = _flatten_tree(draft_tree)
            num_tree_nodes = len(flat_token_ids)

            # --- Step 2: Determine what to feed the model ---
            past_kv = cache_state.past_key_values if cache_state else None
            prefix_length = cache_state.seq_len if cache_state and cache_hit else 0

            if cache_hit and prefix_length > 0:
                # Cache hit: only feed the new draft tokens.
                # We explicitly DO NOT append prompt_ids[-1] to avoid duplicate
                # tokens corrupting the target KV Cache.
                input_token_ids = flat_token_ids
            else:
                # No cache (or first call): feed prompt + draft tokens
                input_token_ids = prompt_ids + flat_token_ids
                prefix_length = len(prompt_ids)
                past_kv = None  # Don't use stale cache

            input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=self.device)
            # Shape: [1, input_len]

            # --- Step 3: Build tree-attention mask ---
            attn_mask_bool, position_ids = build_tree_attention(
                topology_map=topology_map,
                prefix_length=prefix_length,
                device=self.device,
            )
            # attn_mask_bool shape: [1, 1, total_len, total_len]
            # position_ids shape: [1, total_len]

            # When using KV cache, we only need the query rows for the
            # new tokens (the prefix is already cached)
            if past_kv is not None:
                # Slice to only the new input tokens' rows
                new_len = len(input_token_ids)
                attn_mask_bool = attn_mask_bool[:, :, -new_len:, :]
                position_ids = position_ids[:, -new_len:]

            # Convert bool mask → float mask (0.0 / -inf) for HF compatibility
            attn_mask_float = bool_mask_to_float(attn_mask_bool, dtype=self._model.dtype)

            # --- Step 4: Forward pass ---
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attn_mask_float,
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            # --- Step 5: Update session cache ---
            new_past_kv = _to_legacy_cache(outputs.past_key_values)
            new_seq_len = int(new_past_kv[0][0].shape[2]) if new_past_kv else 0

            if cache_state is not None:
                cache_state.past_key_values = new_past_kv
                cache_state.seq_len = new_seq_len

            # --- Step 6: Extract tree-position logits and verify ---
            # all_logits[j] predicts the token at position j+1 (causal LM semantics).
            # Tree nodes are in BFS order in the input; node i is at input position
            # (base + i) where base=0 on cache hit (tree only) else prefix_length.
            all_logits = outputs.logits  # [1, input_len, vocab_size]

            tree_logits = torch.zeros(
                num_tree_nodes, all_logits.shape[-1],
                dtype=all_logits.dtype,
                device=self.device,
            )
            # Shape: [num_tree_nodes, vocab_size]

            base = 0 if cache_hit else prefix_length
            for i in range(num_tree_nodes):
                if i == 0:
                    # Root: logits that predict the first tree token
                    if cache_hit and prefix_length > 0 and cache_state.next_root_logit is not None:
                        tree_logits[i] = cache_state.next_root_logit
                    elif prefix_length > 0:
                        tree_logits[i] = all_logits[0, prefix_length - 1, :]
                    else:
                        tree_logits[i] = all_logits[0, 0, :]
                else:
                    # Child at node i: predicted by logits at parent's position.
                    # For linear chains parent == i-1, but for branching trees
                    # siblings share the same parent, so we look up the topology.
                    parent = topology_map[i]
                    tree_logits[i] = all_logits[0, base + parent, :]

            draft_tokens_tensor = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)

            # --- Task 3.3: Speculative Acceptance Routing ---
            if temperature > 0.0:
                # Convert target logits to probabilities
                target_probs = torch.softmax(tree_logits / temperature, dim=-1)
                # Convert draft log_probs back to probabilities
                draft_probs = torch.exp(torch.tensor(flat_log_probs, dtype=torch.float32, device=self.device))
                result_data = verify_stochastic_tree(
                    draft_tokens=draft_tokens_tensor,
                    draft_probs=draft_probs,
                    target_probs=target_probs,
                    topology_map=topology_map,
                )
            else:
                # Fallback to pure greedy verification
                result_data = verify_greedy_tree(
                    draft_tokens=draft_tokens_tensor,
                    target_logits=tree_logits,
                    topology_map=topology_map,
                )

            # --- Step 7: Build VerificationResult ---
            accepted_ids = result_data.accepted_tokens
            correction = result_data.bonus_token if result_data.bonus_token >= 0 else None
            num_accepted = result_data.num_accepted
            leaf_idx = result_data.accepted_leaf_index

            # When the full path to a leaf was accepted, verify_greedy_tree returns
            # bonus = argmax(tree_logits[leaf]) = the leaf token (duplicate). The
            # correct bonus is the next token after the leaf: argmax at position base+leaf_idx.
            if (
                num_accepted > 0
                and leaf_idx >= 0
                and result_data.bonus_token == accepted_ids[-1]
            ):
                correction = all_logits[0, base + leaf_idx, :].argmax(dim=-1).item()

            # --- Step 8: Rollback cache to accepted prefix and update logic states ---
            if session_id is not None and cache_state is not None:
                accepted_total = prefix_length + num_accepted
                if accepted_total <= cache_state.seq_len:
                    self.rollback_cache(session_id, accepted_total)

                if correction is not None and correction >= 0:
                    correction_input = torch.tensor([[correction]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        correction_output = self._model(
                            input_ids=correction_input,
                            past_key_values=cache_state.past_key_values,
                            use_cache=True,
                        )
                    corr_past_kv = _to_legacy_cache(correction_output.past_key_values)
                    cache_state.past_key_values = corr_past_kv
                    cache_state.seq_len = (
                        int(corr_past_kv[0][0].shape[2]) if corr_past_kv else 0
                    )
                    cache_state.next_root_logit = correction_output.logits[0, -1, :].detach().clone()
                else:
                    if num_accepted > 0:
                        if cache_hit:
                            cache_state.next_root_logit = all_logits[0, leaf_idx, :].detach().clone()
                        else:
                            cache_state.next_root_logit = all_logits[0, prefix_length + leaf_idx, :].detach().clone()

            sw.stop()
            return VerificationResult(
                accepted_token_ids=accepted_ids,
                correction_token_id=correction,
                num_accepted=num_accepted,
                num_draft_tokens=num_tree_nodes,
                cache_hit=cache_hit,
            )
        finally:
            if session_lock is not None:
                session_lock.release()
    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._is_loaded
