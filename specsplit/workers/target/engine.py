"""Target Engine — session-based KV-cached tree-attention verification.

The ``TargetEngine`` wraps the large, accurate language model. Given a draft
token tree from the Draft Worker, it performs a forward pass using tree
attention to verify which draft tokens are accepted under the target
distribution.

Architecture Notes:
    - **Session-based KV caching**: Maintains a ``dict[session_id, KVCacheState]``
      mapping that stores pre-allocated :class:`StaticKVCache` per session.
      This eliminates prompt recomputation and ``torch.cat`` reallocation
      across verification rounds within the same generation session.
    - After verification, ``rollback_cache`` uses ``StaticKVCache.rollback()``
      (O(1) pointer update) for linear chains, or ``StaticKVCache.compact()``
      for branching trees, to crop the cache to the accepted prefix.
    - Uses greedy verification: for each position in the tree, if
      ``argmax(target_logits)`` matches the drafted token, it is accepted.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specsplit.core.cache_utils import legacy_to_dynamic_cache
from specsplit.core.config import TargetWorkerConfig
from specsplit.core.telemetry import Stopwatch
from specsplit.core.verification import (
    verify_greedy_tree,
    verify_stochastic_tree,
)
from specsplit.workers.target.kv_cache import StaticKVCache
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

    Wraps a :class:`StaticKVCache` providing O(1) rollback and
    zero-copy slice-assignment appends.

    Attributes:
        cache: The pre-allocated static KV cache for this session,
            or ``None`` if the session was created before the model
            was loaded.
        seq_len: The current cached sequence length (number of tokens
            whose KV projections are stored).
        next_root_logit: The logit predicting the token directly following the
            accepted sequence. Maintained to perfectly align the target model
            upon a cache hit without duplicate forwarding.
    """

    cache: StaticKVCache | None = None
    seq_len: int = 0
    next_root_logit: torch.Tensor | None = None
    last_accessed: float = 0.0  # D3: timestamp for TTL garbage collection


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

    The KV cache uses :class:`StaticKVCache` — a pre-allocated, pointer-
    based cache that provides O(1) rollback and zero-copy appends, compared
    to HuggingFace's default ``DynamicCache`` which requires ``torch.cat``
    on every append.

    Args:
        config: Target worker configuration (model name, device, etc.).
    """

    def __init__(self, config: TargetWorkerConfig | None = None) -> None:
        self.config = config or TargetWorkerConfig()
        self.device = torch.device(self.config.device)
        self._model: Any = None  # transformers.AutoModelForCausalLM
        self._tokenizer: Any = None  # transformers.AutoTokenizer
        self._is_loaded = False
        self._use_4d_mask = True  # D2: set to False when FA2 is detected

        # Session → KV cache mapping (LRU ordered).
        # Concurrency: each session_id has a single lock; concurrent requests
        # for the same session are serialized to avoid cache/next_root_logit races.
        self._session_caches: OrderedDict[str, KVCacheState] = OrderedDict()
        self._session_locks: dict[str, threading.Lock] = {}

        # D3: Session TTL garbage collection
        self._session_ttl_seconds = self.config.session_ttl_seconds
        self._ttl_thread = threading.Thread(
            target=self._ttl_gc_loop,
            daemon=True,
            name="target-engine-ttl-gc",
        )
        self._ttl_thread.start()

        logger.info(
            "TargetEngine initialized (model=%s, device=%s, max_sessions=%d, ttl=%.0fs)",
            self.config.model_name,
            self.config.device,
            self.config.max_sessions,
            self._session_ttl_seconds,
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

        # D2: Detect Flash Attention 2 — 4D custom masks are incompatible
        attn_impl = getattr(self._model.config, '_attn_implementation', None)
        if attn_impl == 'flash_attention_2':
            self._use_4d_mask = False
            logger.warning(
                "Flash Attention 2 detected — disabling 4D tree attention mask. "
                "Position IDs will still encode tree structure, but the custom "
                "attention pattern will not be enforced."
            )

        logger.info("Target model loaded: %s on %s", self.config.model_name, self.device)

    # --------------------------------------------------------------------- #
    # StaticKVCache factory
    # --------------------------------------------------------------------- #

    def _create_static_cache(self) -> StaticKVCache:
        """Create a StaticKVCache initialized from the loaded model's config.

        Extracts ``num_hidden_layers``, ``num_key_value_heads`` (for GQA models)
        or ``num_attention_heads``, ``hidden_size``, and ``max_position_embeddings``
        from the model config to size the pre-allocated buffers.

        Returns:
            A fresh :class:`StaticKVCache` sized for this model.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Cannot create StaticKVCache before model is loaded.")

        cfg = self._model.config
        num_layers = cfg.num_hidden_layers
        # GQA models (Llama, Qwen2, Mistral) use num_key_value_heads
        num_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        max_seq_len = getattr(cfg, "max_position_embeddings", 4096)

        return StaticKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            batch_size=1,
            dtype=torch.float16,
            device=self.device,
        )

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
            cache = self._session_caches[session_id]
            cache.last_accessed = time.time()  # D3: touch timestamp
            logger.debug(
                "Session cache hit: %s (seq_len=%d)",
                session_id,
                cache.seq_len,
            )
            return cache, True

        # Enforce max-sessions limit with LRU eviction
        if len(self._session_caches) >= self.config.max_sessions:
            evict_id, _ = self._session_caches.popitem(last=False)  # Remove oldest (first)
            self._session_locks.pop(evict_id, None)
            logger.warning(
                "Evicted LRU session %s (max_sessions=%d)", evict_id, self.config.max_sessions
            )

        # Create a new session with a StaticKVCache
        static_cache = self._create_static_cache() if self._is_loaded else None
        cache = KVCacheState(cache=static_cache, last_accessed=time.time())
        self._session_caches[session_id] = cache
        self._session_locks[session_id] = threading.Lock()
        logger.debug("Session cache created: %s", session_id)
        return cache, False

    def end_session(self, session_id: str) -> bool:
        """Terminate a session and free its KV cache from memory.

        Thread-safe: acquires the session lock before destroying the cache
        to ensure no active ``verify_draft_tree`` thread is using the
        tensors being deleted (Issue 10).

        Args:
            session_id: The session to terminate.

        Returns:
            True if the session existed and was removed, False otherwise.
        """
        lock = self._session_locks.get(session_id)
        if lock is not None:
            lock.acquire()
        try:
            cache = self._session_caches.pop(session_id, None)
            if cache is not None:
                # Explicitly delete tensors to free GPU memory
                cache.cache = None
                cache.next_root_logit = None
                logger.info("Session ended and cache released: %s", session_id)
                return True
            logger.debug("Session not found (already ended?): %s", session_id)
            return False
        finally:
            if lock is not None:
                lock.release()
            self._session_locks.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        """Number of active sessions with cached KV state."""
        return len(self._session_caches)

    def _ttl_gc_loop(self) -> None:
        """Background loop that purges stale sessions every 60 seconds."""
        while True:
            time.sleep(60)
            try:
                self.purge_stale_sessions()
            except Exception:
                logger.exception("Error during TTL session purge")

    def purge_stale_sessions(self, ttl_seconds: float | None = None) -> int:
        """Remove sessions that have not been accessed within the TTL.

        Args:
            ttl_seconds: Override the configured TTL (for testing).

        Returns:
            Number of purged sessions.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._session_ttl_seconds
        now = time.time()
        stale_ids = [
            sid for sid, cache in self._session_caches.items()
            if (now - cache.last_accessed) > ttl
        ]
        for sid in stale_ids:
            cache = self._session_caches.pop(sid, None)
            self._session_locks.pop(sid, None)
            if cache is not None:
                cache.cache = None
                cache.next_root_logit = None
        if stale_ids:
            logger.info("Purged %d stale sessions (ttl=%.0fs): %s", len(stale_ids), ttl, stale_ids)
        return len(stale_ids)

    # --------------------------------------------------------------------- #
    # KV Cache rollback
    # --------------------------------------------------------------------- #

    def rollback_cache(
        self,
        session_id: str,
        accepted_depth: int,
        accepted_tree_indices: list[int] | None = None,
        prefix_length: int = 0,
    ) -> None:
        """Compact the KV cache to only the accepted prefix + accepted path.

        Uses :class:`StaticKVCache` for O(1) rollback (linear chains)
        or ``compact()`` (branching trees with non-contiguous accepted
        positions).

        Args:
            session_id: The session whose cache to roll back.
            accepted_depth: Total number of tokens in the accepted prefix
                (measured from the start of the full cached sequence).
                Used for simple O(1) rollback when ``accepted_tree_indices``
                is None.
            accepted_tree_indices: BFS indices (0-based into the tree
                portion of the input) of the accepted path nodes, in
                order from root to leaf.  When provided, the cache is
                compacted to ``prefix[:prefix_length] + tree[accepted_tree_indices]``.
            prefix_length: Length of the prompt/prefix portion of the
                cached sequence.  Only used when ``accepted_tree_indices``
                is provided.

        Raises:
            KeyError: If the session does not exist.
            ValueError: If ``accepted_depth`` exceeds the current cache length.
        """
        if session_id not in self._session_caches:
            raise KeyError(f"No active session with id={session_id!r}")

        cache_state = self._session_caches[session_id]

        if cache_state.cache is None:
            logger.debug("Rollback no-op: session %s has no cached KV state", session_id)
            cache_state.seq_len = 0
            return

        if accepted_depth > cache_state.seq_len:
            raise ValueError(
                f"accepted_depth={accepted_depth} exceeds cached seq_len={cache_state.seq_len} "
                f"for session {session_id!r}"
            )

        if accepted_depth == cache_state.seq_len and accepted_tree_indices is None:
            logger.debug("Rollback no-op: accepted_depth == seq_len (%d)", accepted_depth)
            return

        if accepted_tree_indices is not None and len(accepted_tree_indices) > 0:
            # --- Branching tree compaction via StaticKVCache.compact() ---
            prefix_indices = list(range(prefix_length))
            tree_global_indices = [prefix_length + i for i in accepted_tree_indices]
            keep_indices = prefix_indices + tree_global_indices

            cache_state.cache.compact(keep_indices)
            new_seq_len = len(keep_indices)
        else:
            # --- O(1) pointer-based rollback for linear chains ---
            cache_state.cache.rollback(accepted_depth)
            new_seq_len = accepted_depth

        old_len = cache_state.seq_len
        cache_state.seq_len = new_seq_len

        logger.debug(
            "KV cache rolled back: session=%s, %d → %d tokens%s",
            session_id,
            old_len,
            new_seq_len,
            " (compact)" if accepted_tree_indices else " (O(1) rollback)",
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
            # Extract cached KV from StaticKVCache if available
            past_kv = None
            prefix_length = cache_state.seq_len if cache_state and cache_hit else 0

            if cache_hit and prefix_length > 0 and cache_state and cache_state.cache is not None:
                # Cache hit: get KV tensors from StaticKVCache
                past_kv = cache_state.cache.get_all_kv()
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

            # D2: Flash Attention 2 is incompatible with 4D custom masks.
            # When FA2 is active, skip the mask entirely (position_ids still
            # encode tree structure for positional encoding).
            if self._use_4d_mask:
                attn_mask_float = bool_mask_to_float(attn_mask_bool, dtype=self._model.dtype)
            else:
                attn_mask_float = None

            if past_kv is not None and isinstance(past_kv, tuple):
                past_kv = legacy_to_dynamic_cache(past_kv, self._model.config)

            # --- Step 4: Forward pass ---
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attn_mask_float,
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            # --- Step 5: Update session cache with StaticKVCache ---
            if cache_state is not None and cache_state.cache is not None:
                # Convert HF output to 5D tensors and append to StaticKVCache
                hf_past_kv = _to_legacy_cache(outputs.past_key_values)
                if hf_past_kv is not None:
                    # The HF model returns the FULL past_key_values including
                    # what we fed in. We need only the NEW positions.
                    full_seq_len = hf_past_kv[0][0].shape[2]
                    new_start = cache_state.cache.seq_len
                    new_token_count = full_seq_len - new_start

                    if new_token_count > 0:
                        # Extract only the new KV entries
                        new_keys_list = [layer[0][:, :, new_start:, :] for layer in hf_past_kv]
                        new_values_list = [layer[1][:, :, new_start:, :] for layer in hf_past_kv]
                        new_keys = torch.stack(new_keys_list, dim=0)
                        new_values = torch.stack(new_values_list, dim=0)
                        cache_state.cache.append(new_keys, new_values)

                    cache_state.seq_len = cache_state.cache.seq_len
                else:
                    cache_state.seq_len = 0

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
                if topology_map[i] == -1:
                    # Root node: logits that predict the first tree token.
                    # Fix A2: ALL roots (not just i==0) must get the prompt's
                    # last-position logit, not index into all_logits[0, -1, :].
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

            # Fix A3 / Bug 2: When the full path to a leaf was accepted
            # (diverged=False), the bonus token from tree_logits[leaf] predicts
            # the LEAF token itself (duplicate). The correct bonus is the NEXT
            # token after the leaf, sampled from all_logits[base + leaf].
            # Using the explicit `diverged` flag instead of the old hacky
            # comparison (bonus_token == accepted_ids[-1]) which breaks
            # under stochastic sampling.
            if not result_data.diverged and num_accepted > 0 and leaf_idx >= 0:
                next_logits = all_logits[0, base + leaf_idx, :]
                if temperature > 0.0:
                    next_probs = torch.softmax(next_logits / temperature, dim=-1)
                    correction = torch.multinomial(next_probs, 1).item()
                else:
                    correction = next_logits.argmax(dim=-1).item()

            # --- Step 8: Rollback cache to accepted prefix and update logic states ---
            if session_id is not None and cache_state is not None:
                accepted_total = prefix_length + num_accepted
                # Fix A1: Extract accepted_indices from verification result
                # for proper KV cache compaction on branching trees.
                accepted_tree_indices = getattr(result_data, 'accepted_indices', None)
                if accepted_total <= cache_state.seq_len:
                    self.rollback_cache(
                        session_id,
                        accepted_total,
                        accepted_tree_indices=accepted_tree_indices,
                        prefix_length=prefix_length,
                    )

                if correction is not None and correction >= 0:
                    correction_input = torch.tensor([[correction]], dtype=torch.long, device=self.device)

                    # FIX: Wrap the correction cache as well
                    corr_kv = cache_state.cache.get_all_kv() if cache_state.cache else None
                    if corr_kv is not None and isinstance(corr_kv, tuple):
                        corr_kv = legacy_to_dynamic_cache(corr_kv, self._model.config)

                    with torch.no_grad():
                        correction_output = self._model(
                            input_ids=correction_input,
                            past_key_values=corr_kv,
                            use_cache=True,
                        )
                    # Append correction token's KV to the StaticKVCache
                    corr_hf = _to_legacy_cache(correction_output.past_key_values)
                    if corr_hf is not None and cache_state.cache is not None:
                        corr_full_len = corr_hf[0][0].shape[2]
                        corr_start = cache_state.cache.seq_len
                        corr_new = corr_full_len - corr_start
                        if corr_new > 0:
                            corr_keys = torch.stack(
                                [layer[0][:, :, corr_start:, :] for layer in corr_hf], dim=0
                            )
                            corr_values = torch.stack(
                                [layer[1][:, :, corr_start:, :] for layer in corr_hf], dim=0
                            )
                            cache_state.cache.append(corr_keys, corr_values)
                        cache_state.seq_len = cache_state.cache.seq_len
                    cache_state.next_root_logit = correction_output.logits[0, -1, :].detach().clone()
                else:
                    if num_accepted > 0:
                        if cache_hit:
                            cache_state.next_root_logit = all_logits[0, leaf_idx, :].detach().clone()
                        else:
                            cache_state.next_root_logit = all_logits[0, prefix_length + leaf_idx, :].detach().clone()
                    else:
                        # Fix Bug 2: Total rejection (num_accepted==0, correction is None).
                        # On cache hit, all_logits has only tree rows [0..num_tree_nodes-1];
                        # row 0 predicts what follows the first tree token (root). Use it
                        # so the next round has valid next_root_logit (avoiding OOB when
                        # next_root_logit was None and elif prefix_length>0 used wrong index).
                        if cache_hit:
                            cache_state.next_root_logit = all_logits[0, 0, :].detach().clone()
                        elif prefix_length > 0:
                            cache_state.next_root_logit = all_logits[
                                0, prefix_length - 1, :
                            ].detach().clone()
                        else:
                            cache_state.next_root_logit = None

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
