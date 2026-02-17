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
    - Uses rejection sampling: for each position in the tree, if
      ``p_target(x) >= p_draft(x)``, the token is accepted; otherwise
      it is accepted with probability ``p_target(x) / p_draft(x)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from specsplit.core.config import TargetWorkerConfig
from specsplit.core.telemetry import Stopwatch

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
    cache_hit: bool = False

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted (0.0–1.0)."""
        total = self.num_accepted + (1 if self.correction_token_id is not None else 0)
        return self.num_accepted / total if total > 0 else 0.0


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
    """

    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
    seq_len: int = 0


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

        # Session → KV cache mapping
        self._session_caches: dict[str, KVCacheState] = {}

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
        any causal-LM architecture is supported out of the box.

        .. todo::
            Uncomment the real loading code once the environment has
            ``transformers`` and a model downloaded.
        """
        # TODO(model-loading): Uncomment for production:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        #
        # self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     self.config.model_name,
        #     torch_dtype=torch.float16,
        #     device_map=self.config.device,
        # ).eval()
        self._is_loaded = True
        logger.info("Target model loaded: %s", self.config.model_name)

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
            logger.debug("Session cache hit: %s (seq_len=%d)", session_id,
                         self._session_caches[session_id].seq_len)
            return self._session_caches[session_id], True

        # Enforce max-sessions limit with LRU eviction
        if len(self._session_caches) >= self.config.max_sessions:
            evict_id = next(iter(self._session_caches))
            self.end_session(evict_id)
            logger.warning("Evicted oldest session %s (max_sessions=%d)",
                           evict_id, self.config.max_sessions)

        cache = KVCacheState()
        self._session_caches[session_id] = cache
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
        if cache is not None:
            # Explicitly delete tensors to free GPU memory
            cache.past_key_values = None
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

        # Crop each layer's (key, value) tensors along the seq_len dimension (dim=2)
        # HuggingFace past_key_values shape: (key, value) each (batch, heads, seq, head_dim)
        rolled_back: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_key, layer_value in cache.past_key_values:
            rolled_back.append((
                layer_key[:, :, :accepted_depth, :].contiguous(),
                layer_value[:, :, :accepted_depth, :].contiguous(),
            ))

        cache.past_key_values = tuple(rolled_back)
        old_len = cache.seq_len
        cache.seq_len = accepted_depth

        logger.debug(
            "KV cache rolled back: session=%s, %d → %d tokens",
            session_id, old_len, accepted_depth,
        )

    # --------------------------------------------------------------------- #
    # Verification
    # --------------------------------------------------------------------- #

    def verify_draft_tree(
        self,
        prompt_ids: list[int],
        draft_tree: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> VerificationResult:
        """Verify a draft token tree against the target model's distribution.

        When a ``session_id`` is provided, the engine reuses (or creates) a
        KV cache for that session.  After verification, the cache is
        automatically rolled back to the accepted prefix.

        Args:
            prompt_ids: Original prompt token IDs.
            draft_tree: Draft tree as a list of dicts (from ``TokenNode.to_dict()``).
                Each dict has keys ``token_id``, ``log_prob``, ``children``.
            session_id: Optional session ID for KV cache reuse.  If ``None``,
                verification is fully stateless (no caching).

        Returns:
            A ``VerificationResult`` with accepted tokens, an optional
            correction token, and a ``cache_hit`` flag.

        .. todo::
            Implement tree-attention forward pass and rejection sampling.
            The production version should:
            1. Build a tree-attention mask from ``draft_tree``.
            2. Concatenate prompt + draft token IDs into a single input.
            3. Forward through the model with ``past_key_values`` from the
               session cache (if available).
            4. Apply rejection sampling per-position.
            5. Call ``rollback_cache()`` to crop to accepted prefix.
        """
        sw = Stopwatch()
        sw.start()

        # --- Session cache lookup ---
        cache_hit = False
        cache_state: KVCacheState | None = None

        if session_id is not None:
            cache_state, cache_hit = self.get_or_create_session(session_id)

        # --- Flatten the tree into the first-branch path (stub) ---
        # TODO(verification): Replace with real tree-attention forward pass.
        #
        # Production implementation outline:
        #
        #   # 1. Build input_ids for the tree
        #   new_token_ids = _flatten_tree_first_branch(draft_tree)
        #   input_ids = torch.tensor([new_token_ids], device=self.device)
        #
        #   # 2. Forward with cached KV
        #   with torch.no_grad():
        #       outputs = self._model(
        #           input_ids=input_ids,
        #           past_key_values=cache_state.past_key_values if cache_state else None,
        #           use_cache=True,
        #       )
        #
        #   # 3. Update the session cache
        #   if cache_state is not None:
        #       cache_state.past_key_values = outputs.past_key_values
        #       cache_state.seq_len = outputs.past_key_values[0][0].shape[2]
        #
        #   # 4. Rejection sampling → accepted_ids, correction_id
        #   ...
        #
        #   # 5. Rollback to accepted prefix
        #   if session_id is not None:
        #       prompt_len = len(prompt_ids) if not cache_hit else cache_state.seq_len
        #       self.rollback_cache(session_id, prompt_len + len(accepted_ids))

        # Stub: accept all tokens from the first branch
        accepted: list[int] = []
        node_list = draft_tree
        while node_list:
            node = node_list[0]  # Follow the first branch
            accepted.append(node["token_id"])
            node_list = node.get("children", [])

        correction = None  # Stub: no correction needed

        # Stub: simulate cache update
        if cache_state is not None:
            cache_state.seq_len = len(prompt_ids) + len(accepted)

        sw.stop()
        logger.debug(
            "Verification complete: session=%s, cache_hit=%s, accepted=%d, elapsed=%.3f ms",
            session_id,
            cache_hit,
            len(accepted),
            sw.elapsed_ms,
        )

        return VerificationResult(
            accepted_token_ids=accepted,
            correction_token_id=correction,
            num_accepted=len(accepted),
            cache_hit=cache_hit,
        )

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._is_loaded
