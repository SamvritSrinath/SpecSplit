"""Vocabulary alignment utilities for heterogeneous model configs."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VocabBridge:
    """Remaps token IDs when Draft and Target models have different vocabularies.

    In heterogeneous speculative decoding (e.g., matching a smaller draft model
    from one family with a larger target from another), the token integer IDs
    often do not perfectly align, even if the underlying string space is similar.
    This bridge converts Draft tokens to Target tokens before verification, and
    Target tokens back to Draft tokens when a correction is issued.

    Args:
        draft_tokenizer: The tokenizer matching the draft model.
        target_tokenizer: The tokenizer matching the target model.
    """

    def __init__(self, draft_tokenizer: Any, target_tokenizer: Any) -> None:
        self.draft_tokenizer = draft_tokenizer
        self.target_tokenizer = target_tokenizer

        # Vocab boundaries
        self.draft_vocab_size = len(draft_tokenizer)
        self.target_vocab_size = len(target_tokenizer)

        logger.info(
            "VocabBridge initialized: draft_vocab=%d, target_vocab=%d",
            self.draft_vocab_size,
            self.target_vocab_size,
        )

        # Precompute mappings for token IDs where strings match exactly
        self._draft_to_target: dict[int, int] = {}
        self._target_to_draft: dict[int, int] = {}
        self._build_mappings()

    def _build_mappings(self) -> None:
        """Heuristically build string-based exact-match mappings."""
        # For a full production implementation, one would iterate over
        # draft_tokenizer.get_vocab() and map strings to the target tokenizer.
        draft_vocab = self.draft_tokenizer.get_vocab()
        
        # Determine fallback tokens for out-of-bounds IDs
        self.target_unk_token = getattr(self.target_tokenizer, "unk_token_id", 0)
        if self.target_unk_token is None:
            self.target_unk_token = 0

        self.draft_unk_token = getattr(self.draft_tokenizer, "unk_token_id", 0)
        if self.draft_unk_token is None:
            self.draft_unk_token = 0
            
        # Map by token string
        for token_text, draft_id in draft_vocab.items():
            target_id = self.target_tokenizer.convert_tokens_to_ids(token_text)
            if target_id is not None and target_id != self.target_unk_token:
                self._draft_to_target[draft_id] = target_id
                self._target_to_draft[target_id] = draft_id
            else:
                self._draft_to_target[draft_id] = self.target_unk_token
        
        # Ensure minimum identity fallback
        min_vocab = min(self.draft_vocab_size, self.target_vocab_size)
        for i in range(min_vocab):
            if i not in self._draft_to_target:
                self._draft_to_target[i] = i
            if i not in self._target_to_draft:
                self._target_to_draft[i] = i

    def draft_to_target_id(self, draft_id: int) -> int:
        """Map a draft token ID to its closest target token ID."""
        return self._draft_to_target.get(draft_id, self.target_unk_token)

    def target_to_draft_id(self, target_id: int) -> int:
        """Map a target token ID back to the draft token ID."""
        return self._target_to_draft.get(target_id, self.draft_unk_token)
