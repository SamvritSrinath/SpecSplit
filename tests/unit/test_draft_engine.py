"""Unit tests for the DraftEngine (mocked, no real model loading)."""

from __future__ import annotations

from specsplit.core.config import DraftWorkerConfig
from specsplit.workers.draft.engine import DraftEngine, TokenNode


class TestDraftEngine:
    """Tests for DraftEngine initialization and stub generation."""

    def test_initialization(self, draft_config: DraftWorkerConfig):
        """Engine should initialize without loading a model."""
        engine = DraftEngine(config=draft_config)
        assert not engine.is_loaded
        assert engine.config.device == "cpu"

    def test_generate_draft_tree_stub(self, draft_config: DraftWorkerConfig):
        """Stub generation should produce the correct tree structure."""
        engine = DraftEngine(config=draft_config)

        roots = engine.generate_draft_tree(
            prompt_ids=[1, 2, 3],
            k=3,
            num_beams=2,
        )

        # Should have `num_beams` root nodes
        assert len(roots) == 2
        assert all(isinstance(r, TokenNode) for r in roots)

        # Each root should have a chain of depth k-1 children
        node = roots[0]
        depth = 0
        while node.children:
            assert len(node.children) >= 1
            node = node.children[0]
            depth += 1
        assert depth == 2  # k=3, so k-1=2 levels of children

    def test_to_dict_serialization(self, draft_config: DraftWorkerConfig):
        """TokenNode.to_dict() should produce a serializable structure."""
        engine = DraftEngine(config=draft_config)
        roots = engine.generate_draft_tree(prompt_ids=[1], k=2, num_beams=1)

        d = roots[0].to_dict()
        assert "token_id" in d
        assert "log_prob" in d
        assert "children" in d
        assert isinstance(d["children"], list)

    def test_reset_cache(self, draft_config: DraftWorkerConfig):
        """reset_cache should not raise."""
        engine = DraftEngine(config=draft_config)
        engine.reset_cache()  # Should not raise
