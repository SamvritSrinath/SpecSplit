"""Unit tests for the DraftEngine (mocked model, no real downloads)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from specsplit.core.config import DraftWorkerConfig
from specsplit.workers.draft.engine import DraftEngine, TokenNode


def _make_mock_model(vocab_size: int = 50, deterministic_token: int = 42) -> MagicMock:
    """Create a mock causal LM that always predicts ``deterministic_token``.

    The mock model's ``.forward()`` returns an object with:
    - ``.logits``: shape ``[1, seq_len, vocab_size]`` where argmax → deterministic_token
    - ``.past_key_values``: a 2-layer fake KV cache tuple
    """
    model = MagicMock()
    model.dtype = torch.float16
    model.to.return_value = model
    model.eval.return_value = model

    def fake_forward(input_ids, past_key_values=None, use_cache=True, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, vocab_size)
        logits[:, :, deterministic_token] = 10.0  # argmax → deterministic_token

        # Fake KV cache: 2 layers
        cache_len = seq_len
        if past_key_values is not None:
            cache_len += past_key_values[0][0].shape[2]
        fake_kv = tuple(
            (torch.zeros(1, 4, cache_len, 8), torch.zeros(1, 4, cache_len, 8))
            for _ in range(2)
        )

        out = MagicMock()
        out.logits = logits
        out.past_key_values = fake_kv
        return out

    model.side_effect = fake_forward
    model.__call__ = fake_forward
    return model


def _make_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer with pad_token set."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    return tokenizer


class TestDraftEngine:
    """Tests for DraftEngine initialization and generation with mocked model."""

    def test_initialization(self, draft_config: DraftWorkerConfig) -> None:
        """Engine should initialize without loading a model."""
        engine = DraftEngine(config=draft_config)
        assert not engine.is_loaded
        assert engine.config.device == "cpu"

    @patch("specsplit.workers.draft.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.draft.engine.AutoModelForCausalLM.from_pretrained")
    def test_load_model(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        draft_config: DraftWorkerConfig,
    ) -> None:
        """load_model should call from_pretrained and mark engine as loaded."""
        mock_model_cls.return_value = _make_mock_model()
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        engine = DraftEngine(config=draft_config)
        engine.load_model()

        assert engine.is_loaded
        mock_model_cls.assert_called_once()
        mock_tokenizer_cls.assert_called_once_with(draft_config.model_name)

    @patch("specsplit.workers.draft.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.draft.engine.AutoModelForCausalLM.from_pretrained")
    def test_generate_draft_tree_structure(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        draft_config: DraftWorkerConfig,
    ) -> None:
        """generate_draft_tree should produce correct tree depth and types."""
        mock_model_cls.return_value = _make_mock_model(deterministic_token=7)
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        engine = DraftEngine(config=draft_config)
        engine.load_model()

        roots = engine.generate_draft_tree(
            prompt_ids=[1, 2, 3],
            k=3,
            num_beams=2,
        )

        # Should have `num_beams` root nodes
        assert len(roots) == 2
        assert all(isinstance(r, TokenNode) for r in roots)

        # Each root should head a chain of depth k (k tokens, k-1 levels of children)
        node = roots[0]
        depth = 0
        while node.children:
            assert len(node.children) >= 1
            node = node.children[0]
            depth += 1
        assert depth == 2  # k=3 → 3 tokens → root + 2 children levels

    @patch("specsplit.workers.draft.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.draft.engine.AutoModelForCausalLM.from_pretrained")
    def test_generate_draft_tree_greedy_tokens(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        draft_config: DraftWorkerConfig,
    ) -> None:
        """With temperature=0 (greedy), all tokens should be the argmax token."""
        target_token = 7
        mock_model_cls.return_value = _make_mock_model(deterministic_token=target_token)
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        engine = DraftEngine(config=draft_config)
        engine.load_model()

        roots = engine.generate_draft_tree(
            prompt_ids=[1, 2, 3],
            k=4,
            num_beams=1,
            temperature=0.0,
        )

        assert len(roots) == 1

        # Walk the chain and check all token IDs are the deterministic token
        node = roots[0]
        chain_ids = [node.token_id]
        while node.children:
            node = node.children[0]
            chain_ids.append(node.token_id)

        assert len(chain_ids) == 4
        assert all(tid == target_token for tid in chain_ids)

    def test_to_dict_serialization(self, draft_config: DraftWorkerConfig) -> None:
        """TokenNode.to_dict() should produce a serializable structure."""
        node = TokenNode(token_id=42, log_prob=-0.5, children=[
            TokenNode(token_id=7, log_prob=-0.3)
        ])
        d = node.to_dict()
        assert d["token_id"] == 42
        assert d["log_prob"] == -0.5
        assert len(d["children"]) == 1
        assert d["children"][0]["token_id"] == 7

    def test_reset_cache(self, draft_config: DraftWorkerConfig) -> None:
        """reset_cache should not raise."""
        engine = DraftEngine(config=draft_config)
        engine.reset_cache()  # Should not raise
