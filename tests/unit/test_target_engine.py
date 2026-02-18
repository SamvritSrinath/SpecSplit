"""Unit tests for the TargetEngine — session caching, rollback, and verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from specsplit.core.config import TargetWorkerConfig
from specsplit.workers.target.engine import (
    KVCacheState,
    TargetEngine,
    VerificationResult,
    _flatten_tree,
)


@pytest.fixture
def target_engine() -> TargetEngine:
    """A TargetEngine configured for CPU with a small session limit."""
    config = TargetWorkerConfig(
        model_name="gpt2",
        device="cpu",
        grpc_port=50052,
        max_sessions=3,
    )
    return TargetEngine(config=config)


@pytest.fixture
def fake_kv_cache() -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """A fake past_key_values with 2 layers, seq_len=10.

    Shape per tensor: (batch=1, heads=4, seq_len=10, head_dim=8).
    """
    layers = []
    for _ in range(2):
        key = torch.randn(1, 4, 10, 8)
        value = torch.randn(1, 4, 10, 8)
        layers.append((key, value))
    return tuple(layers)


def _make_mock_target_model(
    vocab_size: int = 100,
    accept_tokens: list[int] | None = None,
) -> MagicMock:
    """Create a mock target model whose forward returns controlled logits.

    Args:
        vocab_size: Size of the vocabulary.
        accept_tokens: If provided, the mock will return logits where
            ``argmax(logits[i]) == accept_tokens[i]`` for each position.
            If ``None``, defaults to always predicting token 0.
    """
    model = MagicMock()
    model.dtype = torch.float16
    model.eval.return_value = model
    model.to.return_value = model

    accept_list = accept_tokens or []

    def fake_forward(
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        batch, seq_len = input_ids.shape

        logits = torch.zeros(batch, seq_len, vocab_size)
        for i in range(seq_len):
            if i < len(accept_list):
                logits[0, i, accept_list[i]] = 10.0
            else:
                logits[0, i, 0] = 10.0  # default: predict token 0

        # Fake KV cache
        cache_len = seq_len
        if past_key_values is not None:
            cache_len += past_key_values[0][0].shape[2]
        fake_kv = tuple(
            (torch.zeros(1, 4, cache_len, 8), torch.zeros(1, 4, cache_len, 8)) for _ in range(2)
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


# =========================================================================
# Flatten Tree Helper Tests
# =========================================================================


class TestFlattenTree:
    """Tests for the _flatten_tree helper."""

    def test_single_chain(self) -> None:
        """A linear chain should flatten to sequential indices."""
        tree = [
            {
                "token_id": 10,
                "log_prob": -0.1,
                "children": [
                    {
                        "token_id": 20,
                        "log_prob": -0.2,
                        "children": [{"token_id": 30, "log_prob": -0.3, "children": []}],
                    }
                ],
            }
        ]
        token_ids, topology = _flatten_tree(tree)
        assert token_ids == [10, 20, 30]
        assert topology == [-1, 0, 1]

    def test_branching_tree(self) -> None:
        """A tree with branching should produce correct parent indices."""
        tree = [
            {
                "token_id": 10,
                "log_prob": -0.1,
                "children": [
                    {"token_id": 20, "log_prob": -0.2, "children": []},
                    {"token_id": 30, "log_prob": -0.3, "children": []},
                ],
            }
        ]
        token_ids, topology = _flatten_tree(tree)
        assert token_ids == [10, 20, 30]
        assert topology == [-1, 0, 0]

    def test_empty_tree(self) -> None:
        """An empty tree should produce empty lists."""
        token_ids, topology = _flatten_tree([])
        assert token_ids == []
        assert topology == []

    def test_multiple_roots(self) -> None:
        """Multiple root nodes should each have parent -1."""
        tree = [
            {"token_id": 10, "log_prob": -0.1, "children": []},
            {"token_id": 20, "log_prob": -0.2, "children": []},
        ]
        token_ids, topology = _flatten_tree(tree)
        assert token_ids == [10, 20]
        assert topology == [-1, -1]


# =========================================================================
# TargetEngine Initialization
# =========================================================================


class TestTargetEngineInit:
    """Tests for TargetEngine initialization."""

    def test_init_defaults(self, target_engine: TargetEngine) -> None:
        assert not target_engine.is_loaded
        assert target_engine.active_sessions == 0

    def test_init_max_sessions(self, target_engine: TargetEngine) -> None:
        assert target_engine.config.max_sessions == 3


# =========================================================================
# Session Management
# =========================================================================


class TestSessionManagement:
    """Tests for session creation, retrieval, and eviction."""

    def test_create_new_session(self, target_engine: TargetEngine) -> None:
        cache, hit = target_engine.get_or_create_session("sess-1")
        assert not hit
        assert isinstance(cache, KVCacheState)
        assert cache.seq_len == 0
        assert target_engine.active_sessions == 1

    def test_reuse_existing_session(self, target_engine: TargetEngine) -> None:
        target_engine.get_or_create_session("sess-1")
        _cache, hit = target_engine.get_or_create_session("sess-1")
        assert hit
        assert target_engine.active_sessions == 1

    def test_multiple_sessions(self, target_engine: TargetEngine) -> None:
        for i in range(3):
            target_engine.get_or_create_session(f"sess-{i}")
        assert target_engine.active_sessions == 3

    def test_lru_eviction(self, target_engine: TargetEngine) -> None:
        """Creating a 4th session should evict the oldest (max_sessions=3)."""
        for i in range(3):
            target_engine.get_or_create_session(f"sess-{i}")
        # This should evict sess-0
        target_engine.get_or_create_session("sess-3")
        assert target_engine.active_sessions == 3
        # sess-0 should be gone
        _, hit = target_engine.get_or_create_session("sess-0")
        assert not hit  # re-created, not a cache hit

    def test_end_session_existing(self, target_engine: TargetEngine) -> None:
        target_engine.get_or_create_session("sess-1")
        assert target_engine.end_session("sess-1") is True
        assert target_engine.active_sessions == 0

    def test_end_session_nonexistent(self, target_engine: TargetEngine) -> None:
        assert target_engine.end_session("no-such-session") is False


# =========================================================================
# KV Cache Rollback
# =========================================================================


class TestRollbackCache:
    """Tests for rollback_cache — cropping KV tensors to accepted prefix."""

    def test_rollback_crops_tensors(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        """Rollback should slice each layer's K/V to accepted_depth."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=6)

        assert cache.seq_len == 6
        for layer_k, layer_v in cache.past_key_values:
            assert layer_k.shape[2] == 6
            assert layer_v.shape[2] == 6

    def test_rollback_to_zero(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        """Rolling back to 0 should produce zero-length seq dim."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=0)
        assert cache.seq_len == 0
        for layer_k, _layer_v in cache.past_key_values:
            assert layer_k.shape[2] == 0

    def test_rollback_noop_same_depth(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        """Rollback to current seq_len should be a no-op."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=10)
        assert cache.seq_len == 10
        for layer_k, _ in cache.past_key_values:
            assert layer_k.shape[2] == 10

    def test_rollback_no_cache_state(self, target_engine: TargetEngine) -> None:
        """Rollback on a session with no past_key_values should reset to 0."""
        target_engine.get_or_create_session("sess-1")
        target_engine.rollback_cache("sess-1", accepted_depth=0)
        assert target_engine._session_caches["sess-1"].seq_len == 0

    def test_rollback_unknown_session_raises(self, target_engine: TargetEngine) -> None:
        with pytest.raises(KeyError, match="no-such"):
            target_engine.rollback_cache("no-such", accepted_depth=5)

    def test_rollback_exceeds_seq_len_raises(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        with pytest.raises(ValueError, match="exceeds cached seq_len"):
            target_engine.rollback_cache("sess-1", accepted_depth=15)

    def test_rollback_preserves_data(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        """The cropped prefix should contain the original data, not zeros."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        # Save original first 4 positions of layer 0 key
        original_prefix = fake_kv_cache[0][0][:, :, :4, :].clone()

        target_engine.rollback_cache("sess-1", accepted_depth=4)

        rolled_back_key = cache.past_key_values[0][0]
        assert torch.allclose(rolled_back_key, original_prefix)


# =========================================================================
# Verification with Mocked Model
# =========================================================================


class TestVerifyWithMockedModel:
    """Tests for verify_draft_tree using a mocked target model."""

    def _make_tree(self, token_ids: list[int]) -> list[dict]:
        """Build a linear chain of token nodes for testing."""
        if not token_ids:
            return []
        nodes: list[dict] = []
        current: dict = {"token_id": token_ids[0], "log_prob": -0.1, "children": []}
        nodes.append(current)
        for tid in token_ids[1:]:
            child: dict = {"token_id": tid, "log_prob": -0.1, "children": []}
            current["children"] = [child]
            current = child
        return nodes

    @patch("specsplit.workers.target.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.target.engine.AutoModelForCausalLM.from_pretrained")
    def test_verify_all_accepted(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
    ) -> None:
        """When target argmax matches all draft tokens, all should be accepted."""
        draft_ids = [10, 20, 30]
        # With corrected logit alignment semantics:
        # - To verify draft token i, we use logits from its PARENT position
        # - For roots: use logits from last prefix position (position 1)
        # - For children: use logits from parent position
        # Position 0 (prompt[0]) → logits predict prompt[1] (don't care, use 0)
        # Position 1 (prompt[1]) → logits predict tree[0]=10 (root)
        # Position 2 (tree[0])   → logits predict tree[1]=20 (child of tree[0])
        # Position 3 (tree[1])   → logits predict tree[2]=30 (child of tree[1])
        # Position 4 (tree[2])   → logits predict next (don't care, use 0)
        prompt_ids = [1, 2]
        # accept_tokens[i] = what argmax(logits[i]) should be
        # Positions 0 and 4 are placeholders (not used in verification); value 0 is arbitrary
        accept_tokens = [0, 10, 20, 30, 0]
        mock_model_cls.return_value = _make_mock_target_model(
            accept_tokens=accept_tokens,
        )
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        config = TargetWorkerConfig(model_name="gpt2", device="cpu", max_sessions=3)
        engine = TargetEngine(config=config)
        engine.load_model()

        tree = self._make_tree(draft_ids)
        result = engine.verify_draft_tree(
            prompt_ids=prompt_ids,
            draft_tree=tree,
            session_id=None,
        )

        assert isinstance(result, VerificationResult)
        assert result.accepted_token_ids == draft_ids
        assert result.num_accepted == len(draft_ids)

    @patch("specsplit.workers.target.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.target.engine.AutoModelForCausalLM.from_pretrained")
    def test_verify_partial_acceptance(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
    ) -> None:
        """When target disagrees at position 2, only first 2 tokens accepted."""
        draft_ids = [10, 20, 30]
        prompt_ids = [1, 2]
        # With corrected semantics, target agrees with 10, 20 but predicts 99 where draft says 30:
        # Position 1 (prompt[1]) → logits predict 10 (tree[0])
        # Position 2 (tree[0])   → logits predict 20 (tree[1])
        # Position 3 (tree[1])   → logits predict 99 (MISMATCH, draft=30)
        accept_tokens = [0, 10, 20, 99, 0]
        mock_model_cls.return_value = _make_mock_target_model(
            accept_tokens=accept_tokens,
        )
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        config = TargetWorkerConfig(model_name="gpt2", device="cpu", max_sessions=3)
        engine = TargetEngine(config=config)
        engine.load_model()

        tree = self._make_tree(draft_ids)
        result = engine.verify_draft_tree(
            prompt_ids=prompt_ids,
            draft_tree=tree,
            session_id=None,
        )

        assert result.num_accepted == 2
        assert result.accepted_token_ids == [10, 20]
        assert result.correction_token_id == 99

    @patch("specsplit.workers.target.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.target.engine.AutoModelForCausalLM.from_pretrained")
    def test_verify_empty_tree(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
    ) -> None:
        """Verify with an empty draft tree should return empty result."""
        mock_model_cls.return_value = _make_mock_target_model()
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        config = TargetWorkerConfig(model_name="gpt2", device="cpu", max_sessions=3)
        engine = TargetEngine(config=config)
        engine.load_model()

        result = engine.verify_draft_tree(
            prompt_ids=[1, 2],
            draft_tree=[],
            session_id=None,
        )

        assert result.num_accepted == 0
        assert result.accepted_token_ids == []
        assert result.correction_token_id is None

    @patch("specsplit.workers.target.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.target.engine.AutoModelForCausalLM.from_pretrained")
    def test_session_creates_cache(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
    ) -> None:
        """First call with session_id should create a cache (cache_hit=False)."""
        draft_ids = [10]
        prompt_ids = [1, 2, 3]
        accept_tokens = list(prompt_ids) + draft_ids
        mock_model_cls.return_value = _make_mock_target_model(
            accept_tokens=accept_tokens,
        )
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        config = TargetWorkerConfig(model_name="gpt2", device="cpu", max_sessions=3)
        engine = TargetEngine(config=config)
        engine.load_model()

        tree = self._make_tree(draft_ids)
        result = engine.verify_draft_tree(
            prompt_ids=prompt_ids,
            draft_tree=tree,
            session_id="sess-A",
        )
        assert result.cache_hit is False
        assert engine.active_sessions == 1

    @patch("specsplit.workers.target.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.target.engine.AutoModelForCausalLM.from_pretrained")
    def test_session_reuses_cache(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
    ) -> None:
        """Second call with same session_id should be a cache hit."""
        draft_ids = [10]
        prompt_ids = [1]
        accept_tokens = list(prompt_ids) + draft_ids
        mock_model_cls.return_value = _make_mock_target_model(
            accept_tokens=accept_tokens,
        )
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        config = TargetWorkerConfig(model_name="gpt2", device="cpu", max_sessions=3)
        engine = TargetEngine(config=config)
        engine.load_model()

        tree = self._make_tree(draft_ids)
        engine.verify_draft_tree(prompt_ids=prompt_ids, draft_tree=tree, session_id="sess-A")
        result = engine.verify_draft_tree(
            prompt_ids=prompt_ids,
            draft_tree=self._make_tree([20]),
            session_id="sess-A",
        )
        assert result.cache_hit is True
