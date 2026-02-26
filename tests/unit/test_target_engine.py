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
from specsplit.workers.target.kv_cache import StaticKVCache


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
def fake_static_kv_cache() -> StaticKVCache:
    """A StaticKVCache with 2 layers, 4 heads, max_seq_len=64, head_dim=8.

    Pre-populated with 10 random entries to simulate an in-use cache.
    """
    cache = StaticKVCache(
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        head_dim=8,
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    # Populate with 10 positions of random data
    keys = torch.randn(2, 1, 4, 10, 8)
    values = torch.randn(2, 1, 4, 10, 8)
    cache.append(keys, values)
    return cache


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

    # Add model config for StaticKVCache initialization
    model.config = MagicMock()
    model.config.num_hidden_layers = 2
    model.config.num_attention_heads = 4
    model.config.num_key_value_heads = 4
    model.config.hidden_size = 32  # 4 heads * 8 head_dim
    model.config.max_position_embeddings = 512

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

        # When given StaticKVCache, simulate in-place update and return it
        if past_key_values is not None and hasattr(past_key_values, "append"):
            new_len = seq_len
            if new_len > 0:
                keys = torch.zeros(2, batch, 4, new_len, 8)
                values = torch.zeros(2, batch, 4, new_len, 8)
                past_key_values.append(keys, values)
            out = MagicMock()
            out.logits = logits
            out.past_key_values = past_key_values
            return out

        # Legacy path: fake tuple KV cache
        cache_len = seq_len
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                cache_len += past_key_values[0][0].shape[2]
            elif hasattr(past_key_values, "get_seq_length"):
                cache_len += past_key_values.get_seq_length()
            elif hasattr(past_key_values, "seq_len"):
                cache_len += past_key_values.seq_len
        fake_kv = tuple(
            (torch.zeros(batch, 4, cache_len, 8), torch.zeros(batch, 4, cache_len, 8))
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


# =========================================================================
# Flatten Tree Helper Tests
# =========================================================================


class TestFlattenTree:
    """Tests for the _flatten_tree helper."""

    def _flatten(self, tree: list) -> tuple[list[int], list[int], list[float]]:
        """Call _flatten_tree with test defaults and return first 3 values."""
        result = _flatten_tree(tree, vocab_size=32000, device=torch.device("cpu"), dtype=torch.float32)
        return result[0], result[1], result[2]

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
        token_ids, topology, log_probs = self._flatten(tree)
        assert token_ids == [10, 20, 30]
        assert topology == [-1, 0, 1]
        assert log_probs == [-0.1, -0.2, -0.3]

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
        token_ids, topology, log_probs = self._flatten(tree)
        assert token_ids == [10, 20, 30]
        assert topology == [-1, 0, 0]
        assert log_probs == [-0.1, -0.2, -0.3]

    def test_empty_tree(self) -> None:
        """An empty tree should produce empty lists."""
        token_ids, topology, log_probs = self._flatten([])
        assert token_ids == []
        assert topology == []
        assert log_probs == []

    def test_multiple_roots(self) -> None:
        """Multiple root nodes should each have parent -1."""
        tree = [
            {"token_id": 10, "log_prob": -0.1, "children": []},
            {"token_id": 20, "log_prob": -0.2, "children": []},
        ]
        token_ids, topology, log_probs = self._flatten(tree)
        assert token_ids == [10, 20]
        assert topology == [-1, -1]
        assert log_probs == [-0.1, -0.2]


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
    """Tests for rollback_cache — using StaticKVCache O(1) rollback and compact."""

    def test_rollback_crops_tensors(
        self,
        target_engine: TargetEngine,
        fake_static_kv_cache: StaticKVCache,
    ) -> None:
        """Rollback should set the StaticKVCache seq_len to accepted_depth."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.cache = fake_static_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=6)

        assert cache.seq_len == 6
        assert cache.cache.seq_len == 6

    def test_rollback_to_zero(
        self,
        target_engine: TargetEngine,
        fake_static_kv_cache: StaticKVCache,
    ) -> None:
        """Rolling back to 0 should set seq_len to 0."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.cache = fake_static_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=0)
        assert cache.seq_len == 0
        assert cache.cache.seq_len == 0

    def test_rollback_noop_same_depth(
        self,
        target_engine: TargetEngine,
        fake_static_kv_cache: StaticKVCache,
    ) -> None:
        """Rollback to current seq_len should be a no-op."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.cache = fake_static_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=10)
        assert cache.seq_len == 10
        assert cache.cache.seq_len == 10

    def test_rollback_no_cache_state(self, target_engine: TargetEngine) -> None:
        """Rollback on a session with no StaticKVCache should reset to 0."""
        target_engine.get_or_create_session("sess-1")
        target_engine.rollback_cache("sess-1", accepted_depth=0)
        assert target_engine._session_caches["sess-1"].seq_len == 0

    def test_rollback_unknown_session_raises(self, target_engine: TargetEngine) -> None:
        with pytest.raises(KeyError, match="no-such"):
            target_engine.rollback_cache("no-such", accepted_depth=5)

    def test_rollback_exceeds_seq_len_raises(
        self,
        target_engine: TargetEngine,
        fake_static_kv_cache: StaticKVCache,
    ) -> None:
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.cache = fake_static_kv_cache
        cache.seq_len = 10

        with pytest.raises(ValueError, match="exceeds cached seq_len"):
            target_engine.rollback_cache("sess-1", accepted_depth=15)

    def test_rollback_preserves_data(
        self,
        target_engine: TargetEngine,
        fake_static_kv_cache: StaticKVCache,
    ) -> None:
        """The cropped prefix should contain the original data, not zeros."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.cache = fake_static_kv_cache
        cache.seq_len = 10

        # Save original first 4 positions
        original_keys = fake_static_kv_cache.get_all_kv()[0][0][:, :, :4, :].clone()

        target_engine.rollback_cache("sess-1", accepted_depth=4)

        # After rollback, the first 4 positions should be preserved
        rolled_keys = cache.cache.get_all_kv()[0][0][:, :, :4, :]
        assert torch.allclose(rolled_keys, original_keys)

    def test_compact_non_contiguous(
        self,
        target_engine: TargetEngine,
        fake_static_kv_cache: StaticKVCache,
    ) -> None:
        """Compact with non-contiguous indices for branching tree rollback."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.cache = fake_static_kv_cache
        cache.seq_len = 10

        # Keep prefix (0-4) + selected tree positions (6, 8)
        target_engine.rollback_cache(
            "sess-1",
            accepted_depth=7,  # Not used when tree_indices provided
            accepted_tree_indices=[1, 3],
            prefix_length=5,
        )
        # Should keep: [0,1,2,3,4] (prefix) + [5+1, 5+3] = [0,1,2,3,4,6,8] = 7 positions
        assert cache.seq_len == 7


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

    @patch("specsplit.workers.target.engine.verify_greedy_tree")
    @patch("specsplit.workers.target.engine.verify_stochastic_tree")
    @patch("specsplit.workers.target.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.target.engine.AutoModelForCausalLM.from_pretrained")
    def test_verify_greedy_path_when_temperature_zero(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        mock_verify_stochastic: MagicMock,
        mock_verify_greedy: MagicMock,
    ) -> None:
        """When temperature=0, verify_greedy_tree is used (stochastic path not used)."""
        draft_ids = [10, 20]
        prompt_ids = [1, 2]
        accept_tokens = [0, 10, 20, 0]
        mock_verify_greedy.return_value = MagicMock(
            accepted_tokens=draft_ids,
            bonus_token=99,
            num_accepted=2,
            accepted_leaf_index=1,
            diverged=True,
            accepted_indices=[0, 1],
        )
        mock_model_cls.return_value = _make_mock_target_model(accept_tokens=accept_tokens)
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        config = TargetWorkerConfig(model_name="gpt2", device="cpu", max_sessions=3)
        engine = TargetEngine(config=config)
        engine.load_model()

        tree = self._make_tree(draft_ids)
        engine.verify_draft_tree(
            prompt_ids=prompt_ids,
            draft_tree=tree,
            session_id=None,
            temperature=0.0,
        )

        mock_verify_greedy.assert_called_once()
        mock_verify_stochastic.assert_not_called()

    @patch("specsplit.workers.target.engine.verify_greedy_tree")
    @patch("specsplit.workers.target.engine.verify_stochastic_tree")
    @patch("specsplit.workers.target.engine.AutoTokenizer.from_pretrained")
    @patch("specsplit.workers.target.engine.AutoModelForCausalLM.from_pretrained")
    def test_verify_stochastic_path_when_temperature_positive(
        self,
        mock_model_cls: MagicMock,
        mock_tokenizer_cls: MagicMock,
        mock_verify_stochastic: MagicMock,
        mock_verify_greedy: MagicMock,
    ) -> None:
        """When temperature > 0, verify_stochastic_tree is used (greedy path not used)."""
        draft_ids = [10, 20]
        prompt_ids = [1, 2]
        accept_tokens = [0, 10, 20, 0]
        mock_verify_stochastic.return_value = MagicMock(
            accepted_tokens=draft_ids,
            bonus_token=99,
            num_accepted=2,
            accepted_leaf_index=1,
            diverged=True,
            accepted_indices=[0, 1],
        )
        mock_model_cls.return_value = _make_mock_target_model(accept_tokens=accept_tokens)
        mock_tokenizer_cls.return_value = _make_mock_tokenizer()

        config = TargetWorkerConfig(model_name="gpt2", device="cpu", max_sessions=3)
        engine = TargetEngine(config=config)
        engine.load_model()

        tree = self._make_tree(draft_ids)
        result = engine.verify_draft_tree(
            prompt_ids=prompt_ids,
            draft_tree=tree,
            session_id=None,
            temperature=0.5,
        )

        mock_verify_stochastic.assert_called_once()
        mock_verify_greedy.assert_not_called()
        assert result.num_accepted == 2
        assert result.accepted_token_ids == draft_ids
        assert result.correction_token_id == 99
