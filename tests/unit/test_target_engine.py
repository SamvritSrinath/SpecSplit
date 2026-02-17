"""Unit tests for the TargetEngine — session-based KV caching and rollback."""

from __future__ import annotations

import pytest
import torch

from specsplit.core.config import TargetWorkerConfig
from specsplit.workers.target.engine import KVCacheState, TargetEngine, VerificationResult


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


class TestTargetEngineInit:
    """Tests for TargetEngine initialization."""

    def test_init_defaults(self, target_engine: TargetEngine):
        assert not target_engine.is_loaded
        assert target_engine.active_sessions == 0

    def test_init_max_sessions(self, target_engine: TargetEngine):
        assert target_engine.config.max_sessions == 3


class TestSessionManagement:
    """Tests for session creation, retrieval, and eviction."""

    def test_create_new_session(self, target_engine: TargetEngine):
        cache, hit = target_engine.get_or_create_session("sess-1")
        assert not hit
        assert isinstance(cache, KVCacheState)
        assert cache.seq_len == 0
        assert target_engine.active_sessions == 1

    def test_reuse_existing_session(self, target_engine: TargetEngine):
        target_engine.get_or_create_session("sess-1")
        cache, hit = target_engine.get_or_create_session("sess-1")
        assert hit
        assert target_engine.active_sessions == 1

    def test_multiple_sessions(self, target_engine: TargetEngine):
        for i in range(3):
            target_engine.get_or_create_session(f"sess-{i}")
        assert target_engine.active_sessions == 3

    def test_lru_eviction(self, target_engine: TargetEngine):
        """Creating a 4th session should evict the oldest (max_sessions=3)."""
        for i in range(3):
            target_engine.get_or_create_session(f"sess-{i}")
        # This should evict sess-0
        target_engine.get_or_create_session("sess-3")
        assert target_engine.active_sessions == 3
        # sess-0 should be gone
        _, hit = target_engine.get_or_create_session("sess-0")
        assert not hit  # re-created, not a cache hit

    def test_end_session_existing(self, target_engine: TargetEngine):
        target_engine.get_or_create_session("sess-1")
        assert target_engine.end_session("sess-1") is True
        assert target_engine.active_sessions == 0

    def test_end_session_nonexistent(self, target_engine: TargetEngine):
        assert target_engine.end_session("no-such-session") is False


class TestRollbackCache:
    """Tests for rollback_cache — cropping KV tensors to accepted prefix."""

    def test_rollback_crops_tensors(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ):
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
    ):
        """Rolling back to 0 should produce zero-length seq dim."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=0)
        assert cache.seq_len == 0
        for layer_k, layer_v in cache.past_key_values:
            assert layer_k.shape[2] == 0

    def test_rollback_noop_same_depth(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ):
        """Rollback to current seq_len should be a no-op."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        target_engine.rollback_cache("sess-1", accepted_depth=10)
        assert cache.seq_len == 10
        for layer_k, _ in cache.past_key_values:
            assert layer_k.shape[2] == 10

    def test_rollback_no_cache_state(self, target_engine: TargetEngine):
        """Rollback on a session with no past_key_values should reset to 0."""
        target_engine.get_or_create_session("sess-1")
        target_engine.rollback_cache("sess-1", accepted_depth=0)
        assert target_engine._session_caches["sess-1"].seq_len == 0

    def test_rollback_unknown_session_raises(self, target_engine: TargetEngine):
        with pytest.raises(KeyError, match="no-such"):
            target_engine.rollback_cache("no-such", accepted_depth=5)

    def test_rollback_exceeds_seq_len_raises(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ):
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        with pytest.raises(ValueError, match="exceeds cached seq_len"):
            target_engine.rollback_cache("sess-1", accepted_depth=15)

    def test_rollback_preserves_data(
        self,
        target_engine: TargetEngine,
        fake_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ):
        """The cropped prefix should contain the original data, not zeros."""
        cache, _ = target_engine.get_or_create_session("sess-1")
        cache.past_key_values = fake_kv_cache
        cache.seq_len = 10

        # Save original first 4 positions of layer 0 key
        original_prefix = fake_kv_cache[0][0][:, :, :4, :].clone()

        target_engine.rollback_cache("sess-1", accepted_depth=4)

        rolled_back_key = cache.past_key_values[0][0]
        assert torch.allclose(rolled_back_key, original_prefix)


class TestVerifyWithSession:
    """Tests for verify_draft_tree with session-based caching."""

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

    def test_stateless_verify(self, target_engine: TargetEngine):
        """session_id=None should work without creating any session."""
        tree = self._make_tree([10, 20, 30])
        result = target_engine.verify_draft_tree(
            prompt_ids=[1, 2],
            draft_tree=tree,
            session_id=None,
        )
        assert isinstance(result, VerificationResult)
        assert result.accepted_token_ids == [10, 20, 30]
        assert result.cache_hit is False
        assert target_engine.active_sessions == 0

    def test_session_verify_creates_cache(self, target_engine: TargetEngine):
        """First call with a session_id should create a cache (cache_hit=False)."""
        tree = self._make_tree([10, 20])
        result = target_engine.verify_draft_tree(
            prompt_ids=[1, 2, 3],
            draft_tree=tree,
            session_id="sess-A",
        )
        assert result.cache_hit is False
        assert target_engine.active_sessions == 1

    def test_session_verify_reuses_cache(self, target_engine: TargetEngine):
        """Second call with same session_id should be a cache hit."""
        tree = self._make_tree([10])
        target_engine.verify_draft_tree(
            prompt_ids=[1], draft_tree=tree, session_id="sess-A"
        )
        result = target_engine.verify_draft_tree(
            prompt_ids=[1], draft_tree=self._make_tree([20]), session_id="sess-A"
        )
        assert result.cache_hit is True
