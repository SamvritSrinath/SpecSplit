"""Unit tests for specsplit.workers.orchestrator.pipeline — longest path and helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from specsplit.core.config import OrchestratorConfig
from specsplit.proto import spec_decoding_pb2
from specsplit.workers.orchestrator.pipeline import (
    DraftTree,
    PipelineResult,
    _flatten_proto_tree,
    _get_longest_path,
    run_speculative_loop_async,
)


class TestGetLongestPath:
    """Tests for _get_longest_path."""

    def test_single_node(self) -> None:
        """Single root node returns path of one token."""
        draft = DraftTree(token_ids=[42], topology_map=[-1])
        path = _get_longest_path(draft)
        assert path == [42]

    def test_single_chain(self) -> None:
        """Linear chain returns full path root to leaf."""
        draft = DraftTree(
            token_ids=[10, 20, 30],
            topology_map=[-1, 0, 1],
        )
        path = _get_longest_path(draft)
        assert path == [10, 20, 30]

    def test_two_roots_longest_wins(self) -> None:
        """When one root has a deeper subtree, its path is returned."""
        draft = DraftTree(
            token_ids=[1, 2, 3, 4],
            topology_map=[-1, -1, 0, 2],  # roots 0,1; 0->2, 2->3
        )
        # Root 0 -> 2 -> 3 (length 3). Root 1 has no children (length 1).
        path = _get_longest_path(draft)
        assert len(path) == 3
        assert path == [1, 3, 4]

    def test_longer_branch_wins(self) -> None:
        """Longest root-to-leaf path is returned."""
        # Root 0, children 1 and 2. 1->3. So path 0-1-3 has length 3, path 0-2 has length 2.
        draft = DraftTree(
            token_ids=[100, 10, 20, 30],
            topology_map=[-1, 0, 0, 1],
        )
        path = _get_longest_path(draft)
        assert path == [100, 10, 30]

    def test_empty_tree(self) -> None:
        """Empty draft tree returns empty path."""
        draft = DraftTree(token_ids=[], topology_map=[])
        path = _get_longest_path(draft)
        assert path == []

    def test_multiple_equal_length_paths(self) -> None:
        """When several paths have the same max length, one is returned (tie by DFS order)."""
        # Root 0, children 1 and 2. Both leaves. Two paths of length 2.
        draft = DraftTree(
            token_ids=[5, 10, 20],
            topology_map=[-1, 0, 0],
        )
        path = _get_longest_path(draft)
        assert len(path) == 2
        assert path[0] == 5
        assert path[1] in (10, 20)


class TestFlattenProtoTree:
    """Tests for _flatten_proto_tree — must match target engine's BFS ordering."""

    def test_linear_chain(self) -> None:
        """Linear chain should have identical BFS and DFS ordering."""
        leaf = spec_decoding_pb2.TokenNode(token_id=30, log_prob=-0.3)
        mid = spec_decoding_pb2.TokenNode(token_id=20, log_prob=-0.2)
        mid.children.append(leaf)
        root = spec_decoding_pb2.TokenNode(token_id=10, log_prob=-0.1)
        root.children.append(mid)

        token_ids, topology = _flatten_proto_tree([root])
        assert token_ids == [10, 20, 30]
        assert topology == [-1, 0, 1]

    def test_branching_tree_bfs_order(self) -> None:
        """Branching tree should be flattened in BFS (level) order.

        Tree:      10
                  /    \\
                20      30
                |       |
                40      50

        BFS order: [10, 20, 30, 40, 50]
        DFS order: [10, 20, 40, 30, 50]  ← this would be WRONG
        """
        child40 = spec_decoding_pb2.TokenNode(token_id=40, log_prob=-0.4)
        child50 = spec_decoding_pb2.TokenNode(token_id=50, log_prob=-0.5)
        branch20 = spec_decoding_pb2.TokenNode(token_id=20, log_prob=-0.2)
        branch20.children.append(child40)
        branch30 = spec_decoding_pb2.TokenNode(token_id=30, log_prob=-0.3)
        branch30.children.append(child50)
        root = spec_decoding_pb2.TokenNode(token_id=10, log_prob=-0.1)
        root.children.append(branch20)
        root.children.append(branch30)

        token_ids, topology = _flatten_proto_tree([root])

        # BFS ordering: root=0, left_child=1, right_child=2, left_grandchild=3, right_grandchild=4
        assert token_ids == [10, 20, 30, 40, 50]
        assert topology == [-1, 0, 0, 1, 2]

    def test_multiple_roots_bfs(self) -> None:
        """Multiple roots should each have parent -1, flattened in BFS order."""
        root1 = spec_decoding_pb2.TokenNode(token_id=1, log_prob=-0.1)
        root2 = spec_decoding_pb2.TokenNode(token_id=2, log_prob=-0.2)
        child = spec_decoding_pb2.TokenNode(token_id=3, log_prob=-0.3)
        root1.children.append(child)

        token_ids, topology = _flatten_proto_tree([root1, root2])
        # BFS: root1(1) → root2(2) → child-of-root1(3)
        assert token_ids == [1, 2, 3]
        assert topology == [-1, -1, 0]


# ============================================================================
# Helpers for mock stubs
# ============================================================================


def _make_draft_response(
    token_ids: list[int],
    request_id: str = "test",
) -> spec_decoding_pb2.DraftResponse:
    """Build a DraftResponse with a linear chain of TokenNodes.

    Builds bottom-up because protobuf repeated fields copy on append.
    """
    if not token_ids:
        return spec_decoding_pb2.DraftResponse(request_id=request_id)

    # Build the chain bottom-up: leaf first, then wrap each parent around it
    current = spec_decoding_pb2.TokenNode(token_id=token_ids[-1], log_prob=-0.1)
    for tid in reversed(token_ids[:-1]):
        parent = spec_decoding_pb2.TokenNode(token_id=tid, log_prob=-0.1)
        parent.children.append(current)
        current = parent

    return spec_decoding_pb2.DraftResponse(
        request_id=request_id,
        draft_tree=[current],
    )


def _make_verify_response(
    accepted_ids: list[int],
    correction: int,
    num_accepted: int | None = None,
    has_correction: bool = True,
    cache_hit: bool = False,
    request_id: str = "test",
) -> spec_decoding_pb2.VerifyResponse:
    """Build a VerifyResponse."""
    return spec_decoding_pb2.VerifyResponse(
        request_id=request_id,
        accepted_token_ids=accepted_ids,
        correction_token_id=correction,
        has_correction=has_correction,
        num_accepted=num_accepted if num_accepted is not None else len(accepted_ids),
        cache_hit=cache_hit,
    )


class _FakeStub:
    """Mock gRPC stub that returns pre-configured responses.

    When responses are exhausted, repeats the last one (avoids crashes
    from the pipeline's overlapped call pattern).
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._call_idx = 0

    def _next(self, _request: Any) -> Any:
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        # Repeat last response as a safe fallback
        return self._responses[-1]

    def GenerateDrafts(self, request: Any) -> Any:  # noqa: N802
        return self._next(request)

    def VerifyDrafts(self, request: Any) -> Any:  # noqa: N802
        return self._next(request)

    def EndSession(self, request: Any) -> Any:  # noqa: N802
        return spec_decoding_pb2.EndSessionResponse(was_active=False)


# ============================================================================
# Test: Speculation Miss Path (no crash)
# ============================================================================


class TestSpeculationMiss:
    """Verify that the speculation miss path works without NameError.

    Pipeline flow per round:
        1. (concurrent) verify(current_draft) + draft(speculative_context)
        2. Check speculation: accepted_tokens == assumed_path?
        3. If miss: re-draft from corrected context

    Draft stub calls: initial_draft, speculative_draft_per_round, re_draft_on_miss
    """

    def test_miss_does_not_crash(self) -> None:
        """When target rejects all draft tokens, speculation misses do not raise."""
        draft_resp = _make_draft_response([10, 20, 30])
        # 0 accepted → miss ([] != [10,20,30])
        verify_resp = _make_verify_response(accepted_ids=[], correction=99, num_accepted=0)

        # Draft calls: initial(0) + speculative(1) + re-draft(2) + speculative(3) + re-draft(4)
        draft_stub = _FakeStub([draft_resp] * 5)
        # Verify calls: round0 + round1
        target_stub = _FakeStub([verify_resp] * 2)

        cfg = OrchestratorConfig(max_rounds=2, max_output_tokens=100, max_draft_tokens=3)
        result = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=[1, 2, 3],
                config=cfg,
                session_id="",
            )
        )

        # Should complete without NameError, all rounds should be misses
        assert result.speculation_hit_rate == 0.0
        assert result.total_rounds == 2
        # Each round produces 1 bonus token (correction=99)
        assert len(result.output_tokens) == 2

    def test_miss_counter_increments(self) -> None:
        """Speculation miss counter increments when accepted ≠ assumed path."""
        draft_resp = _make_draft_response([10, 20])
        # Partial accept: accepted=[10] ≠ assumed=[10,20] → miss
        verify_resp = _make_verify_response(
            accepted_ids=[10], correction=55, num_accepted=1
        )

        # Draft calls: initial + speculative + re-draft
        draft_stub = _FakeStub([draft_resp] * 3)
        target_stub = _FakeStub([verify_resp])

        cfg = OrchestratorConfig(max_rounds=1, max_output_tokens=100, max_draft_tokens=2)
        result = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=[1],
                config=cfg,
                session_id="",
            )
        )

        # 1 round, partial accept → miss
        assert result.speculation_hit_rate == 0.0
        assert result.total_rounds == 1
        # Output: [10, 55] (1 accepted + correction)
        assert result.output_tokens == [10, 55]


# ============================================================================
# Test: Speculation Hit Path
# ============================================================================


class TestSpeculationHit:
    """Verify that the speculation hit path reuses the speculative draft.

    Speculation is a hit only when BOTH:
      1. accepted_tokens == assumed_path (full path accepted)
      2. bonus_token == speculative_draft.first_root (continuation matches)
    """

    def test_hit_when_path_and_bonus_match(self) -> None:
        """Speculation hit when path is accepted AND bonus == draft root."""
        draft_resp = _make_draft_response([10, 20, 30])
        # ALL tokens accepted AND correction matches speculative draft's root (10)
        verify_resp_hit = _make_verify_response(
            accepted_ids=[10, 20, 30], correction=10, num_accepted=3
        )
        # Second round miss to stop cleanly
        verify_resp_miss = _make_verify_response(
            accepted_ids=[], correction=77, num_accepted=0
        )

        # Draft calls: initial + speculative(r0) + speculative(r1) + re-draft(miss)
        draft_stub = _FakeStub([draft_resp] * 4)
        # Verify calls: round0(hit) + round1(miss)
        target_stub = _FakeStub([verify_resp_hit, verify_resp_miss])

        cfg = OrchestratorConfig(max_rounds=2, max_output_tokens=100, max_draft_tokens=3)
        result = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=[1, 2],
                config=cfg,
                session_id="",
            )
        )

        assert result.total_rounds == 2
        # Round 0: hit (accepted [10,20,30] == assumed [10,20,30] AND bonus 10 == root 10)
        # Round 1: miss (accepted [] != assumed [10,20,30])
        assert result.speculation_hit_rate == 0.5  # 1 hit / 2 total

    def test_miss_when_bonus_differs(self) -> None:
        """Path match alone is insufficient — bonus must match draft root."""
        draft_resp = _make_draft_response([10, 20, 30])
        # ALL tokens accepted BUT correction (99) != speculative draft root (10) → miss
        verify_resp = _make_verify_response(
            accepted_ids=[10, 20, 30], correction=99, num_accepted=3
        )

        # Draft calls: initial + speculative(r0) + re-draft(miss)
        draft_stub = _FakeStub([draft_resp] * 3)
        target_stub = _FakeStub([verify_resp])

        cfg = OrchestratorConfig(max_rounds=1, max_output_tokens=100, max_draft_tokens=3)
        result = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=[1, 2],
                config=cfg,
                session_id="",
            )
        )

        assert result.total_rounds == 1
        # Path matched but bonus didn't → speculation miss
        assert result.speculation_hit_rate == 0.0


# ============================================================================
# Test: EOS Termination
# ============================================================================


class TestEOSTermination:
    """Verify that the pipeline stops on EOS token."""

    def test_stops_at_eos_in_accepted(self) -> None:
        """Pipeline stops when EOS token appears in accepted tokens."""
        draft_resp = _make_draft_response([10, 2, 30])  # 2 = EOS
        # Target accepts all, correction=99
        verify_resp = _make_verify_response(
            accepted_ids=[10, 2, 30], correction=99, num_accepted=3
        )

        # Draft calls: initial + speculative(r0)
        draft_stub = _FakeStub([draft_resp] * 2)
        target_stub = _FakeStub([verify_resp])

        cfg = OrchestratorConfig(max_rounds=10, max_output_tokens=100, max_draft_tokens=3)
        result = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=[1],
                config=cfg,
                session_id="",
                eos_token_id=2,
            )
        )

        # Should have truncated at EOS (token 2), output = [10, 2]
        assert result.output_tokens == [10, 2]
        assert result.total_rounds == 1


# ============================================================================
# Test: Acceptance Rate Calculation
# ============================================================================


class TestAcceptanceRate:
    """Verify that acceptance rate uses path depth, not tree node count."""

    def test_acceptance_uses_path_depth(self) -> None:
        """acceptance_rate = total_accepted / total_path_depth."""
        draft_resp = _make_draft_response([10, 20, 30])
        # Accept 2 out of 3 path tokens → miss
        verify_resp = _make_verify_response(
            accepted_ids=[10, 20], correction=55, num_accepted=2
        )

        # Draft calls: initial + speculative + re-draft
        draft_stub = _FakeStub([draft_resp] * 3)
        target_stub = _FakeStub([verify_resp])

        cfg = OrchestratorConfig(max_rounds=1, max_output_tokens=100, max_draft_tokens=3)
        result = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=[1],
                config=cfg,
                session_id="",
            )
        )

        # acceptance = 2 accepted / 3 path_depth ≈ 0.667 (NOT 2/total_tree_nodes)
        assert abs(result.acceptance_rate - 2.0 / 3.0) < 0.01

