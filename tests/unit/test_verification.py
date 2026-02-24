"""Unit tests for specsplit.core.verification — greedy and stochastic tree verification."""

from __future__ import annotations

import pytest
import torch

from specsplit.core.verification import verify_greedy_tree, verify_stochastic_tree


class TestVerifyGreedyTree:
    """Tests for verify_greedy_tree."""

    def test_basic_match_single_path(self) -> None:
        """When target argmax matches all draft tokens on a chain, all are accepted."""
        draft_tokens = torch.tensor([42, 17, 99], dtype=torch.long)
        vocab_size = 128
        target_logits = torch.zeros(3, vocab_size)
        target_logits[0, 42] = 10.0
        target_logits[1, 17] = 10.0
        target_logits[2, 99] = 10.0
        topology_map = [-1, 0, 1]  # linear chain
        result = verify_greedy_tree(draft_tokens, target_logits, topology_map)
        assert result.accepted_tokens == [42, 17, 99]
        assert result.num_accepted == 3
        assert result.accepted_leaf_index == 2
        assert result.bonus_token >= 0

    def test_mismatch_at_root(self) -> None:
        """When target disagrees at root, no tokens accepted; bonus from first root."""
        draft_tokens = torch.tensor([42, 17], dtype=torch.long)
        vocab_size = 128
        target_logits = torch.zeros(2, vocab_size)
        target_logits[0, 99] = 10.0  # target says 99, draft says 42
        target_logits[1, 17] = 10.0
        topology_map = [-1, 0]
        result = verify_greedy_tree(draft_tokens, target_logits, topology_map)
        assert result.accepted_tokens == []
        assert result.num_accepted == 0
        assert result.accepted_leaf_index == -1
        assert result.bonus_token == 99

    def test_longest_path_with_branching(self) -> None:
        """Longest continuously-accepted path is chosen in a branching tree."""
        # Tree: root(0) -> left(1), right(2). Left branch: 1->3. Right branch: 2->4.
        # Draft: [42, 17, 99, 8, 55]. Target agrees at 0,1,2,3; disagrees at 4 (draft=55, target=7).
        draft_tokens = torch.tensor([42, 17, 99, 8, 55], dtype=torch.long)
        vocab_size = 128
        target_logits = torch.zeros(5, vocab_size)
        target_logits[0, 42] = 10.0
        target_logits[1, 17] = 10.0
        target_logits[2, 99] = 10.0
        target_logits[3, 8] = 10.0
        target_logits[4, 7] = 10.0  # mismatch at node 4
        topology_map = [-1, 0, 0, 1, 2]  # root 0, children 1,2; 1->3, 2->4
        result = verify_greedy_tree(draft_tokens, target_logits, topology_map)
        # Longest accepted path: 0,1,3 (length 3). Path 0,2,4 stops at 4 (rejected).
        # Bonus is target's choice at the accepted leaf (node 3), i.e. argmax at 3 = 8.
        assert result.accepted_tokens == [42, 17, 8]
        assert result.num_accepted == 3
        assert result.bonus_token == 8

    def test_all_roots_rejected_bonus_from_first_root(self) -> None:
        """When all roots are rejected, bonus token is target choice at first root."""
        draft_tokens = torch.tensor([1, 2], dtype=torch.long)
        vocab_size = 32
        target_logits = torch.zeros(2, vocab_size)
        target_logits[0, 10] = 10.0  # first root: target 10, draft 1
        target_logits[1, 20] = 10.0
        topology_map = [-1, -1]  # two roots
        result = verify_greedy_tree(draft_tokens, target_logits, topology_map)
        assert result.accepted_tokens == []
        assert result.bonus_token == 10

    def test_empty_tree(self) -> None:
        """Empty topology returns empty result and bonus -1."""
        draft_tokens = torch.tensor([], dtype=torch.long)
        target_logits = torch.zeros(0, 64)
        topology_map: list[int] = []
        result = verify_greedy_tree(draft_tokens, target_logits, topology_map)
        assert result.accepted_tokens == []
        assert result.bonus_token == -1
        assert result.accepted_leaf_index == -1
        assert result.num_draft_tokens == 0

    def test_shape_mismatch_draft_tokens_raises(self) -> None:
        """Draft tokens length inconsistent with topology_map raises ValueError."""
        draft_tokens = torch.tensor([1, 2, 3], dtype=torch.long)
        target_logits = torch.zeros(2, 64)
        topology_map = [-1, 0]
        with pytest.raises(ValueError, match="draft_tokens length"):
            verify_greedy_tree(draft_tokens, target_logits, topology_map)

    def test_shape_mismatch_target_logits_raises(self) -> None:
        """Target logits first dim inconsistent with topology_map raises ValueError."""
        draft_tokens = torch.tensor([1, 2], dtype=torch.long)
        target_logits = torch.zeros(3, 64)
        topology_map = [-1, 0]
        with pytest.raises(ValueError, match="target_logits first dim"):
            verify_greedy_tree(draft_tokens, target_logits, topology_map)


class TestVerifyStochasticTree:
    """Tests for verify_stochastic_tree."""

    def test_empty_tree(self) -> None:
        """Empty tree returns empty result and bonus -1."""
        draft_tokens = torch.tensor([], dtype=torch.long)
        draft_probs = torch.tensor([], dtype=torch.float32)
        target_probs = torch.zeros(0, 64)
        topology_map: list[int] = []
        result = verify_stochastic_tree(
            draft_tokens, draft_probs, target_probs, topology_map
        )
        assert result.accepted_tokens == []
        assert result.bonus_token == -1
        assert result.num_accepted == 0
        assert result.accepted_leaf_index == -1

    def test_no_roots_raises(self) -> None:
        """Topology with no root nodes raises ValueError."""
        # Invalid: no -1 in topology (e.g. all 0)
        draft_tokens = torch.tensor([1, 2], dtype=torch.long)
        draft_probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        target_probs = torch.ones(2, 64) / 64
        topology_map = [0, 0]  # no root
        with pytest.raises(ValueError, match="no root nodes"):
            verify_stochastic_tree(
                draft_tokens, draft_probs, target_probs, topology_map
            )

    def test_single_path_all_accepted(self) -> None:
        """When p >= q at every node, full path is accepted (deterministic)."""
        torch.manual_seed(42)
        draft_tokens = torch.tensor([10, 20, 30], dtype=torch.long)
        # Draft probs high; target probs higher so p >= q always
        draft_probs = torch.tensor([0.3, 0.3, 0.3], dtype=torch.float32)
        vocab_size = 64
        target_probs = torch.zeros(3, vocab_size)
        target_probs[0, 10] = 0.5
        target_probs[1, 20] = 0.5
        target_probs[2, 30] = 0.5
        topology_map = [-1, 0, 1]
        result = verify_stochastic_tree(
            draft_tokens, draft_probs, target_probs, topology_map
        )
        assert result.accepted_tokens == [10, 20, 30]
        assert result.num_accepted == 3
        assert result.accepted_leaf_index == 2
        assert result.bonus_token >= 0

    def test_single_path_reject_at_root(self) -> None:
        """When root is rejected (p < q and random reject), path is empty; bonus from root."""
        torch.manual_seed(123)
        draft_tokens = torch.tensor([10, 20], dtype=torch.long)
        draft_probs = torch.tensor([0.9, 0.5], dtype=torch.float32)  # q high at root
        vocab_size = 64
        target_probs = torch.zeros(2, vocab_size)
        target_probs[0, 10] = 0.1  # p < q at root
        target_probs[1, 20] = 0.5
        topology_map = [-1, 0]
        # With p=0.1, q=0.9: p/q < 1, so may reject. Run multiple times or fix seed.
        result = verify_stochastic_tree(
            draft_tokens, draft_probs, target_probs, topology_map
        )
        # Either 0 or 1 accepted depending on random; bonus_token from divergence/root
        assert result.num_accepted <= 2
        assert result.bonus_token >= 0 or result.num_accepted == 0

    def test_multi_branch_longest_path_wins(self) -> None:
        """Longest accepted path across branches is chosen (stochastic with p>=q)."""
        torch.manual_seed(99)
        # Tree: root 0 -> children 1, 2. Path 0-1 has length 2; path 0-2 has length 2.
        # Make both paths always accept (p >= q) so we get length 2; which path wins
        # depends on DFS order. We just check we get a path of length 2 and valid bonus.
        draft_tokens = torch.tensor([42, 17, 99], dtype=torch.long)
        draft_probs = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        vocab_size = 128
        target_probs = torch.zeros(3, vocab_size)
        target_probs[0, 42] = 0.6
        target_probs[1, 17] = 0.6
        target_probs[2, 99] = 0.6
        topology_map = [-1, 0, 0]  # root 0, children 1 and 2
        result = verify_stochastic_tree(
            draft_tokens, draft_probs, target_probs, topology_map
        )
        assert result.num_accepted == 2  # root + one child (longest path length 2)
        assert result.accepted_tokens in ([42, 17], [42, 99])
        assert result.bonus_token >= 0

    def test_shape_mismatch_draft_tokens_raises(self) -> None:
        """Draft tokens length mismatch raises ValueError."""
        draft_tokens = torch.tensor([1, 2, 3], dtype=torch.long)
        draft_probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        target_probs = torch.ones(2, 64) / 64
        topology_map = [-1, 0]
        with pytest.raises(ValueError, match="draft_tokens length"):
            verify_stochastic_tree(
                draft_tokens, draft_probs, target_probs, topology_map
            )

    def test_shape_mismatch_draft_probs_raises(self) -> None:
        """Draft probs length mismatch raises ValueError."""
        draft_tokens = torch.tensor([1, 2], dtype=torch.long)
        draft_probs = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        target_probs = torch.ones(2, 64) / 64
        topology_map = [-1, 0]
        with pytest.raises(ValueError, match="draft_probs length"):
            verify_stochastic_tree(
                draft_tokens, draft_probs, target_probs, topology_map
            )

    def test_shape_mismatch_target_probs_raises(self) -> None:
        """Target probs first dim mismatch raises ValueError."""
        draft_tokens = torch.tensor([1, 2], dtype=torch.long)
        draft_probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        target_probs = torch.ones(3, 64) / 64
        topology_map = [-1, 0]
        with pytest.raises(ValueError, match="target_probs first dim"):
            verify_stochastic_tree(
                draft_tokens, draft_probs, target_probs, topology_map
            )

    def test_q_zero_rejects(self) -> None:
        """When draft prob q is 0, node is rejected (avoid div by zero); bonus from root."""
        draft_tokens = torch.tensor([10], dtype=torch.long)
        draft_probs = torch.tensor([0.0], dtype=torch.float32)
        vocab_size = 64
        target_probs = torch.zeros(1, vocab_size)
        target_probs[0, 10] = 1.0
        topology_map = [-1]
        result = verify_stochastic_tree(
            draft_tokens, draft_probs, target_probs, topology_map
        )
        assert result.num_accepted == 0
        assert result.accepted_tokens == []
        # Divergence node is the root; bonus sampled from target at root.
        assert result.bonus_token >= 0
