"""Unit tests for specsplit.workers.orchestrator.pipeline — longest path and helpers."""

from __future__ import annotations

from specsplit.workers.orchestrator.pipeline import DraftTree, _get_longest_path


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
