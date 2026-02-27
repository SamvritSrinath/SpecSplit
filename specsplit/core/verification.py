"""Verification Mathematics for Disaggregated Speculative Decoding.

This module implements the core acceptance/rejection logic used by the
Target Worker to validate draft token trees.  We start with **strictly
greedy** decoding (temperature = 0.0) as a correctness baseline before
moving to stochastic rejection sampling.

Greedy Verification Algorithm
-----------------------------
Given:
    - ``draft_tokens[i]``: The token ID drafted at tree position ``i``.
    - ``target_logits[i]``: The target model's logit vector at position ``i``.
    - ``topology_map[i]``: The parent index of tree position ``i``
      (-1 for roots).

For each position ``i``, compute ``argmax(target_logits[i])``.  A
drafted token is **accepted** if it matches the target's greedy choice.

The algorithm then walks the topology map to find the **longest
continuous path from a root to a leaf** where every node along the
path is accepted.  The "bonus token" is the target's greedy choice
at the first divergence point (or at the accepted leaf, extending
the sequence by one).

This entire comparison is done on-device via ``torch.argmax`` and
boolean indexing — no CPU↔GPU synchronization until the final small
result extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Unified Result Dataclass
# ============================================================================


@dataclass(frozen=True)
class VerificationResult:
    """Unified result for both greedy and stochastic tree verification.

    Attributes:
        accepted_leaf_index: The local tree-node index of the last
            accepted node on the longest accepted path.  If no tokens
            were accepted, this is ``-1``.
        accepted_tokens: Ordered list of accepted token IDs along the
            longest accepted path (root → leaf).
        bonus_token: The target model's greedy/sampled choice at the
            divergence point (i.e., the token that *would* follow the
            accepted prefix).  This is always produced — it extends the
            output by one token for free.
        accepted_indices: The local tree-node indices of the accepted
            nodes, in path order (root → leaf).  Useful for KV cache
            rollback and position tracking.
        num_draft_tokens: Total number of draft tokens in the tree
            (for computing acceptance rate).
        diverged: Whether the path ended at a rejection point (True)
            or was fully accepted to a leaf node (False).  Used by
            the TargetEngine to correctly sample the bonus token from
            the right logit position.
    """

    accepted_leaf_index: int
    accepted_tokens: list[int]
    bonus_token: int
    accepted_indices: list[int]
    num_draft_tokens: int
    diverged: bool

    @property
    def num_accepted(self) -> int:
        """Number of draft tokens accepted."""
        return len(self.accepted_tokens)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of the tree that was accepted (0.0-1.0).

        Computed as ``num_accepted / num_draft_tokens``.  Note that this
        measures the *path* acceptance, not the full tree utilization.
        """
        if self.num_draft_tokens == 0:
            return 0.0
        return self.num_accepted / self.num_draft_tokens


# ============================================================================
# Core: Greedy Tree Verification
# ============================================================================


def verify_greedy_tree(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    topology_map: list[int],
) -> VerificationResult:
    """Verify a draft token tree against target logits using greedy decoding.

    All comparisons are performed on-device.  The only CPU↔GPU sync
    happens at the very end when extracting the small result lists.

    Args:
        draft_tokens: Flat tensor of drafted token IDs.
            Shape: ``[num_tree_nodes]``, dtype: ``torch.long``.
            ``draft_tokens[i]`` is the token drafted at tree position ``i``.
        target_logits: Target model's logit vectors at each tree position.
            Shape: ``[num_tree_nodes, vocab_size]``, dtype: ``torch.float*``.
            ``target_logits[i]`` corresponds to tree position ``i``.
        topology_map: List of parent indices for each tree node.
            ``topology_map[i] = j`` means node ``i``'s parent is ``j``.
            ``topology_map[i] = -1`` means node ``i`` is a root.
            Length: ``num_tree_nodes``.

    Returns:
        A :class:`VerificationResult` with the longest accepted path,
        the bonus token, and metadata.

    Raises:
        ValueError: If tensor shapes are inconsistent with the topology map.

    Example::

        >>> draft_tokens = torch.tensor([42, 17, 99, 8, 55])
        >>> # Suppose target argmax at each position is [42, 17, 99, 7, 55]
        >>> target_logits = torch.zeros(5, 100)
        >>> target_logits[0, 42] = 10.0  # matches draft
        >>> target_logits[1, 17] = 10.0  # matches draft
        >>> target_logits[2, 99] = 10.0  # matches draft
        >>> target_logits[3, 7]  = 10.0  # MISMATCH (draft=8, target=7)
        >>> target_logits[4, 55] = 10.0  # matches but parent rejected
        >>> topology_map = [-1, 0, 0, 1, 2]  # binary tree
        >>> result = verify_greedy_tree(draft_tokens, target_logits, topology_map)
        >>> result.accepted_tokens  # Longest: root(42) → left(17), stops at 8≠7
        [42, 17]
        >>> result.bonus_token  # Target says 7 where draft said 8
        7
    """
    num_tree_nodes = len(topology_map)

    # --- Input validation ---
    if draft_tokens.shape[0] != num_tree_nodes:
        raise ValueError(
            f"draft_tokens length ({draft_tokens.shape[0]}) != "
            f"topology_map length ({num_tree_nodes})"
        )
    if target_logits.shape[0] != num_tree_nodes:
        raise ValueError(
            f"target_logits first dim ({target_logits.shape[0]}) != "
            f"topology_map length ({num_tree_nodes})"
        )

    # --- Handle empty tree ---
    if num_tree_nodes == 0:
        logger.debug("Empty draft tree — nothing to verify")
        return VerificationResult(
            accepted_leaf_index=-1,
            accepted_tokens=[],
            bonus_token=-1,
            accepted_indices=[],
            num_draft_tokens=0,
            diverged=True,
        )

    # ------------------------------------------------------------------
    # Step 1: Compute target's greedy choices (on-device, no sync)
    # ------------------------------------------------------------------
    # target_choices[i] = argmax(target_logits[i])
    # Shape: [num_tree_nodes]
    target_choices: torch.Tensor = target_logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Step 2: Compute per-node acceptance (on-device boolean mask)
    # ------------------------------------------------------------------
    # accepted_mask[i] = True if draft_tokens[i] == target_choices[i]
    # Shape: [num_tree_nodes]
    accepted_mask: torch.Tensor = draft_tokens.eq(target_choices)

    # ------------------------------------------------------------------
    # Step 3: Build children lists from topology map (CPU, O(n))
    # ------------------------------------------------------------------
    # children[i] = list of child indices of node i
    children: list[list[int]] = [[] for _ in range(num_tree_nodes)]
    roots: list[int] = []
    for i, parent in enumerate(topology_map):
        if parent == -1:
            roots.append(i)
        else:
            children[parent].append(i)

    # ------------------------------------------------------------------
    # Step 4: Single CPU sync to get the boolean mask as a Python list
    # ------------------------------------------------------------------
    # This is the ONLY CPU↔GPU synchronization point.  We transfer a
    # small boolean tensor (num_tree_nodes elements) rather than the
    # full logits tensor.
    accepted_list: list[bool] = accepted_mask.cpu().tolist()
    # Also transfer target choices for bonus token extraction
    target_choices_list: list[int] = target_choices.cpu().tolist()

    # ------------------------------------------------------------------
    # Step 5: DFS to find the longest continuously-accepted path
    # ------------------------------------------------------------------
    # We use iterative DFS with explicit stack to avoid recursion limits.
    # Each stack entry is (node_index, current_path).
    best_path: list[int] = []
    best_divergence_node: int = -1  # Node where we extract the bonus token
    best_diverged: bool = True  # Whether the best path ended at a rejection

    # Stack entries: (node_index, path_so_far)
    stack: list[tuple[int, list[int]]] = [(r, []) for r in roots]

    while stack:
        node_idx, path = stack.pop()

        if not accepted_list[node_idx]:
            # This node is REJECTED.  The path up to (but not including)
            # this node is a candidate for the longest accepted path.
            if len(path) > len(best_path):
                best_path = path
                best_divergence_node = node_idx
                best_diverged = True
            continue

        # Node is accepted — extend the path
        new_path = [*path, node_idx]

        if not children[node_idx]:
            # Leaf node — this path is complete and fully accepted
            if len(new_path) > len(best_path):
                best_path = new_path
                # For a fully-accepted leaf, the bonus token comes from
                # the target's prediction at this leaf position.  We use
                # the target's argmax at this leaf as the "next" token.
                best_divergence_node = node_idx
                best_diverged = False  # Fully accepted to leaf
        else:
            # Internal node — continue DFS into children
            for child_idx in children[node_idx]:
                stack.append((child_idx, new_path))

    # ------------------------------------------------------------------
    # Step 6: Extract results
    # ------------------------------------------------------------------
    accepted_tokens: list[int] = [draft_tokens[idx].item() for idx in best_path]

    # Bonus token: the target's greedy choice at the divergence point.
    # - If the path ended at a rejection: bonus = target's choice there.
    # - If the path went all the way to a leaf (fully accepted):
    #   bonus = target's choice at the leaf (extends sequence by 1).
    if best_divergence_node >= 0:
        bonus_token = target_choices_list[best_divergence_node]
    else:
        # Edge case: all roots were rejected on the first token.
        # Use the target's choice at the first root as the bonus.
        bonus_token = target_choices_list[roots[0]] if roots else -1

    accepted_leaf_index = best_path[-1] if best_path else -1

    result = VerificationResult(
        accepted_leaf_index=accepted_leaf_index,
        accepted_tokens=accepted_tokens,
        bonus_token=bonus_token,
        accepted_indices=list(best_path),
        num_draft_tokens=num_tree_nodes,
        diverged=best_diverged,
    )

    logger.debug(
        "Greedy verification: %d/%d accepted (%.1f%%), bonus_token=%d, leaf_idx=%d, diverged=%s",
        result.num_accepted,
        num_tree_nodes,
        result.acceptance_rate * 100,
        bonus_token,
        accepted_leaf_index,
        best_diverged,
    )

    return result


def verify_stochastic_tree(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    topology_map: list[int],
    draft_probs_full: torch.Tensor | None = None,
) -> VerificationResult:
    """Perform stochastic verification with rejection sampling over the full tree.

    Explores all root-to-leaf paths via DFS. At each node, the draft token is
    accepted if p_target >= p_draft, otherwise accepted with probability
    p_target / p_draft. The longest accepted path across all branches is chosen;
    the bonus (correction) token is sampled from the target distribution at the
    divergence node (rejection point) or at the accepted leaf.

    Args:
        draft_tokens: [num_nodes] tensor of drafted token IDs.
        draft_probs: [num_nodes] tensor of probabilities the draft assigned to
            the drafted token at each node (legacy; used when draft_probs_full=None).
        target_probs: [num_nodes, vocab_size] tensor of target model probabilities.
        topology_map: List mapping node index to parent index (-1 for root).
        draft_probs_full: Optional [num_nodes, vocab_size] full draft distribution.
            When present, residual = max(0, P_target - P_draft) for correct sampling.

    Returns:
        VerificationResult with the longest accepted path and sampled bonus token.

    Raises:
        ValueError: If shapes are inconsistent or the tree has no root nodes.
    """
    num_tree_nodes = len(topology_map)
    if draft_tokens.shape[0] != num_tree_nodes:
        raise ValueError(
            f"draft_tokens length ({draft_tokens.shape[0]}) != "
            f"topology_map length ({num_tree_nodes})"
        )
    if draft_probs.shape[0] != num_tree_nodes:
        raise ValueError(
            f"draft_probs length ({draft_probs.shape[0]}) != "
            f"topology_map length ({num_tree_nodes})"
        )
    if target_probs.shape[0] != num_tree_nodes:
        raise ValueError(
            f"target_probs first dim ({target_probs.shape[0]}) != "
            f"topology_map length ({num_tree_nodes})"
        )

    if num_tree_nodes == 0:
        return VerificationResult(
            accepted_leaf_index=-1,
            accepted_tokens=[],
            bonus_token=-1,
            accepted_indices=[],
            num_draft_tokens=0,
            diverged=True,
        )

    children: dict[int, list[int]] = {}
    for i, p in enumerate(topology_map):
        children.setdefault(p, []).append(i)
    roots = children.get(-1, [])
    if not roots:
        raise ValueError("verify_stochastic_tree: draft tree has no root nodes")

    best_path: list[int] = []
    best_divergence_node = -1
    best_diverged: bool = True
    stack: list[tuple[int, list[int]]] = [(r, []) for r in roots]

    while stack:
        node_idx, path = stack.pop()
        token_id = draft_tokens[node_idx].item()
        p_val = target_probs[node_idx, token_id].item()
        q_val = draft_probs[node_idx].item()

        if q_val <= 0:
            accepted = False
        elif p_val >= q_val:
            accepted = True
        else:
            accepted = torch.rand(1).item() < (p_val / q_val)

        if not accepted:
            if len(path) > len(best_path):
                best_path = path
                best_divergence_node = node_idx
                best_diverged = True
            elif best_divergence_node < 0:
                # Rejection at root or first node: use this node for bonus sampling.
                best_divergence_node = node_idx
            continue

        new_path = [*path, node_idx]
        node_children = children.get(node_idx, [])

        if not node_children:
            # Leaf node — fully accepted path
            if len(new_path) > len(best_path):
                best_path = new_path
                best_divergence_node = node_idx
                best_diverged = False  # Fully accepted to leaf
        else:
            if len(new_path) > len(best_path):
                best_path = new_path
                best_divergence_node = node_idx
                best_diverged = False  # Tentatively not diverged (may update)
            for c in node_children:
                stack.append((c, new_path))

    accepted_tokens = [draft_tokens[i].item() for i in best_path]
    if best_divergence_node >= 0:
        p_dist = target_probs[best_divergence_node]
        if best_diverged:
            # Rejection: sample from normalized residual distribution.
            # residual(t) = max(0, P_target(t) - P_draft(t)) for all tokens.
            if draft_probs_full is not None:
                q_dist = draft_probs_full[best_divergence_node]
                residual_probs = torch.relu(p_dist - q_dist)
            else:
                # Legacy: only have p_draft at drafted token, assume 0 elsewhere.
                draft_token_id = draft_tokens[best_divergence_node].item()
                q_val = draft_probs[best_divergence_node]
                residual_probs = p_dist.clone()
                residual_probs[draft_token_id] = torch.relu(
                    p_dist[draft_token_id] - q_val
                )
            residual_sum = residual_probs.sum()
            if residual_sum > 0:
                bonus_token = torch.multinomial(
                    residual_probs / residual_sum, 1
                ).item()
            else:
                bonus_token = torch.multinomial(p_dist, 1).item()
        else:
            # Fully accepted path: sample from target distribution at leaf
            bonus_token = torch.multinomial(p_dist, 1).item()
    else:
        bonus_token = -1
    current_node = best_path[-1] if best_path else -1
    return VerificationResult(
        accepted_tokens=accepted_tokens,
        bonus_token=bonus_token,
        num_draft_tokens=num_tree_nodes,
        accepted_leaf_index=current_node,
        accepted_indices=list(best_path),
        diverged=best_diverged,
    )
