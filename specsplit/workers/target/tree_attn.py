"""Custom Tree Attention Masking for Disaggregated Speculative Decoding.

When the Target Worker receives a flat list of draft token IDs and a
"topology map" (a list of parent indices), it must verify the entire tree
in a **single** forward pass.  This module builds the custom 2D boolean
attention mask and 1D ``position_ids`` tensor required to achieve that.

Terminology
-----------
- **Topology map**: A list of length ``num_tree_nodes`` where
  ``topology_map[i]`` is the *local* index (0-based into the tree nodes
  array) of node ``i``'s parent, or ``-1`` if node ``i`` is a root of
  the tree.  This is the standard flat-tree representation used in
  SpecInfer / Medusa / Eagle.
- **Prefix**: The already-processed prompt tokens whose KV projections
  live in the KV cache.  Every tree node attends to the full prefix.
- **Total sequence**: ``[prefix tokens | tree tokens]`` concatenated.

Attention Rules
~~~~~~~~~~~~~~~
A tree node at position ``j`` (0-indexed within the tree) is allowed to
attend to:
    1. **All prefix tokens** (positions ``0 .. prefix_length-1``).
    2. **Itself**.
    3. **All of its ancestors** in the tree (following parent pointers up).

It must **NOT** attend to siblings, cousins, or any other branch.

Position ID Rules
~~~~~~~~~~~~~~~~~
Siblings at the same depth in the tree represent *alternative*
continuations at the same logical decoding step, so they share the same
``position_id``.  Specifically:
    - Prefix positions: ``0, 1, ..., prefix_length - 1``
    - Tree node positions: ``prefix_length + depth_of_node``

Example
-------
Consider a tree with topology_map = [-1, 0, 0, 1, 2] and prefix_length = 3::

        Prefix: [p0, p1, p2]
        Tree:       t0
                   /    \\
                 t1      t2
                 |       |
                 t3      t4

    Depths:  t0=0, t1=1, t2=1, t3=2, t4=2
    Position IDs: [0,1,2,  3,4,4,5,5]
                   prefix   tree

    Attention mask (tree portion only — prefix columns are all True):
        t0 attends to: prefix + t0
        t1 attends to: prefix + t0, t1
        t2 attends to: prefix + t0, t2
        t3 attends to: prefix + t0, t1, t3
        t4 attends to: prefix + t0, t2, t4
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Core: Build Tree Attention Mask & Position IDs
# ============================================================================


def build_tree_attention(
    topology_map: list[int],
    prefix_length: int,
    device: torch.device | str = "cpu",
    tree_rows_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a causal tree-attention mask and position IDs tensor.

    Args:
        topology_map: List of parent indices for each tree node.
            ``topology_map[i] = j`` means node ``i``'s parent is node ``j``.
            ``topology_map[i] = -1`` means node ``i`` is a root.
            Length: ``num_tree_nodes``.
        prefix_length: Number of prefix tokens already in the KV cache.
            These positions are always attended to by every tree node.
        device: Torch device for the output tensors.
        tree_rows_only: If True, allocate only [num_tree_nodes, total_len] mask
            (for cache hit when only tree rows are needed). Avoids O(total_len²)
            allocation when 99%+ would be discarded by slicing.

    Returns:
        A tuple ``(attention_mask, position_ids)`` where:
        - ``attention_mask``: ``torch.bool`` tensor of shape
          ``[1, 1, Q, total_len]`` with Q = num_tree_nodes if tree_rows_only
          else total_len. ``True`` = allowed to attend.
        - ``position_ids``: ``torch.long`` tensor of shape ``[1, Q]``.

    Raises:
        ValueError: If ``topology_map`` contains an out-of-range parent
            index or forms a cycle.
    """
    num_tree_nodes = len(topology_map)
    total_len = prefix_length + num_tree_nodes

    # ------------------------------------------------------------------
    # Step 1: Compute each node's depth and ancestor set
    # ------------------------------------------------------------------
    depths: list[int] = [0] * num_tree_nodes

    # PR-9: Fast O(N) Tree Attention Ancestor Computation
    # Assert that the topology map is topologically sorted (BFS order).
    # Since nodes refer to parents, the parent must be computed before the child.
    if num_tree_nodes > 0:
        assert all(topology_map[i] < i for i in range(num_tree_nodes) if topology_map[i] != -1), \
            "topology_map is not topologically sorted"

    # ------------------------------------------------------------------
    # Step 2: Build the attention mask (sparse or full)
    # ------------------------------------------------------------------
    if tree_rows_only and num_tree_nodes > 0:
        # Only allocate rows for tree nodes — avoids O(total_len²) when
        # cache hit would slice to last num_tree_nodes rows anyway.
        mask = torch.zeros(num_tree_nodes, total_len, dtype=torch.bool, device=device)
        for i in range(num_tree_nodes):
            if prefix_length > 0:
                mask[i, :prefix_length] = True
            
            parent = topology_map[i]
            if parent != -1:
                depths[i] = depths[parent] + 1
                # PR-9: Inherit parent's ancestors directly via boolean mask slice
                mask[i, prefix_length : prefix_length + i] = mask[parent, prefix_length : prefix_length + i]
                # Include the parent itself
                mask[i, prefix_length + parent] = True
            else:
                depths[i] = 1

        num_rows = num_tree_nodes
        position_ids_len = num_tree_nodes
    else:
        # Full mask: prefix causal + tree portion
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        if prefix_length > 0:
            prefix_causal = torch.tril(
                torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device)
            )
            mask[:prefix_length, :prefix_length] = prefix_causal
        if prefix_length > 0 and num_tree_nodes > 0:
            mask[prefix_length:, :prefix_length] = True
        for i in range(num_tree_nodes):
            row = prefix_length + i
            parent = topology_map[i]
            if parent != -1:
                depths[i] = depths[parent] + 1
                parent_row = prefix_length + parent
                # PR-9: Inherit parent's ancestors directly via boolean mask slice
                mask[row, prefix_length : row] = mask[parent_row, prefix_length : row]
                # Include the parent itself
                mask[row, parent_row] = True
            else:
                depths[i] = 1
        num_rows = total_len
        position_ids_len = total_len

    # ------------------------------------------------------------------
    # Step 3: Build position IDs
    # ------------------------------------------------------------------
    position_ids = torch.zeros(position_ids_len, dtype=torch.long, device=device)
    if tree_rows_only and num_tree_nodes > 0:
        for i in range(num_tree_nodes):
            position_ids[i] = prefix_length + depths[i]
    else:
        if prefix_length > 0:
            position_ids[:prefix_length] = torch.arange(
                prefix_length, dtype=torch.long, device=device
            )
        for i in range(num_tree_nodes):
            position_ids[prefix_length + i] = prefix_length + depths[i]

    # ------------------------------------------------------------------
    # Step 4: Reshape for HuggingFace compatibility
    # ------------------------------------------------------------------
    attention_mask = mask.unsqueeze(0).unsqueeze(0)
    position_ids = position_ids.unsqueeze(0)

    logger.debug(
        "Built tree attention: prefix_len=%d, tree_nodes=%d, total=%d, "
        "mask_rows=%d, tree_only=%s",
        prefix_length,
        num_tree_nodes,
        total_len,
        num_rows,
        tree_rows_only,
    )

    return attention_mask, position_ids





# ============================================================================
# Utility: Convert boolean mask to float mask (for Flash Attention compat)
# ============================================================================


def bool_mask_to_float(
    mask: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Convert a boolean attention mask to a float mask with ``-inf`` masking.

    Some model backends (e.g., Flash Attention v2) expect the attention
    mask as a float tensor where masked positions are ``-inf`` and
    attended positions are ``0.0``.

    Args:
        mask: Boolean mask of shape ``[1, 1, Q, K]``.
            ``True`` = attend, ``False`` = mask out.
        dtype: Output dtype (should match model precision).

    Returns:
        Float mask of the same shape, with ``0.0`` for attended and
        ``-inf`` for masked positions.
    """
    # Shape: [1, 1, Q, K] — unchanged
    float_mask = torch.zeros_like(mask, dtype=dtype)
    float_mask.masked_fill_(~mask, float("-inf"))
    return float_mask
