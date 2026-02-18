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
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Core: Build Tree Attention Mask & Position IDs
# ============================================================================


def build_tree_attention(
    topology_map: list[int],
    prefix_length: int,
    device: torch.device | str = "cpu",
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

    Returns:
        A tuple ``(attention_mask, position_ids)`` where:
        - ``attention_mask``: ``torch.bool`` tensor of shape
          ``[1, 1, total_len, total_len]`` (broadcastable over batch and
          heads).  ``True`` = allowed to attend, ``False`` = masked out.
          The 4D shape is what HuggingFace ``AutoModelForCausalLM``
          expects for custom attention masks.
        - ``position_ids``: ``torch.long`` tensor of shape ``[1, total_len]``.
          Siblings at the same tree depth share the same position ID.

    Raises:
        ValueError: If ``topology_map`` contains an out-of-range parent
            index or forms a cycle.
    """
    num_tree_nodes = len(topology_map)
    total_len = prefix_length + num_tree_nodes
    # Shape: [total_len, total_len]  (will be unsqueezed to 4D at the end)

    # ------------------------------------------------------------------
    # Step 1: Compute each node's depth and ancestor set
    # ------------------------------------------------------------------
    # depths[i] = depth of tree node i (root = 0)
    depths: list[int] = [0] * num_tree_nodes
    # ancestors[i] = set of LOCAL tree-node indices that are ancestors of i
    # (including i itself)
    ancestors: list[set[int]] = [set() for _ in range(num_tree_nodes)]

    for i in range(num_tree_nodes):
        # Walk up the parent chain to compute depth and ancestor set
        visited: set[int] = set()
        cur = i
        depth = 0
        chain: list[int] = [i]

        while cur != -1:
            if cur in visited:
                raise ValueError(
                    f"Cycle detected in topology_map at node {i}: revisited node {cur}"
                )
            if cur < -1 or cur >= num_tree_nodes:
                raise ValueError(
                    f"topology_map[{i}] references out-of-range parent "
                    f"index {cur} (num_tree_nodes={num_tree_nodes})"
                )
            visited.add(cur)
            parent = topology_map[cur]
            if parent != -1:
                depth += 1
                chain.append(parent)
            cur = parent

        depths[i] = depth
        ancestors[i] = set(chain)

    # ------------------------------------------------------------------
    # Step 2: Build the 2D boolean attention mask
    # ------------------------------------------------------------------
    # Start with a standard causal mask for the prefix (lower-triangular)
    # and then fill in the tree portion.
    #
    # Convention: mask[i, j] = True  means position i CAN attend to position j.
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    # Shape: [total_len, total_len]

    # (a) Prefix self-attention: standard causal (lower-triangular)
    if prefix_length > 0:
        prefix_causal = torch.tril(
            torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=device)
        )
        # Shape: [prefix_length, prefix_length]
        mask[:prefix_length, :prefix_length] = prefix_causal

    # (b) Tree nodes attending to prefix: every tree node sees the full prefix
    if prefix_length > 0 and num_tree_nodes > 0:
        # mask[prefix_length:, :prefix_length] = True
        mask[prefix_length:, :prefix_length] = True
        # Shape of region: [num_tree_nodes, prefix_length]

    # (c) Tree nodes attending to other tree nodes (including self):
    # node i attends to node j IFF j ∈ ancestors[i]
    for i in range(num_tree_nodes):
        row = prefix_length + i  # Global position of tree node i
        for anc in ancestors[i]:
            col = prefix_length + anc  # Global position of ancestor
            mask[row, col] = True

    # ------------------------------------------------------------------
    # Step 3: Build position IDs
    # ------------------------------------------------------------------
    # Prefix: 0, 1, ..., prefix_length - 1
    # Tree node i: prefix_length + depths[i]
    position_ids = torch.zeros(total_len, dtype=torch.long, device=device)
    # Shape: [total_len]

    if prefix_length > 0:
        position_ids[:prefix_length] = torch.arange(prefix_length, dtype=torch.long, device=device)

    for i in range(num_tree_nodes):
        position_ids[prefix_length + i] = prefix_length + depths[i]

    # ------------------------------------------------------------------
    # Step 4: Reshape for HuggingFace compatibility
    # ------------------------------------------------------------------
    # HF expects attention_mask of shape [batch, 1, seq_len, seq_len]
    # (the head dimension is broadcast).
    # position_ids: [batch, seq_len]
    attention_mask = mask.unsqueeze(0).unsqueeze(0)
    # Shape: [1, 1, total_len, total_len]

    position_ids = position_ids.unsqueeze(0)
    # Shape: [1, total_len]

    logger.debug(
        "Built tree attention: prefix_len=%d, tree_nodes=%d, total=%d, depths=%s",
        prefix_length,
        num_tree_nodes,
        total_len,
        depths,
    )

    return attention_mask, position_ids


# ============================================================================
# Integration: How to use with AutoModelForCausalLM
# ============================================================================


def tree_attention_forward(
    model: Any,  # transformers.AutoModelForCausalLM
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Any | None = None,
    use_cache: bool = True,
) -> Any:
    """Run a single forward pass with custom tree attention masking.

    This function demonstrates how to integrate the tree attention mask
    produced by :func:`build_tree_attention` into a standard HuggingFace
    ``AutoModelForCausalLM.forward()`` call.

    Args:
        model: A HuggingFace ``AutoModelForCausalLM`` instance (or any model
            whose ``.forward()`` accepts ``attention_mask`` and
            ``position_ids``).
        input_ids: Token IDs for the current step.
            Shape: ``[batch, seq_len]`` where ``seq_len`` may be just the
            new tree tokens if ``past_key_values`` covers the prefix.
        attention_mask: The 4D boolean tree-attention mask from
            :func:`build_tree_attention`.
            Shape: ``[1, 1, total_len, total_len]``.
            When using KV cache, this should be sliced to
            ``[1, 1, new_tokens, total_len]`` because only the new tokens
            need query-side rows, but the key-side spans the full context.
        position_ids: Position IDs from :func:`build_tree_attention`.
            Shape: ``[1, total_len]``.
            When using KV cache, slice to ``[1, new_tokens]`` for the
            new tree tokens only.
        past_key_values: Pre-computed KV cache from previous steps.
            Either a HuggingFace ``DynamicCache`` / tuple, or a
            ``StaticKVCache`` (see ``kv_cache.py``).
        use_cache: Whether to return updated KV cache in outputs.

    Returns:
        Model output object (``CausalLMOutputWithPast``), containing:
        - ``.logits``: shape ``[batch, seq_len, vocab_size]``
        - ``.past_key_values``: updated KV cache (if ``use_cache=True``)

    Example::

        # 1. Build the tree attention for a draft with 5 nodes
        topology_map = [-1, 0, 0, 1, 2]  # binary tree
        prefix_length = 128
        attn_mask, pos_ids = build_tree_attention(
            topology_map, prefix_length, device="cuda:0"
        )

        # 2. Prepare input_ids: prefix + tree tokens (flattened)
        #    If we have a KV cache covering the prefix, only pass tree tokens
        tree_token_ids = torch.tensor([[42, 17, 99, 8, 55]], device="cuda:0")
        # Shape: [1, 5]

        # 3. Slice mask & position_ids for the new tokens only
        #    (prefix is already in the KV cache)
        new_attn_mask = attn_mask[:, :, prefix_length:, :]
        # Shape: [1, 1, 5, 133]  — 5 query rows, 133 total key positions
        new_pos_ids = pos_ids[:, prefix_length:]
        # Shape: [1, 5]

        # 4. Forward pass
        outputs = tree_attention_forward(
            model, tree_token_ids, new_attn_mask, new_pos_ids,
            past_key_values=kv_cache,
        )
        tree_logits = outputs.logits  # Shape: [1, 5, vocab_size]

    Note:
        For models that expect ``attention_mask`` as a float tensor with
        ``-inf`` for masked positions (e.g., some FlashAttention-v2 configs),
        convert the boolean mask::

            float_mask = torch.where(bool_mask, 0.0, float('-inf'))
    """
    # Ensure the boolean mask is converted to the format expected by the
    # specific model architecture.  Most HF models accept bool directly
    # since transformers v4.38+, but we provide a float fallback.
    #
    # NOTE: We do NOT convert here by default — the caller should convert
    # if their model requires float masks.  This keeps the function
    # architecture-agnostic.

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    return outputs


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
