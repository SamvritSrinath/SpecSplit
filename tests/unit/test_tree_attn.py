from __future__ import annotations

import torch

from specsplit.workers.target.tree_attn import build_tree_attention


def test_build_tree_attention_full_mask_includes_self_and_correct_positions() -> None:
    mask, position_ids = build_tree_attention(
        topology_map=[-1, 0, 0, 1, 2],
        prefix_length=3,
        device="cpu",
        tree_rows_only=False,
    )

    assert position_ids.squeeze(0).tolist() == [0, 1, 2, 3, 4, 4, 5, 5]

    full_mask = mask.squeeze(0).squeeze(0)
    assert torch.equal(full_mask[3], torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool))
    assert torch.equal(full_mask[4], torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool))
    assert torch.equal(full_mask[5], torch.tensor([1, 1, 1, 1, 0, 1, 0, 0], dtype=torch.bool))
    assert torch.equal(full_mask[6], torch.tensor([1, 1, 1, 1, 1, 0, 1, 0], dtype=torch.bool))
    assert torch.equal(full_mask[7], torch.tensor([1, 1, 1, 1, 0, 1, 0, 1], dtype=torch.bool))


def test_build_tree_attention_tree_rows_only_includes_self_and_correct_positions() -> None:
    mask, position_ids = build_tree_attention(
        topology_map=[-1, 0, 0, 1, 2],
        prefix_length=3,
        device="cpu",
        tree_rows_only=True,
    )

    assert position_ids.squeeze(0).tolist() == [3, 4, 4, 5, 5]

    tree_mask = mask.squeeze(0).squeeze(0)
    assert torch.equal(tree_mask[0], torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool))
    assert torch.equal(tree_mask[1], torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool))
    assert torch.equal(tree_mask[2], torch.tensor([1, 1, 1, 1, 0, 1, 0, 0], dtype=torch.bool))
    assert torch.equal(tree_mask[3], torch.tensor([1, 1, 1, 1, 1, 0, 1, 0], dtype=torch.bool))
    assert torch.equal(tree_mask[4], torch.tensor([1, 1, 1, 1, 0, 1, 0, 1], dtype=torch.bool))
