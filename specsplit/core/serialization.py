"""Serialization utilities for converting between PyTorch tensors and Python lists.

These helpers are used at the gRPC boundary to convert tensors into protobuf-
compatible formats (lists of ints/floats) and back. All conversions are
device-aware and preserve dtype where applicable.
"""

from __future__ import annotations

import torch


def tensor_to_token_ids(tensor: torch.Tensor) -> list[int]:
    """Convert a 1-D tensor of token IDs to a plain Python list.

    Args:
        tensor: A 1-D ``torch.LongTensor`` (or compatible integer dtype)
            containing vocabulary indices.

    Returns:
        A Python list of ints suitable for protobuf serialization.

    Raises:
        ValueError: If the tensor has more than one dimension.

    Example::

        >>> ids = tensor_to_token_ids(torch.tensor([101, 2003, 1037]))
        >>> ids
        [101, 2003, 1037]
    """
    if tensor.ndim > 1:
        raise ValueError(
            f"Expected a 1-D tensor, got shape {tuple(tensor.shape)}. "
            "Flatten or index before converting."
        )
    return tensor.detach().cpu().tolist()


def token_ids_to_tensor(
    ids: list[int],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Convert a list of token IDs to a 1-D PyTorch tensor.

    Args:
        ids: A list of integer vocabulary indices.
        device: Target device (``"cpu"``, ``"cuda:0"``, etc.).
        dtype: Desired tensor dtype. Defaults to ``torch.long``.

    Returns:
        A 1-D tensor on the specified device.

    Example::

        >>> t = token_ids_to_tensor([101, 2003, 1037], device="cpu")
        >>> t.shape
        torch.Size([3])
    """
    return torch.tensor(ids, dtype=dtype, device=device)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """Convert raw logits to a probability distribution via softmax.

    Applies temperature scaling before softmax. A temperature of 0 is treated
    as greedy (argmax), returning a one-hot distribution.

    Args:
        logits: Raw logits from a language model, shape ``(..., vocab_size)``.
        temperature: Sampling temperature. Values < 1.0 sharpen the
            distribution; values > 1.0 flatten it.
        dim: Dimension along which to apply softmax.

    Returns:
        Probability tensor of the same shape as *logits*.

    Raises:
        ValueError: If temperature is negative.
    """
    if temperature < 0:
        raise ValueError(f"Temperature must be non-negative, got {temperature}")

    if temperature == 0:
        # Greedy: one-hot on the argmax
        indices = logits.argmax(dim=dim, keepdim=True)
        probs = torch.zeros_like(logits)
        probs.scatter_(dim, indices, 1.0)
        return probs

    scaled = logits / temperature
    return torch.softmax(scaled, dim=dim)
