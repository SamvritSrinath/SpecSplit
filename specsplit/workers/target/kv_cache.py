"""Pre-allocated Static KV Cache for Disaggregated Speculative Decoding.

Standard HuggingFace ``past_key_values`` uses a tuple of per-layer
``(key, value)`` tensors that grow via ``torch.cat`` at every decoding
step.  This incurs reallocation + copy overhead that is unacceptable
in a latency-sensitive speculative decoding pipeline.

This module provides a :class:`StaticKVCache` that **pre-allocates** the
full maximum-length buffer once at session start, and then uses simple
slice assignment (``tensor[:, :, ptr:ptr+n, :] = new_data``) for
appends and a single integer update for rollbacks.

Performance Properties
----------------------
- **Append**: O(n) in the number of *new* tokens only (slice write,
  no reallocation, no ``torch.cat``).
- **Rollback**: O(1) — updates the ``seq_len`` pointer.  Stale data
  beyond the pointer is ignored and overwritten on the next append.
- **Memory**: Fixed at ``2 * num_layers * batch * num_heads * max_seq_len
  * head_dim * sizeof(dtype)`` bytes.  No dynamic growth.

Tensor Layout
-------------
::

    key_cache:   [num_layers, batch, num_heads, max_seq_len, head_dim]
    value_cache: [num_layers, batch, num_heads, max_seq_len, head_dim]

The ``seq_len`` pointer tracks how many valid positions are stored.
Only ``[:, :, :, :seq_len, :]`` contains meaningful data.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class StaticKVCache:
    """Pre-allocated, pointer-based KV cache for a single session.

    Allocates fixed-size key and value buffers at initialization.
    Appends use slice assignment, and rollbacks update a single integer.

    Args:
        num_layers: Number of transformer layers (attention blocks).
        num_heads: Number of attention heads per layer.
        max_seq_len: Maximum sequence length to support.  The buffers
            are allocated to this size and never grow.
        head_dim: Dimension per attention head.
        batch_size: Batch dimension (typically 1 for inference).
        dtype: Tensor dtype (e.g., ``torch.float16``).
        device: Torch device (e.g., ``"cuda:0"``).

    Example::

        cache = StaticKVCache(
            num_layers=32, num_heads=32,
            max_seq_len=2048, head_dim=128,
            dtype=torch.float16, device="cuda:0",
        )

        # After a forward pass produces new KV projections:
        # new_keys shape:   [num_layers, batch, num_heads, new_len, head_dim]
        # new_values shape: [num_layers, batch, num_heads, new_len, head_dim]
        cache.append(new_keys, new_values)

        # After verification, roll back to accepted prefix:
        cache.rollback(accepted_length=128)  # O(1)!

        # Get KV for a specific layer to pass into model:
        k, v = cache.get_kv_for_layer(layer_idx=0)
        # k shape: [batch, num_heads, seq_len, head_dim]
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # -----------------------------------------------------------------
        # Pre-allocate the full KV buffers — filled with zeros.
        # Shape: [num_layers, batch_size, num_heads, max_seq_len, head_dim]
        # -----------------------------------------------------------------
        self.key_cache: torch.Tensor = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype,
            device=self.device,
        )
        self.value_cache: torch.Tensor = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype,
            device=self.device,
        )

        # Pointer: number of valid positions currently stored.
        self._seq_len: int = 0

        # Memory budget in MB (for logging)
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        total_elems = 2 * num_layers * batch_size * num_heads * max_seq_len * head_dim
        mem_mb = (total_elems * elem_bytes) / (1024 * 1024)

        logger.info(
            "StaticKVCache allocated: layers=%d, heads=%d, max_seq=%d, "
            "head_dim=%d, dtype=%s, device=%s, memory=%.1f MB",
            num_layers, num_heads, max_seq_len, head_dim,
            dtype, self.device, mem_mb,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def seq_len(self) -> int:
        """Number of valid positions currently stored in the cache."""
        return self._seq_len

    @property
    def remaining_capacity(self) -> int:
        """Number of additional positions that can be appended."""
        return self.max_seq_len - self._seq_len

    @property
    def is_full(self) -> bool:
        """Whether the cache has reached maximum capacity."""
        return self._seq_len >= self.max_seq_len

    # ------------------------------------------------------------------ #
    # Append
    # ------------------------------------------------------------------ #

    def append(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> None:
        """Insert new KV projections at the current pointer position.

        Uses slice assignment (``cache[:, :, :, ptr:ptr+n, :] = data``)
        which is an in-place write — **no** ``torch.cat``, **no**
        ``.clone()``, **no** reallocation.

        Args:
            new_keys: New key projections.
                Shape: ``[num_layers, batch, num_heads, new_len, head_dim]``.
            new_values: New value projections.
                Shape: ``[num_layers, batch, num_heads, new_len, head_dim]``.

        Raises:
            ValueError: If the new tokens would exceed ``max_seq_len``.
            ValueError: If tensor shapes don't match the cache configuration.
        """
        new_len = new_keys.shape[3]
        # Shape assertion: new_keys should be
        #   [num_layers, batch_size, num_heads, new_len, head_dim]

        if new_keys.shape != (self.num_layers, self.batch_size, self.num_heads,
                              new_len, self.head_dim):
            raise ValueError(
                f"new_keys shape {tuple(new_keys.shape)} doesn't match cache "
                f"config: expected [{self.num_layers}, {self.batch_size}, "
                f"{self.num_heads}, *, {self.head_dim}]"
            )

        if new_values.shape != new_keys.shape:
            raise ValueError(
                f"new_values shape {tuple(new_values.shape)} != "
                f"new_keys shape {tuple(new_keys.shape)}"
            )

        end = self._seq_len + new_len
        if end > self.max_seq_len:
            raise ValueError(
                f"Cannot append {new_len} positions: current seq_len="
                f"{self._seq_len}, max_seq_len={self.max_seq_len} "
                f"(would need {end})"
            )

        # In-place slice assignment — no allocation, no copy
        # Target region: [:, :, :, seq_len:seq_len+new_len, :]
        self.key_cache[:, :, :, self._seq_len:end, :] = new_keys
        self.value_cache[:, :, :, self._seq_len:end, :] = new_values

        old_len = self._seq_len
        self._seq_len = end

        logger.debug(
            "KV cache append: %d → %d positions (+%d new)",
            old_len, self._seq_len, new_len,
        )

    # ------------------------------------------------------------------ #
    # Rollback — O(1)
    # ------------------------------------------------------------------ #

    def rollback(self, accepted_length: int) -> None:
        """Roll back the cache to a given accepted prefix length.

        **This is an O(1) operation.**  It simply moves the ``seq_len``
        pointer backwards.  No tensor data is copied or zeroed.  Stale
        data beyond the new pointer is harmless — it will be overwritten
        by the next :meth:`append` call, and it is never read because
        all consumers use ``[:, :, :, :seq_len, :]`` slices.

        Args:
            accepted_length: The number of positions to keep (measured
                from the start of the sequence).  Must satisfy
                ``0 <= accepted_length <= seq_len``.

        Raises:
            ValueError: If ``accepted_length`` is negative or exceeds
                the current ``seq_len``.
        """
        if accepted_length < 0:
            raise ValueError(
                f"accepted_length must be >= 0, got {accepted_length}"
            )
        if accepted_length > self._seq_len:
            raise ValueError(
                f"accepted_length ({accepted_length}) exceeds current "
                f"seq_len ({self._seq_len})"
            )

        if accepted_length == self._seq_len:
            logger.debug("Rollback no-op: accepted_length == seq_len (%d)", self._seq_len)
            return

        old_len = self._seq_len
        self._seq_len = accepted_length
        # That's it.  O(1).  No tensor ops.

        logger.debug(
            "KV cache rollback: %d → %d positions (freed %d)",
            old_len, self._seq_len, old_len - self._seq_len,
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def get_kv_for_layer(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the valid (key, value) tensors for a specific layer.

        Returns views (not copies) sliced to the current ``seq_len``.
        These are suitable for passing directly into a transformer layer's
        attention computation.

        Args:
            layer_idx: Index of the transformer layer (0-based).

        Returns:
            A ``(key, value)`` tuple of tensors, each with shape
            ``[batch, num_heads, seq_len, head_dim]``.

        Raises:
            IndexError: If ``layer_idx`` is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range "
                f"[0, {self.num_layers})"
            )

        # Return a VIEW into the pre-allocated buffer — no copy.
        # Shape: [batch, num_heads, seq_len, head_dim]
        key = self.key_cache[layer_idx, :, :, :self._seq_len, :]
        value = self.value_cache[layer_idx, :, :, :self._seq_len, :]
        return key, value

    def get_all_kv(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Get valid (key, value) pairs for ALL layers as a HF-compatible tuple.

        This format is compatible with HuggingFace's ``past_key_values``
        argument: a tuple of ``(key, value)`` pairs per layer.

        Returns:
            A tuple of length ``num_layers``, where each element is
            ``(key, value)`` with shape ``[batch, num_heads, seq_len, head_dim]``.

        Note:
            The returned tensors are **views** into the pre-allocated cache.
            Do not modify them in-place unless you intend to update the cache.
        """
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(self.num_layers):
            k, v = self.get_kv_for_layer(i)
            layers.append((k, v))
        return tuple(layers)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the cache to empty (seq_len = 0).

        Like :meth:`rollback`, this is O(1) — it only resets the pointer.
        The pre-allocated buffers remain in GPU memory for reuse.
        """
        self._seq_len = 0
        logger.debug("KV cache reset to empty (buffers retained)")

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"StaticKVCache(layers={self.num_layers}, heads={self.num_heads}, "
            f"seq_len={self._seq_len}/{self.max_seq_len}, "
            f"head_dim={self.head_dim}, dtype={self.dtype}, "
            f"device={self.device})"
        )
