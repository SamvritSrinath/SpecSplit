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

.. note::

    This module is **not yet wired** into :class:`TargetEngine`, which
    currently uses HuggingFace's standard ``past_key_values`` tuple
    with slice-based rollback.  ``StaticKVCache`` is prepared as a
    future performance optimization to eliminate ``torch.cat``
    reallocation overhead in the verification hot path.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

try:
    from transformers.cache_utils import Cache as HFCache
except ImportError:
    HFCache = None  # type: ignore[misc, assignment]


class StaticKVCache(HFCache if HFCache is not None else object):  # type: ignore[misc]
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
        if HFCache is not None:
            super().__init__(layers=[])  # Bypass HF layers; we override update()
        elif not hasattr(self, "layers"):  # Fallback when transformers absent
            self.layers = []
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # -----------------------------------------------------------------
        # Pre-allocate the full KV buffers (empty — no zeroing).
        # _seq_len tracks valid data; retrieval only reads [:seq_len].
        # Zeroing would synchronously write ~2GB for 7B/4K, providing no benefit.
        # Shape: [num_layers, batch_size, num_heads, max_seq_len, head_dim]
        # -----------------------------------------------------------------
        self.key_cache: torch.Tensor = torch.empty(
            num_layers,
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=self.device,
        )
        self.value_cache: torch.Tensor = torch.empty(
            num_layers,
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
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
            num_layers,
            num_heads,
            max_seq_len,
            head_dim,
            dtype,
            self.device,
            mem_mb,
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

        if new_keys.shape != (
            self.num_layers,
            self.batch_size,
            self.num_heads,
            new_len,
            self.head_dim,
        ):
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
        self.key_cache[:, :, :, self._seq_len : end, :] = new_keys
        self.value_cache[:, :, :, self._seq_len : end, :] = new_values

        old_len = self._seq_len
        self._seq_len = end

        logger.debug(
            "KV cache append: %d → %d positions (+%d new)",
            old_len,
            self._seq_len,
            new_len,
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
            raise ValueError(f"accepted_length must be >= 0, got {accepted_length}")
        if accepted_length > self._seq_len:
            raise ValueError(
                f"accepted_length ({accepted_length}) exceeds current seq_len ({self._seq_len})"
            )

        if accepted_length == self._seq_len:
            logger.debug("Rollback no-op: accepted_length == seq_len (%d)", self._seq_len)
            return

        old_len = self._seq_len
        self._seq_len = accepted_length
        # That's it.  O(1).  No tensor ops.

        logger.debug(
            "KV cache rollback: %d → %d positions (freed %d)",
            old_len,
            self._seq_len,
            old_len - self._seq_len,
        )

    def compact(self, keep_indices: list[int]) -> None:
        """Re-order the cache to keep only specified positions.

        Used for branching tree KV cache compaction where the accepted
        path contains non-contiguous BFS positions.  Unlike
        :meth:`rollback`, this performs actual tensor operations
        (``torch.index_select``), but it is still far cheaper than
        re-computing the full KV cache from scratch.

        Args:
            keep_indices: Ordered list of cache positions to retain.
                Must be valid indices in ``[0, seq_len)``.

        Raises:
            ValueError: If any index is out of range.
        """
        if not keep_indices:
            self._seq_len = 0
            logger.debug("KV cache compact: cleared (empty keep_indices)")
            return

        max_idx = max(keep_indices)
        if max_idx >= self._seq_len:
            raise ValueError(
                f"compact index {max_idx} >= current seq_len {self._seq_len}"
            )

        idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=self.device)
        new_len = len(keep_indices)

        # index_select along the seq_len dimension (dim=3)
        selected_keys = torch.index_select(self.key_cache, 3, idx_tensor)
        selected_values = torch.index_select(self.value_cache, 3, idx_tensor)

        # Write back into the pre-allocated buffer
        self.key_cache[:, :, :, :new_len, :] = selected_keys
        self.value_cache[:, :, :, :new_len, :] = selected_values

        old_len = self._seq_len
        self._seq_len = new_len

        logger.debug(
            "KV cache compact: %d → %d positions (kept %d indices)",
            old_len,
            self._seq_len,
            new_len,
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
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        # Return a VIEW into the pre-allocated buffer — no copy.
        # Shape: [batch, num_heads, seq_len, head_dim]
        key = self.key_cache[layer_idx, :, :, : self._seq_len, :]
        value = self.value_cache[layer_idx, :, :, : self._seq_len, :]
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
    # HuggingFace Adapter
    # ------------------------------------------------------------------ #

    @staticmethod
    def stack_hf_cache(
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert HuggingFace's cache format to 5D tensors for ``append()``.

        HuggingFace models output ``past_key_values`` as::

            tuple[  # num_layers
                tuple[
                    Tensor[batch, heads, seq, head_dim],  # keys
                    Tensor[batch, heads, seq, head_dim],  # values
                ],
                ...
            ]

        This method stacks them into unified 5D tensors::

            keys:   [num_layers, batch, heads, seq, head_dim]
            values: [num_layers, batch, heads, seq, head_dim]

        Args:
            past_key_values: HuggingFace ``past_key_values`` tuple.

        Returns:
            A ``(stacked_keys, stacked_values)`` tuple of 5D tensors
            suitable for passing to ``StaticKVCache.append()``.
        """
        keys = torch.stack([layer[0] for layer in past_key_values], dim=0)
        values = torch.stack([layer[1] for layer in past_key_values], dim=0)
        return keys, values

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
    # HuggingFace Cache interface — native in-place updates (no torch.cat)
    # ------------------------------------------------------------------ #

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache in-place. Called by HuggingFace attention layers.

        Writes key/value states directly into the pre-allocated buffers via
        slice assignment or index_copy_. No torch.cat, no reallocation.

        Args:
            key_states: [batch, num_heads, new_len, head_dim]
            value_states: Same shape.
            layer_idx: Layer index.
            cache_kwargs: May contain "cache_position" tensor of write indices.

        Returns:
            (keys, values) for this layer, shape [batch, num_heads, seq_len, head_dim].
        """
        new_len = key_states.shape[2]
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")

        # Layer slice: [batch, num_heads, max_seq_len, head_dim]
        k_slice = self.key_cache[layer_idx]
        v_slice = self.value_cache[layer_idx]

        if cache_position is not None:
            if not isinstance(cache_position, torch.Tensor):
                cache_position = torch.tensor(
                    cache_position, dtype=torch.long, device=key_states.device
                )
            try:
                k_slice.index_copy_(2, cache_position, key_states)
                v_slice.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                k_slice[:, :, cache_position] = key_states
                v_slice[:, :, cache_position] = value_states
                
            # PR-3: Determine the intended sequence length dynamically.
            # We must return the full sliced tensor up to this length for ALL layers,
            # even though self._seq_len is only advanced on the final layer.
            intended_new_seq_len = max(self._seq_len, int(cache_position.max().item()) + 1)
            
            if layer_idx == self.num_layers - 1:
                self._seq_len = intended_new_seq_len
        else:
            # Contiguous append. Only advance _seq_len on last layer to avoid
            # corrupting subsequent layers (they would write to wrong offsets).
            start = self._seq_len
            end = start + new_len
            k_slice[:, :, start:end, :] = key_states
            v_slice[:, :, start:end, :] = value_states
            
            intended_new_seq_len = end
            
            if layer_idx == self.num_layers - 1:
                self._seq_len = intended_new_seq_len

        # PR-3 Fix: Return a slice up to `intended_new_seq_len`, NOT `self._seq_len`.
        # Why is this safe? On a cache hit during the first layer (layer_idx=0), 
        # self._seq_len might still be the old length (or 0 if misconfigured), but the 
        # pre-allocated buffers already contain the valid prefix data! Returning the 
        # larger slice simply exposes the existing valid data + the newly written data 
        # to the attention operation for this specific layer.
        return (
            k_slice[:, :, : intended_new_seq_len, :],
            v_slice[:, :, : intended_new_seq_len, :],
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the number of cached positions (Cache interface)."""
        return self._seq_len

    def crop(self, max_length: int) -> None:
        """Crop cache to max_length (Cache interface, maps to rollback)."""
        self.rollback(max_length)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Keep only specified positions (Cache interface, maps to compact)."""
        self.compact(indices.cpu().tolist())

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


class VirtualKVCache(StaticKVCache):
    """Wraps StaticKVCache with a virtual index layer.

    Instead of physically compacting the buffer on branching rollback,
    maintains a `_l2p` mapping (logical-to-physical). Reads use gathered indices;
    writes use scatter. Compact becomes an O(length) pointer swap.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._l2p: list[int] = list(range(self.max_seq_len))

    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        new_len = new_keys.shape[3]
        if new_keys.shape != (self.num_layers, self.batch_size, self.num_heads, new_len, self.head_dim):
            raise ValueError("new_keys shape mismatch")
        end = self._seq_len + new_len
        if end > self.max_seq_len:
            raise ValueError("Exceeds max_seq_len")

        # Gather the physical indices for these logical positions
        phys_indices = torch.tensor(self._l2p[self._seq_len:end], dtype=torch.long, device=self.device)
        self.key_cache.index_copy_(3, phys_indices, new_keys)
        self.value_cache.index_copy_(3, phys_indices, new_values)

        self._seq_len = end

    def compact(self, keep_indices: list[int]) -> None:
        if not keep_indices:
            self._seq_len = 0
            return

        kept_physical = [self._l2p[i] for i in keep_indices]
        keep_set = set(keep_indices)
        freed_physical = [self._l2p[i] for i in range(self._seq_len) if i not in keep_set]
        
        # logical length becomes len(keep_indices)
        # unaliasing the freed physical slots to the end
        self._l2p[:len(kept_physical)] = kept_physical
        self._l2p[len(kept_physical) : len(kept_physical) + len(freed_physical)] = freed_physical
        
        old_len = self._seq_len
        self._seq_len = len(keep_indices)
        logger.debug("VirtualKVCache compact: %d → %d", old_len, self._seq_len)

    def get_kv_for_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        phys_indices = torch.tensor(self._l2p[:self._seq_len], dtype=torch.long, device=self.device)
        key = self.key_cache[layer_idx].index_select(2, phys_indices)
        value = self.value_cache[layer_idx].index_select(2, phys_indices)
        return key, value

    def get_physical_indices(self) -> torch.Tensor:
        return torch.tensor(self._l2p[:self._seq_len], dtype=torch.long, device=self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_len = key_states.shape[2]
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")

        k_slice = self.key_cache[layer_idx]
        v_slice = self.value_cache[layer_idx]

        if cache_position is not None:
            if not isinstance(cache_position, torch.Tensor):
                cache_position = torch.tensor(
                    cache_position, dtype=torch.long, device=key_states.device
                )
            
            # Map logical positions to physical indices
            physical_pos = torch.tensor(
                [self._l2p[i] for i in cache_position.tolist()], 
                dtype=torch.long, 
                device=key_states.device
            )

            try:
                k_slice.index_copy_(2, physical_pos, key_states)
                v_slice.index_copy_(2, physical_pos, value_states)
            except NotImplementedError:
                k_slice[:, :, physical_pos] = key_states
                v_slice[:, :, physical_pos] = value_states
                
            intended_new_seq_len = max(self._seq_len, int(cache_position.max().item()) + 1)
            if layer_idx == self.num_layers - 1:
                self._seq_len = intended_new_seq_len
        else:
            start = self._seq_len
            end = start + new_len
            
            physical_pos = torch.tensor(self._l2p[start:end], dtype=torch.long, device=self.device)

            k_slice.index_copy_(2, physical_pos, key_states)
            v_slice.index_copy_(2, physical_pos, value_states)
            
            intended_new_seq_len = end
            if layer_idx == self.num_layers - 1:
                self._seq_len = intended_new_seq_len

        # Gather output using physical indices
        active_phys_indices = torch.tensor(self._l2p[:intended_new_seq_len], dtype=torch.long, device=self.device)
        return (
            k_slice.index_select(2, active_phys_indices),
            v_slice.index_select(2, active_phys_indices),
        )
