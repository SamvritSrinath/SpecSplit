"""Cache format conversion utilities for HuggingFace transformers compatibility.

Provides version-agnostic helpers to convert between legacy tuple format and
DynamicCache for causal language models used by SpecSplit.
"""

from __future__ import annotations

from typing import Any, cast

import torch


def cache_to_legacy(
    past_kv: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    """Convert a cache object to a legacy ``((k, v), ...)`` tuple when possible.

    This supports legacy tuples and the newer ``Cache`` API via either
    ``to_legacy_cache()`` or ``.layers``.
    """
    if past_kv is None:
        return None

    if isinstance(past_kv, tuple) and len(past_kv) > 0:
        first = past_kv[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            return past_kv  # type: ignore[return-value]

    if hasattr(past_kv, "to_legacy_cache"):
        # HF DynamicCache returns a tuple of (k, v) tensors; mypy can't
        # express the runtime return type for this polymorphic attribute.
        return cast(
            tuple[tuple[torch.Tensor, torch.Tensor], ...],
            past_kv.to_legacy_cache(),
        )

    if hasattr(past_kv, "layers") and isinstance(past_kv.layers, (list, tuple)):
        return tuple(
            (layer.keys, layer.values)
            for layer in past_kv.layers
            if getattr(layer, "is_initialized", True) and getattr(layer, "keys", None) is not None
        )

    return None


def cache_supports_crop(past_kv: Any) -> bool:
    """Return True when the cache can be safely rolled back in-place."""
    return isinstance(past_kv, tuple) or hasattr(past_kv, "crop")


def crop_cache(past_kv: Any, max_length: int) -> Any:
    """Crop a cache to ``max_length`` tokens."""
    if past_kv is None:
        return None

    if hasattr(past_kv, "crop"):
        past_kv.crop(max_length)
        return past_kv

    legacy = cache_to_legacy(past_kv)
    if legacy is None:
        raise TypeError(f"Unsupported cache type for cropping: {type(past_kv)!r}")

    return tuple(
        (
            key_states[:, :, :max_length, :],
            value_states[:, :, :max_length, :],
        )
        for key_states, value_states in legacy
    )


def batch_model_caches(caches: list[Any]) -> Any:
    """Combine single-item caches into one batched cache for a forward pass."""
    if not caches:
        return None

    first = caches[0]
    if first is None:
        return None

    legacy_caches: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = []
    for cache in caches:
        legacy = cache_to_legacy(cache)
        if legacy is None:
            raise TypeError(f"Unsupported cache type for batching: {type(cache)!r}")
        legacy_caches.append(legacy)

    num_layers = len(legacy_caches[0])
    return tuple(
        (
            torch.cat(
                [legacy_caches[item_idx][layer_idx][0] for item_idx in range(len(legacy_caches))],
                dim=0,
            ),
            torch.cat(
                [legacy_caches[item_idx][layer_idx][1] for item_idx in range(len(legacy_caches))],
                dim=0,
            ),
        )
        for layer_idx in range(num_layers)
    )


def slice_batch_item_from_cache(cache: Any, item_idx: int) -> Any:
    """Extract one batch item from a batched cache object."""
    if cache is None:
        return None

    legacy = cache_to_legacy(cache)
    if legacy is None:
        raise TypeError(f"Unsupported cache type for slicing: {type(cache)!r}")
    return tuple(
        (
            key_states[item_idx : item_idx + 1],
            value_states[item_idx : item_idx + 1],
        )
        for key_states, value_states in legacy
    )


def legacy_to_dynamic_cache(
    past_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    config: Any,
) -> Any:
    """Convert legacy (key, value) tuple cache to DynamicCache for model forward.

    Version-agnostic: tries from_legacy_cache first, then DynamicCache constructor,
    finally falls back to returning the legacy tuple (many models accept it).

    Args:
        past_kv: Legacy cache as tuple of (key, value) per layer.
        config: Model config (PreTrainedConfig) for DynamicCache.

    Returns:
        DynamicCache instance or the original tuple if conversion fails.
    """
    if past_kv is None or not isinstance(past_kv, tuple) or len(past_kv) == 0:
        return past_kv

    from transformers import DynamicCache

    # Try from_legacy_cache if available (transformers ~4.36+)
    if hasattr(DynamicCache, "from_legacy_cache"):
        try:
            return DynamicCache.from_legacy_cache(past_kv)
        except (TypeError, AttributeError):
            pass

    # Try DynamicCache constructor with ddp_cache_data and config
    try:
        return DynamicCache(ddp_cache_data=past_kv, config=config)
    except (TypeError, AttributeError, IndexError):
        pass

    # Try without config — creates one DynamicLayer per tuple entry (avoids
    # config/model layer-count mismatch, e.g. GPT-2 or mocks).
    try:
        return DynamicCache(ddp_cache_data=past_kv)
    except (TypeError, AttributeError, IndexError):
        pass

    # Fallback: return legacy tuple (many models accept it directly)
    return past_kv
