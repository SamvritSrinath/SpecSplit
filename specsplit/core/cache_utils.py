"""Cache format conversion utilities for HuggingFace transformers compatibility.

Provides version-agnostic helpers to convert between legacy tuple format and
DynamicCache, supporting different transformers API versions.
"""

from __future__ import annotations

from typing import Any

import torch


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
