"""Helpers for loading local HuggingFace causal-LM checkpoints used by workers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import torch

logger = logging.getLogger(__name__)

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _get_explicit_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a mock-safe attribute."""
    if isinstance(obj, Mock) and name not in obj.__dict__:
        return default
    return getattr(obj, name, default)


def _read_local_config_json(model_name: str) -> dict[str, Any]:
    """Return the local ``config.json`` contents when ``model_name`` is a path."""
    config_path = Path(model_name) / "config.json"
    if not config_path.is_file():
        return {}

    try:
        data = json.loads(config_path.read_text())
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
        return {}
    except OSError as exc:
        logger.warning("Failed to read checkpoint config %s: %s", config_path, exc)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid checkpoint config %s: %s", config_path, exc)
    return {}


def _parse_dtype(value: Any) -> torch.dtype | None:
    """Return a torch dtype from a serialized config value."""
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return _DTYPE_ALIASES.get(value.removeprefix("torch.").lower())
    return None


def get_checkpoint_dtype(
    model_name: str,
    *,
    default: torch.dtype | None = None,
) -> torch.dtype | None:
    """Return the preferred compute dtype recorded in a local checkpoint."""
    config_json = _read_local_config_json(model_name)
    if config_json:
        quant_config = config_json.get("quantization_config") or {}
        for candidate in (
            quant_config.get("bnb_4bit_compute_dtype"),
            config_json.get("torch_dtype"),
        ):
            dtype = _parse_dtype(candidate)
            if dtype is not None:
                return dtype
    return default


def get_model_config(model_or_config: Any) -> Any:
    """Return a model config object from either a model or config."""
    return getattr(model_or_config, "config", model_or_config)


def get_model_vocab_size(model: Any) -> int:
    """Best-effort vocabulary size for a loaded causal language model."""
    config = get_model_config(model)
    try:
        vocab_size = int(_get_explicit_attr(config, "vocab_size", 0) or 0)
    except (TypeError, ValueError):
        vocab_size = 0
    if vocab_size > 0:
        return vocab_size

    lm_head = getattr(model, "lm_head", None)
    try:
        out_features = int(_get_explicit_attr(lm_head, "out_features", 0) or 0)
    except (TypeError, ValueError):
        out_features = 0
    if out_features > 0:
        return out_features

    return 0
