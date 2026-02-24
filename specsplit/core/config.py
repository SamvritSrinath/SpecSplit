"""Pydantic configuration models for SpecSplit services.

All settings can be overridden via environment variables with the ``SPECSPLIT_``
prefix. For example, ``SPECSPLIT_DRAFT_MODEL_NAME=gpt2`` overrides the default
draft model name.

Settings can also be loaded from a YAML or JSON config file using
:func:`load_config_file`. Priority (highest → lowest):
    1. Constructor kwargs / CLI arguments
    2. Environment variables (``SPECSPLIT_*`` prefix)
    3. Config file values
    4. Field defaults

Example YAML config file::

    orchestrator:
      draft_address: "localhost:50051"
      target_address: "localhost:50052"
      max_rounds: 50
      max_output_tokens: 1024
      tokenizer_model: "Qwen/Qwen2.5-7B-Instruct"
    draft:
      model_name: "Qwen/Qwen2.5-0.5B-Instruct"
      max_draft_tokens: 5
    target:
      model_name: "Qwen/Qwen2.5-7B-Instruct"

Usage::

    from specsplit.core.config import OrchestratorConfig, load_config_file

    file_cfg = load_config_file("config.yaml")
    cfg = OrchestratorConfig(**file_cfg.get("orchestrator", {}))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class DraftWorkerConfig(BaseSettings):
    """Configuration for the Draft Worker (small, fast LLM)."""

    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model identifier for the draft model.",
    )
    device: str = Field(
        default="cuda:0",
        description="Torch device string (e.g., 'cuda:0', 'cpu').",
    )
    max_draft_tokens: int = Field(
        default=5,
        ge=1,
        le=64,
        description="Maximum depth of the speculative draft tree (K).",
    )
    num_beams: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Branching factor at each tree level.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        description="Sampling temperature for draft generation.",
    )
    grpc_port: int = Field(
        default=50051,
        description="gRPC server port for the Draft Worker.",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        description="gRPC thread pool size.",
    )

    model_config = {"env_prefix": "SPECSPLIT_DRAFT_"}


class TargetWorkerConfig(BaseSettings):
    """Configuration for the Target Worker (large, accurate LLM)."""

    model_name: str = Field(
        default="meta-llama/Llama-2-7b-hf",
        description="HuggingFace model identifier for the target model.",
    )
    device: str = Field(
        default="cuda:0",
        description="Torch device string.",
    )
    grpc_port: int = Field(
        default=50052,
        description="gRPC server port for the Target Worker.",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        description="gRPC thread pool size.",
    )
    max_sessions: int = Field(
        default=16,
        ge=1,
        description="Maximum concurrent KV cache sessions. Oldest sessions "
        "are evicted (LRU) when this limit is reached.",
    )
    max_tree_nodes: int = Field(
        default=2048,
        ge=1,
        le=65536,
        description="Maximum number of nodes in a draft tree per VerifyDrafts request. "
        "Requests exceeding this are rejected to limit resource use.",
    )
    max_prompt_tokens: int = Field(
        default=8192,
        ge=1,
        description="Maximum prompt length (token count) per request. "
        "Requests exceeding this are rejected.",
    )

    model_config = {"env_prefix": "SPECSPLIT_TARGET_"}


class OrchestratorConfig(BaseSettings):
    """Configuration for the Orchestrator (pipeline coordinator)."""

    draft_address: str = Field(
        default="localhost:50051",
        description="gRPC address of the Draft Worker.",
    )
    target_address: str = Field(
        default="localhost:50052",
        description="gRPC address of the Target Worker.",
    )
    max_rounds: int = Field(
        default=20,
        ge=1,
        description="Maximum number of draft → verify rounds per prompt.",
    )
    timeout_s: float = Field(
        default=30.0,
        gt=0,
        description="Per-RPC timeout in seconds.",
    )
    max_output_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum total tokens to generate per prompt.",
    )
    max_draft_tokens: int = Field(
        default=5,
        ge=1,
        le=64,
        description="Draft tree depth (K / gamma) forwarded to the Draft Worker.",
    )
    # Task 4.1: Synthetic Latency Rig (milliseconds)
    simulated_rtt_ms: float = Field(default=0.0, description="Injected network latency per RPC")

    use_target_kv_cache: bool = Field(
        default=False,
        description="If False (naive mode), target verifies statelessly each round — no session KV cache, no flush. Use for testing that orchestrator/draft/target communication works.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature for verification. 0.0 = greedy, >0.0 = stochastic rejection sampling.",
    )
    tokenizer_model: str = Field(
        default="gpt2",
        description="HuggingFace model name for the tokenizer. Must match the target/draft model (e.g. Qwen2/Qwen2.5-7B-Instruct); otherwise acceptance is 0%% and output is gibberish.",
    )

    model_config = {"env_prefix": "SPECSPLIT_ORCH_"}


# ============================================================================
# Config File Loader
# ============================================================================


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML or JSON file.

    The file should contain top-level keys matching the service names:
    ``orchestrator``, ``draft``, and/or ``target``.  Each key maps to a
    dict of field names → values.

    Args:
        path: Path to the config file (``.yaml``, ``.yml``, or ``.json``).

    Returns:
        A dict with keys ``"orchestrator"``, ``"draft"``, ``"target"``
        (any or all may be absent if not specified in the file).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.

    Example::

        cfg = load_config_file("specsplit.yaml")
        orch = OrchestratorConfig(**cfg.get("orchestrator", {}))
        draft = DraftWorkerConfig(**cfg.get("draft", {}))
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    suffix = filepath.suffix.lower()
    text = filepath.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported config file format '{suffix}'. Use .yaml, .yml, or .json."
        )

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping, got {type(data).__name__}")

    logger.info("Loaded config from %s (sections: %s)", filepath, list(data.keys()))
    return data
