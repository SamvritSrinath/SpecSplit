"""Pydantic configuration models for SpecSplit services.

All settings can be overridden via environment variables with the ``SPECSPLIT_``
prefix. For example, ``SPECSPLIT_DRAFT_MODEL_NAME=gpt2`` overrides the default
draft model name.

Usage::

    from specsplit.core.config import DraftWorkerConfig

    cfg = DraftWorkerConfig()          # reads from env
    print(cfg.model_name, cfg.device)
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


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
        default=256,
        ge=1,
        description="Maximum total tokens to generate per prompt.",
    )
    max_draft_tokens: int = Field(
        default=5,
        ge=1,
        le=64,
        description="Draft tree depth (K / gamma) forwarded to the Draft Worker.",
    )

    model_config = {"env_prefix": "SPECSPLIT_ORCH_"}
