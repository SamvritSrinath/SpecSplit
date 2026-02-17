"""Shared pytest fixtures for SpecSplit tests."""

from __future__ import annotations

import pytest
import torch

from specsplit.core.config import DraftWorkerConfig, OrchestratorConfig, TargetWorkerConfig


@pytest.fixture
def draft_config() -> DraftWorkerConfig:
    """Draft worker config pointing at CPU for CI-friendly tests."""
    return DraftWorkerConfig(
        model_name="gpt2",
        device="cpu",
        max_draft_tokens=3,
        num_beams=2,
        temperature=1.0,
        grpc_port=50051,
    )


@pytest.fixture
def target_config() -> TargetWorkerConfig:
    """Target worker config pointing at CPU for CI-friendly tests."""
    return TargetWorkerConfig(
        model_name="gpt2",
        device="cpu",
        grpc_port=50052,
    )


@pytest.fixture
def orchestrator_config() -> OrchestratorConfig:
    """Orchestrator config with test-friendly defaults."""
    return OrchestratorConfig(
        draft_address="localhost:50051",
        target_address="localhost:50052",
        max_rounds=3,
        timeout_s=5.0,
        max_output_tokens=32,
    )


@pytest.fixture
def sample_token_ids() -> list[int]:
    """A sample list of token IDs for testing."""
    return [101, 2003, 1037, 3459, 102]


@pytest.fixture
def sample_tensor(sample_token_ids: list[int]) -> torch.Tensor:
    """A sample 1-D tensor of token IDs."""
    return torch.tensor(sample_token_ids, dtype=torch.long)


@pytest.fixture
def sample_logits() -> torch.Tensor:
    """A sample logits tensor (batch=1, vocab_size=5)."""
    return torch.tensor([[2.0, 1.0, 0.5, -1.0, -2.0]])
