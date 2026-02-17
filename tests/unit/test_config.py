"""Unit tests for specsplit.core.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from specsplit.core.config import DraftWorkerConfig, OrchestratorConfig, TargetWorkerConfig


class TestDraftWorkerConfig:
    """Tests for DraftWorkerConfig defaults and env overrides."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        """Default values should be sensible."""
        # Clear any env overrides so we test the true Field defaults.
        monkeypatch.delenv("SPECSPLIT_DRAFT_MODEL_NAME", raising=False)
        monkeypatch.delenv("SPECSPLIT_DRAFT_DEVICE", raising=False)
        monkeypatch.delenv("SPECSPLIT_DRAFT_MAX_DRAFT_TOKENS", raising=False)
        monkeypatch.delenv("SPECSPLIT_DRAFT_TEMPERATURE", raising=False)
        monkeypatch.delenv("SPECSPLIT_DRAFT_GRPC_PORT", raising=False)

        cfg = DraftWorkerConfig()
        assert cfg.model_name == "gpt2"
        assert cfg.device == "cuda:0"
        assert cfg.max_draft_tokens == 5
        assert cfg.temperature == 1.0
        assert cfg.grpc_port == 50051

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch):
        """Environment variables with SPECSPLIT_DRAFT_ prefix should override."""
        monkeypatch.setenv("SPECSPLIT_DRAFT_MODEL_NAME", "distilgpt2")
        monkeypatch.setenv("SPECSPLIT_DRAFT_DEVICE", "cpu")
        monkeypatch.setenv("SPECSPLIT_DRAFT_MAX_DRAFT_TOKENS", "10")

        cfg = DraftWorkerConfig()
        assert cfg.model_name == "distilgpt2"
        assert cfg.device == "cpu"
        assert cfg.max_draft_tokens == 10

    def test_validation_max_draft_tokens(self):
        """max_draft_tokens should respect ge=1, le=64 bounds."""
        with pytest.raises(ValidationError):
            DraftWorkerConfig(max_draft_tokens=0)

        with pytest.raises(ValidationError):
            DraftWorkerConfig(max_draft_tokens=100)


class TestTargetWorkerConfig:
    """Tests for TargetWorkerConfig."""

    def test_defaults(self):
        cfg = TargetWorkerConfig()
        assert cfg.grpc_port == 50052
        assert "llama" in cfg.model_name.lower() or cfg.model_name


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.draft_address == "localhost:50051"
        assert cfg.target_address == "localhost:50052"
        assert cfg.max_rounds == 20
        assert cfg.timeout_s == 30.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SPECSPLIT_ORCH_MAX_ROUNDS", "5")
        cfg = OrchestratorConfig()
        assert cfg.max_rounds == 5
