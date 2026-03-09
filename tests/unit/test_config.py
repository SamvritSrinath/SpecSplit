"""Unit tests for specsplit.core.config."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from specsplit.core.config import DraftWorkerConfig, OrchestratorConfig, TargetWorkerConfig
from specsplit.workers.orchestrator.client import Orchestrator, _resolve_tokenizer_model


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

    def test_resolve_tokenizer_model_prefers_worker_models(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SPECSPLIT_DRAFT_MODEL_NAME", "/models/qwen-draft")
        monkeypatch.setenv("SPECSPLIT_TARGET_MODEL_NAME", "/models/qwen-target")

        assert _resolve_tokenizer_model("gpt2") == "/models/qwen-target"

    def test_resolve_tokenizer_model_prefers_probed_workers_over_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SPECSPLIT_DRAFT_MODEL_NAME", "/models/env-draft")
        monkeypatch.setenv("SPECSPLIT_TARGET_MODEL_NAME", "/models/env-target")

        assert _resolve_tokenizer_model(
            "gpt2",
            draft_model_name="/models/ping-draft",
            target_model_name="/models/ping-target",
        ) == "/models/ping-target"

    def test_ensure_tokenizer_uses_ping_vocab_sizes_without_loading_worker_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        class _FakeTokenizer:
            pad_token = None
            eos_token = "</s>"

        def _fake_from_pretrained(model_name: str):
            calls.append(model_name)
            return _FakeTokenizer()

        import transformers

        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            staticmethod(_fake_from_pretrained),
        )

        orch = Orchestrator(
            config=OrchestratorConfig(strict_vocab_check=True),
            model_name="meta-llama/Llama-3.1-8B",
        )
        orch._draft_worker_model_name = "/bad/draft/path"
        orch._target_worker_model_name = "/bad/target/path"
        orch._draft_worker_vocab_size = 128256
        orch._target_worker_vocab_size = 128256

        orch._ensure_tokenizer()

        assert calls == ["meta-llama/Llama-3.1-8B"]

    def test_ensure_tokenizer_warns_on_ping_vocab_mismatch_instead_of_crashing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class _FakeTokenizer:
            pad_token = None
            eos_token = "</s>"

        import transformers

        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda _: _FakeTokenizer()),
        )

        orch = Orchestrator(
            config=OrchestratorConfig(strict_vocab_check=True),
            model_name="meta-llama/Llama-3.1-8B",
        )
        orch._draft_worker_model_name = "/bad/draft/path"
        orch._target_worker_model_name = "/bad/target/path"
        orch._draft_worker_vocab_size = 32000
        orch._target_worker_vocab_size = 64000

        with caplog.at_level(logging.WARNING):
            orch._ensure_tokenizer()

        assert "different vocab sizes" in caplog.text
