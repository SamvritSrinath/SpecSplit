"""Unit tests for orchestrator-side run logging and telemetry export."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from specsplit.core.config import OrchestratorConfig
from specsplit.workers.orchestrator.client import (
    _build_run_report,
    _default_telemetry_output_path,
)
from specsplit.workers.orchestrator.pipeline import PipelineResult


def test_build_run_report_captures_summary_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run reports should include benchmark-style metrics and env provenance."""
    monkeypatch.setenv("SPECSPLIT_ORCH_MAX_DRAFT_TOKENS", "7")
    monkeypatch.setenv("SPECSPLIT_ORCH_DRAFT_TEMPERATURE", "0.30")
    monkeypatch.setenv("SPECSPLIT_DRAFT_NUM_BEAMS", "4")
    monkeypatch.setenv("SPECSPLIT_TARGET_MODEL_NAME", "/models/target")

    config = OrchestratorConfig(
        max_rounds=2,
        max_output_tokens=32,
        max_draft_tokens=7,
        draft_temperature=0.30,
        verify_temperature=0.10,
    )
    result = PipelineResult(
        output_tokens=[10, 11, 12],
        total_rounds=2,
        acceptance_rate=0.75,
        speculation_hit_rate=0.5,
        wall_time_ms=24.0,
        network_idle_ms=9.5,
        per_round_acceptance=[
            {
                "round": 0,
                "accepted": 2,
                "path_depth": 3,
                "tree_nodes": 3,
                "acceptance_rate": 0.6667,
            },
            {
                "round": 1,
                "accepted": 1,
                "path_depth": 1,
                "tree_nodes": 1,
                "acceptance_rate": 1.0,
            },
        ],
        telemetry=[
            {
                "operation": "initial_draft",
                "wall_time_ms": 4.0,
                "metadata": {"round_idx": 0},
            },
            {
                "operation": "overlapped_round",
                "wall_time_ms": 6.0,
                "metadata": {"round_idx": 0},
            },
            {
                "operation": "overlapped_round",
                "wall_time_ms": 8.0,
                "metadata": {"round_idx": 1},
            },
        ],
        timeline_events=[
            {"event_type": "run_started", "metadata": {}},
            {"event_type": "run_completed", "metadata": {}},
        ],
    )

    report = _build_run_report(
        prompt="hello world",
        output_text="generated output",
        result=result,
        config=config,
        model_name="gpt2",
        run_id="run-123",
        started_at_iso="2026-03-08T10:00:00.000000+0000",
        ended_at_iso="2026-03-08T10:00:00.024000+0000",
        output_path="telemetry/orchestrator-run-20260308T100000.000000+0000.json",
        telemetry_spans=[
            {
                "operation": "full_pipeline",
                "wall_time_ms": 24.0,
                "metadata": {"prompt_len": 11},
            },
        ],
        telemetry_events=result.timeline_events,
        worker_metadata={
            "draft": {"model_name": "qwen-draft", "vocab_size": 32000},
            "target": {"model_name": "qwen-target", "vocab_size": 64000},
        },
    )

    assert report["run_id"] == "run-123"
    assert report["summary"]["ttft_ms"] == 10.0
    assert report["summary"]["tpot_ms"] == 8.0
    assert report["summary"]["generated_tokens"] == 3
    assert report["summary"]["total_network_idle_ms"] == 9.5
    assert report["effective_config"]["max_draft_tokens"] == 7
    assert report["environment"]["SPECSPLIT_DRAFT_NUM_BEAMS"] == "4"
    assert report["environment"]["SPECSPLIT_ORCH_MAX_DRAFT_TOKENS"] == "7"
    assert report["worker_metadata"]["target"]["model_name"] == "qwen-target"


def test_default_telemetry_output_path_is_timestamped(tmp_path: Path) -> None:
    """Ordinary runs should default to a timestamped telemetry artifact path."""
    started_at = datetime(2026, 3, 8, 10, 11, 12, 123456, tzinfo=timezone.utc)

    path = _default_telemetry_output_path(started_at=started_at, root=tmp_path)

    assert path.parent == tmp_path
    assert path.suffix == ".json"
    assert re.fullmatch(
        r"orchestrator-run-\d{8}T\d{6}\.\d{6}[+-]\d{4}\.json",
        path.name,
    )
