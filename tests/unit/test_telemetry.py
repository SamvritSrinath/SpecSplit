"""Unit tests for specsplit.core.telemetry."""

from __future__ import annotations

import json
import time
from pathlib import Path

from specsplit.core.telemetry import Stopwatch, TelemetryLogger


class TestStopwatch:
    """Tests for the Stopwatch timer."""

    def test_basic_timing(self):
        """Elapsed time should be positive after start/stop."""
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)  # 10 ms
        sw.stop()
        assert sw.elapsed_ms > 0
        assert sw.elapsed_ns > 0

    def test_context_manager(self):
        """Context manager should auto-start and auto-stop."""
        with Stopwatch() as sw:
            time.sleep(0.01)
        assert sw.elapsed_ms >= 5  # Allow some slack

    def test_elapsed_seconds(self):
        """elapsed_s should be consistent with elapsed_ms."""
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        sw.stop()
        assert abs(sw.elapsed_s * 1000 - sw.elapsed_ms) < 1  # < 1ms difference


class TestTelemetryLogger:
    """Tests for structured telemetry logging."""

    def test_span_recording(self):
        """Spans should be recorded with correct metadata."""
        tlog = TelemetryLogger(service_name="test-service")

        with tlog.span("test_op", key="value") as span_id:
            time.sleep(0.005)
            assert isinstance(span_id, str)
            assert len(span_id) == 16  # hex UUID prefix

        assert len(tlog.spans) == 1
        span = tlog.spans[0]
        assert span.operation == "test_op"
        assert span.metadata["key"] == "value"
        assert span.wall_time_ms > 0

    def test_multiple_spans(self):
        """Multiple spans should accumulate."""
        tlog = TelemetryLogger()

        for i in range(3):
            with tlog.span(f"op_{i}"):
                pass

        assert len(tlog.spans) == 3

    def test_export_json(self, tmp_path: Path):
        """Export should produce valid JSON with correct structure."""
        tlog = TelemetryLogger(service_name="export-test")

        with tlog.span("sample_op", tokens=42):
            pass

        out_file = tmp_path / "telemetry.json"
        tlog.export(out_file)

        data = json.loads(out_file.read_text())
        assert data["service"] == "export-test"
        assert data["num_spans"] == 1
        assert data["spans"][0]["operation"] == "sample_op"
        assert data["spans"][0]["metadata"]["tokens"] == 42

    def test_reset(self):
        """Reset should clear all recorded spans."""
        tlog = TelemetryLogger()
        with tlog.span("op"):
            pass
        assert len(tlog.spans) == 1
        tlog.reset()
        assert len(tlog.spans) == 0
