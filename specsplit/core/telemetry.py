"""High-precision timing and structured telemetry logging for SpecSplit.

Provides a ``Stopwatch`` for nanosecond-precision wall-clock measurement and
a ``TelemetryLogger`` for emitting structured JSON spans that can be collected
for distributed tracing and benchmarking.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Stopwatch — high-precision timing
# =============================================================================


@dataclass
class Stopwatch:
    """Nanosecond-precision wall-clock stopwatch with context-manager support.

    Usage::

        sw = Stopwatch()
        sw.start()
        # ... do work ...
        sw.stop()
        print(f"Elapsed: {sw.elapsed_ms:.3f} ms")

    Or as a context manager::

        with Stopwatch() as sw:
            # ... do work ...
        print(f"Elapsed: {sw.elapsed_ms:.3f} ms")
    """

    _start_ns: int = field(default=0, init=False, repr=False)
    _stop_ns: int = field(default=0, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def start(self) -> Stopwatch:
        """Begin timing."""
        self._start_ns = time.perf_counter_ns()
        self._running = True
        return self

    def stop(self) -> Stopwatch:
        """Stop timing and record elapsed nanoseconds."""
        self._stop_ns = time.perf_counter_ns()
        self._running = False
        return self

    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        end = time.perf_counter_ns() if self._running else self._stop_ns
        return end - self._start_ns

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000

    def __enter__(self) -> Stopwatch:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


# =============================================================================
# TelemetryLogger — structured JSON spans
# =============================================================================


@dataclass
class TelemetrySpan:
    """A single telemetry span representing a timed operation."""

    span_id: str
    operation: str
    wall_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_iso: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the span to a dictionary."""
        return {
            "span_id": self.span_id,
            "operation": self.operation,
            "wall_time_ms": round(self.wall_time_ms, 4),
            "metadata": self.metadata,
            "timestamp": self.timestamp_iso,
        }


class TelemetryLogger:
    """Structured telemetry logger that collects spans and exports them as JSON.

    Each span is tagged with a unique ``span_id`` for distributed tracing.
    Collected spans can be exported to a JSON file for offline analysis.

    Usage::

        tlog = TelemetryLogger(service_name="draft-worker")

        with tlog.span("generate_draft", tokens_processed=128) as span_id:
            # ... do work ...
            pass

        tlog.export("telemetry_output.json")
    """

    def __init__(self, service_name: str = "specsplit") -> None:
        self.service_name = service_name
        self._spans: list[TelemetrySpan] = []

    def span(self, operation: str, **metadata: Any) -> _SpanContext:
        """Create a timed span context manager.

        Args:
            operation: Name of the operation being timed.
            **metadata: Arbitrary key-value pairs to attach to the span.

        Returns:
            A context manager that records timing on exit.
        """
        return _SpanContext(logger=self, operation=operation, metadata=metadata)

    def record_span(self, span: TelemetrySpan) -> None:
        """Record a completed span."""
        self._spans.append(span)
        logger.debug(
            "Span [%s] %s: %.3f ms",
            span.span_id[:8],
            span.operation,
            span.wall_time_ms,
        )

    @property
    def spans(self) -> list[TelemetrySpan]:
        """Return a copy of all recorded spans."""
        return list(self._spans)

    def export(self, path: str | Path) -> None:
        """Export all recorded spans to a JSON file.

        Args:
            path: Output file path. Parent directories are created if needed.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "service": self.service_name,
            "num_spans": len(self._spans),
            "spans": [s.to_dict() for s in self._spans],
        }
        out.write_text(json.dumps(payload, indent=2))
        logger.info("Exported %d spans to %s", len(self._spans), out)

    def reset(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()


class _SpanContext:
    """Context manager for timing a telemetry span."""

    def __init__(
        self,
        logger: TelemetryLogger,
        operation: str,
        metadata: dict[str, Any],
    ) -> None:
        self._logger = logger
        self._operation = operation
        self._metadata = metadata
        self._sw = Stopwatch()
        self.span_id = uuid.uuid4().hex[:16]

    def __enter__(self) -> str:
        """Start timing and return the span ID."""
        self._sw.start()
        return self.span_id

    def __exit__(self, *_: Any) -> None:
        """Stop timing and record the span."""
        self._sw.stop()
        span = TelemetrySpan(
            span_id=self.span_id,
            operation=self._operation,
            wall_time_ms=self._sw.elapsed_ms,
            metadata=self._metadata,
        )
        self._logger.record_span(span)
