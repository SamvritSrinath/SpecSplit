"""SpecSplit Benchmark Runner — load-test the orchestrator and log granular metrics.

Reads a JSONL file of prompts (e.g. a ShareGPT slice), drives the SpecSplit
orchestrator for each prompt, and records per-request metrics to CSV. Supports
sweeping over multiple *Gamma* (draft tree depth) values so you can evaluate
the throughput-acceptance-rate trade-off in a single invocation.

Usage::

    # Single gamma
    python scripts/benchmark_run.py \\
        --dataset prompts.jsonl --output results.csv --gamma 5

    # Sweep over multiple gammas
    python scripts/benchmark_run.py \\
        --dataset prompts.jsonl --output results.csv --gamma 1 3 5 8 12

    # Use environment variables for addresses
    SPECSPLIT_ORCH_DRAFT_ADDRESS=gpu1:50051 \\
    SPECSPLIT_ORCH_TARGET_ADDRESS=gpu2:50052 \\
        python scripts/benchmark_run.py --dataset prompts.jsonl

Dataset format — one JSON object per line, at minimum a ``"prompt"`` field::

    {"prompt": "Explain quantum computing in simple terms."}
    {"prompt": "Write a haiku about distributed systems.", "id": "req-42"}
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from specsplit.core.config import OrchestratorConfig, load_config_file
from specsplit.core.telemetry import TelemetryLogger
from specsplit.workers.orchestrator.client import Orchestrator

logger = logging.getLogger(__name__)

# =========================================================================
# Metric data model
# =========================================================================

CSV_COLUMNS = [
    "request_id",
    "gamma",
    "prompt_length",
    "generated_tokens",
    "ttft_ms",
    "tpot_ms",
    "average_acceptance_rate",
    "per_round_acceptance",
    "total_network_idle_ms",
    "total_latency_ms",
    "num_rounds",
]


@dataclass
class RequestMetrics:
    """Per-request metrics collected during a benchmark run.

    Attributes:
        request_id:  Unique identifier for this request.
        gamma:  Draft tree depth (K) used for this request.
        prompt_length:  Number of tokens in the prompt.
        generated_tokens:  Number of output tokens produced.
        ttft_ms:  Time-to-First-Token (ms) — latency from request start
            to the first accepted/correction token being available.
        tpot_ms:  Time-Per-Output-Token (ms) — average inter-token latency
            across all generated tokens (total / generated_tokens).
        average_acceptance_rate:  Mean fraction of draft tokens accepted
            per verification round (0.0-1.0).
        total_network_idle_ms:  Cumulative time (ms) the orchestrator
            spent waiting on gRPC round-trips (network latency overhead).
        total_latency_ms:  Wall clock time (ms) for the entire request.
        num_rounds:  Number of draft→verify rounds.
    """

    request_id: str = ""
    gamma: int = 0
    prompt_length: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    average_acceptance_rate: float = 0.0
    per_round_acceptance: str = ""
    total_network_idle_ms: float = 0.0
    total_latency_ms: float = 0.0
    num_rounds: int = 0

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {k: getattr(self, k) for k in CSV_COLUMNS}


# =========================================================================
# Instrumented orchestrator wrapper
# =========================================================================


def _ttft_from_telemetry(telemetry: list[dict]) -> float:
    """Compute Time-to-First-Token from pipeline telemetry spans.

    TTFT is the time from request start until the first batch of tokens
    is available, which spans the initial draft generation plus the first
    overlapped verify/draft round.

    Args:
        telemetry: List of span dicts from ``PipelineResult.telemetry``.

    Returns:
        TTFT in milliseconds, or 0.0 if the required spans are absent.
    """
    initial_draft_ms = next(
        (s["wall_time_ms"] for s in telemetry if s["operation"] == "initial_draft"),
        0.0,
    )
    first_round_ms = next(
        (
            s["wall_time_ms"]
            for s in telemetry
            if s["operation"] == "overlapped_round"
            and s["metadata"].get("round_idx") == 0
        ),
        0.0,
    )
    return initial_draft_ms + first_round_ms


class BenchmarkOrchestrator:
    """Wraps the real ``Orchestrator`` and instruments every round.

    For each prompt it records TTFT, per-round timing, acceptance rate,
    and network idle time.  This class exists as a *measurement harness*
    that sits above the orchestrator — it never modifies the generation
    logic itself.

    Args:
        config: Orchestrator configuration.
        gamma: Draft tree depth for this run.
    """

    def __init__(self, config: OrchestratorConfig, gamma: int) -> None:
        self.config = config
        self.gamma = gamma
        # Override max_draft_tokens so the pipeline actually uses this gamma
        gamma_config = config.model_copy(update={"max_draft_tokens": gamma})
        self._orch = Orchestrator(config=gamma_config)
        self._telemetry = TelemetryLogger(service_name="benchmark")

    def connect(self) -> None:
        """Establish gRPC connections."""
        self._orch.connect()

    def run_and_measure(self, prompt: str) -> RequestMetrics:
        """Run a single prompt through the pipeline and measure everything.

        Calls the real orchestrator pipeline and extracts per-request metrics
        directly from the returned ``PipelineResult`` and its telemetry spans.

        Args:
            prompt: Input text prompt.

        Returns:
            A tuple of populated ``RequestMetrics`` instance and the raw telemetry span list.
        """
        request_id = uuid.uuid4().hex[:12]

        with self._telemetry.span(
            "benchmark_request",
            request_id=request_id,
            gamma=self.gamma,
        ):
            _, result = self._orch.run_with_result_sync(prompt)

        # Use the real tokenizer for prompt length if available
        tokenizer = self._orch._ensure_tokenizer()
        real_prompt_tokens = len(tokenizer.encode(prompt))

        generated_tokens = len(result.output_tokens)
        ttft_ms = _ttft_from_telemetry(result.telemetry)
        tpot_ms = result.wall_time_ms / max(generated_tokens, 1)

        per_round_acc_str = ";".join(
            f"{r['round']}:{r['acceptance_rate']:.2f}" 
            for r in result.per_round_acceptance
        )

        metrics = RequestMetrics(
            request_id=request_id,
            gamma=self.gamma,
            prompt_length=real_prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=round(ttft_ms, 3),
            tpot_ms=round(tpot_ms, 3),
            average_acceptance_rate=round(result.acceptance_rate, 4),
            per_round_acceptance=per_round_acc_str,
            total_network_idle_ms=round(result.network_idle_ms, 3),
            total_latency_ms=round(result.wall_time_ms, 3),
            num_rounds=result.total_rounds,
        )
        return metrics, result.telemetry


# =========================================================================
# Dataset loading
# =========================================================================


def load_dataset(path: str) -> list[dict]:
    """Load prompts from a JSONL file.

    Each line must be a JSON object with at least a ``"prompt"`` field.
    Lines with a ``"conversations"`` field (ShareGPT format) extract
    the first human turn as the prompt.

    Args:
        path: Path to the JSONL dataset.

    Returns:
        List of dicts with ``"prompt"`` and optional ``"id"`` keys.
    """
    entries: list[dict] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d: %s", line_num, e)
                continue

            # ShareGPT format support
            if "conversations" in obj and "prompt" not in obj:
                for turn in obj["conversations"]:
                    if turn.get("from") in ("human", "user"):
                        obj["prompt"] = turn["value"]
                        break

            if "prompt" not in obj:
                logger.warning("Skipping line %d: no 'prompt' field", line_num)
                continue

            entries.append(obj)

    logger.info("Loaded %d prompts from %s", len(entries), path)
    return entries


# =========================================================================
# CSV writer
# =========================================================================


def write_csv(rows: list[RequestMetrics], path: str) -> None:
    """Write per-request metrics to a CSV file.

    Args:
        rows: List of ``RequestMetrics`` to write.
        path: Output file path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_row())

    logger.info("Metrics written to %s (%d rows)", out_path, len(rows))


# =========================================================================
# Summary statistics
# =========================================================================


def print_summary(all_metrics: list[RequestMetrics]) -> None:
    """Print a human-readable summary table grouped by gamma."""
    gammas = sorted(set(m.gamma for m in all_metrics))

    print(f"\n{'=' * 90}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'Gamma':>6}  {'Reqs':>5}  {'Avg TTFT':>10}  {'Avg TPOT':>10}  "
        f"{'Avg Accept':>11}  {'Avg NetIdle':>12}  {'Avg Latency':>12}"
    )
    print(f"{'-' * 90}")

    for g in gammas:
        subset = [m for m in all_metrics if m.gamma == g]
        n = len(subset)
        avg_ttft = statistics.mean(m.ttft_ms for m in subset)
        avg_tpot = statistics.mean(m.tpot_ms for m in subset)
        avg_acc = statistics.mean(m.average_acceptance_rate for m in subset)
        avg_idle = statistics.mean(m.total_network_idle_ms for m in subset)
        avg_lat = statistics.mean(m.total_latency_ms for m in subset)

        print(
            f"{g:>6}  {n:>5}  {avg_ttft:>8.2f}ms  {avg_tpot:>8.2f}ms  "
            f"{avg_acc:>10.2%}  {avg_idle:>10.2f}ms  {avg_lat:>10.2f}ms"
        )

    print(f"{'=' * 90}\n")


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    """CLI entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="SpecSplit Benchmark Runner — load-test with per-request metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help='Path to JSONL dataset file (each line: {"prompt": "..."})',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV path for per-request metrics",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        nargs="+",
        default=[5],
        help=(
            "Draft tree depth(s) to sweep. Corresponds to "
            "DraftWorkerConfig.max_draft_tokens.  "
            "Example: --gamma 1 3 5 8 12"
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override max draft→verify rounds per prompt",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override max output tokens per prompt",
    )
    parser.add_argument(
        "--telemetry-output",
        type=str,
        default=None,
        help="Path to export raw telemetry spans (JSON)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file (same format as client.py --config)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    )

    # ---- Load dataset ----
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("No prompts found in %s", args.dataset)
        sys.exit(1)

    # ---- Build config overrides ----
    config_overrides: dict = {}

    # Load config file if provided (Issue 13)
    if args.config:
        file_cfg = load_config_file(args.config)
        config_overrides.update(file_cfg.get("orchestrator", {}))

    # CLI args override config file values
    if args.max_rounds is not None:
        config_overrides["max_rounds"] = args.max_rounds
    if args.max_output_tokens is not None:
        config_overrides["max_output_tokens"] = args.max_output_tokens

    base_config = OrchestratorConfig(**config_overrides)

    # ---- Sweep over gamma values ----
    all_metrics: list[RequestMetrics] = []
    all_telemetry: list[dict] = []

    for gamma in sorted(args.gamma):
        logger.info("=== Starting sweep: gamma=%d (%d prompts) ===", gamma, len(dataset))

        bench_orch = BenchmarkOrchestrator(config=base_config, gamma=gamma)
        bench_orch.connect()

        for i, entry in enumerate(dataset):
            prompt = entry["prompt"]
            req_id = entry.get("id", f"req-{i:04d}")

            logger.info(
                "[gamma=%d] Prompt %d/%d (id=%s): %r",
                gamma,
                i + 1,
                len(dataset),
                req_id,
                prompt[:60],
            )

            metrics, req_telemetry = bench_orch.run_and_measure(prompt)
            metrics.request_id = req_id
            all_metrics.append(metrics)
            all_telemetry.extend(req_telemetry)
            
            # Issue 31: Prevent infinite telemetry accumulation in memory
            bench_orch._orch._telemetry.reset()

        logger.info("=== Completed sweep: gamma=%d ===", gamma)

        # Issue 41: Close gRPC channels to prevent resource leaks
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(bench_orch._orch.close())

    # ---- Write results ----
    write_csv(all_metrics, args.output)

    # ---- Telemetry export (optional) ----
    if args.telemetry_output:
        # Export the per-round accumulated pipeline spans (Issue 24)
        with open(args.telemetry_output, "w") as f:
            json.dump(all_telemetry, f, indent=2)
        logger.info("Telemetry exported → %s", args.telemetry_output)
        
    # ---- Summary ----
    print_summary(all_metrics)
    print(f"Full per-request CSV: {args.output}")


if __name__ == "__main__":
    main()
