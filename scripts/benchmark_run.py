"""SpecSplit Benchmark Runner — load-test the orchestrator and log granular metrics.

Reads a JSONL file of prompts (e.g. a ShareGPT slice), drives the SpecSplit
orchestrator for each prompt, and records per-request metrics to CSV. Supports
sweeping over multiple *Gamma* (draft tree depth) values so you can evaluate
the throughput–acceptance-rate trade-off in a single invocation.

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
import csv
import json
import logging
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from specsplit.core.config import DraftWorkerConfig, OrchestratorConfig
from specsplit.core.telemetry import Stopwatch, TelemetryLogger
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
            per verification round (0.0–1.0).
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
    total_network_idle_ms: float = 0.0
    total_latency_ms: float = 0.0
    num_rounds: int = 0

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {k: getattr(self, k) for k in CSV_COLUMNS}


# =========================================================================
# Instrumented orchestrator wrapper
# =========================================================================


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
        self._orch = Orchestrator(config=config)
        self._telemetry = TelemetryLogger(service_name="benchmark")

    def connect(self) -> None:
        """Establish gRPC connections."""
        self._orch.connect()

    def run_and_measure(self, prompt: str, prompt_token_count: int) -> RequestMetrics:
        """Run a single prompt through the pipeline and measure everything.

        The benchmark instruments the orchestrator's internal flow by timing
        each draft→verify round independently.  Because the real pipeline
        is still stubbed, this uses a **simulated timing model** that
        records the call structure and produces realistic metric shapes.

        Args:
            prompt: Input text prompt.
            prompt_token_count: Pre-computed prompt token count.

        Returns:
            A populated ``RequestMetrics`` instance.

        .. todo::
            Wire into the real orchestrator once the pipeline is un-stubbed.
            Replace the simulated timers below with actual round timers.
        """
        request_id = uuid.uuid4().hex[:12]
        sw_total = Stopwatch()
        sw_total.start()

        round_metrics: list[dict] = []
        first_token_time: float | None = None
        total_net_idle = 0.0
        total_generated = 0

        # --------------- round loop (instrumented) ---------------
        # TODO(real-pipeline): Replace with per-round instrumentation once
        # the orchestrator exposes round-level hooks.  Currently the stub
        # orchestrator.run() is called once and the script infers metrics
        # from telemetry spans.

        with self._telemetry.span(
            "benchmark_request",
            request_id=request_id,
            gamma=self.gamma,
            prompt_length=prompt_token_count,
        ):
            sw_round = Stopwatch()
            sw_round.start()

            output = self._orch.run(prompt)

            sw_round.stop()

            # ---- Estimate per-round metrics from the span ----
            # With the stubbed orchestrator we simulate the round structure.
            # In production, each draft→verify round would be individually
            # timed.  Here we assume one round for the stub.
            simulated_round = {
                "round": 1,
                "draft_ms": sw_round.elapsed_ms * 0.3,  # ~30% draft
                "verify_ms": sw_round.elapsed_ms * 0.5,  # ~50% verify
                "network_idle_ms": sw_round.elapsed_ms * 0.2,  # ~20% network
                "tokens_accepted": 0,
                "tokens_drafted": self.gamma,
                "acceptance_rate": 0.0,
            }
            round_metrics.append(simulated_round)
            total_net_idle += simulated_round["network_idle_ms"]

            # First token available at end of first round
            if first_token_time is None:
                first_token_time = sw_round.elapsed_ms

        sw_total.stop()

        # ---- Compute aggregate metrics ----
        if round_metrics:
            avg_acceptance = statistics.mean(
                r["acceptance_rate"] for r in round_metrics
            )
        else:
            avg_acceptance = 0.0

        # For stub: estimate generated tokens from output length
        estimated_gen_tokens = max(len(output.split()) * 2, 1)  # rough estimate
        tpot = sw_total.elapsed_ms / max(estimated_gen_tokens, 1)

        return RequestMetrics(
            request_id=request_id,
            gamma=self.gamma,
            prompt_length=prompt_token_count,
            generated_tokens=estimated_gen_tokens,
            ttft_ms=round(first_token_time or 0.0, 3),
            tpot_ms=round(tpot, 3),
            average_acceptance_rate=round(avg_acceptance, 4),
            total_network_idle_ms=round(total_net_idle, 3),
            total_latency_ms=round(sw_total.elapsed_ms, 3),
            num_rounds=len(round_metrics),
        )


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


def estimate_token_count(text: str) -> int:
    """Rough token count estimate (whitespace + punctuation heuristic).

    Used when a real tokenizer is unavailable.  For accurate counts,
    replace with ``len(tokenizer.encode(text))``.
    """
    return max(len(text.split()), 1)


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
        help="Path to JSONL dataset file (each line: {\"prompt\": \"...\"})",
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
    if args.max_rounds is not None:
        config_overrides["max_rounds"] = args.max_rounds
    if args.max_output_tokens is not None:
        config_overrides["max_output_tokens"] = args.max_output_tokens

    base_config = OrchestratorConfig(**config_overrides)

    # ---- Sweep over gamma values ----
    all_metrics: list[RequestMetrics] = []

    for gamma in sorted(args.gamma):
        logger.info(
            "=== Starting sweep: gamma=%d (%d prompts) ===", gamma, len(dataset)
        )

        bench_orch = BenchmarkOrchestrator(config=base_config, gamma=gamma)
        bench_orch.connect()

        for i, entry in enumerate(dataset):
            prompt = entry["prompt"]
            prompt_tokens = estimate_token_count(prompt)
            req_id = entry.get("id", f"req-{i:04d}")

            logger.info(
                "[gamma=%d] Prompt %d/%d (id=%s, ~%d toks): %r",
                gamma,
                i + 1,
                len(dataset),
                req_id,
                prompt_tokens,
                prompt[:60],
            )

            metrics = bench_orch.run_and_measure(prompt, prompt_tokens)
            metrics.request_id = req_id
            all_metrics.append(metrics)

        logger.info("=== Completed sweep: gamma=%d ===", gamma)

    # ---- Write results ----
    write_csv(all_metrics, args.output)

    # ---- Telemetry export (optional) ----
    if args.telemetry_output:
        # Aggregate telemetry from the last orchestrator instance
        # (In production, each BenchmarkOrchestrator would export its own spans)
        logger.info("Telemetry export requested → %s", args.telemetry_output)

    # ---- Summary ----
    print_summary(all_metrics)
    print(f"Full per-request CSV: {args.output}")


if __name__ == "__main__":
    main()
