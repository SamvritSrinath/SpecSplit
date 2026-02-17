"""SpecSplit Benchmark Runner — run prompts through the pipeline and collect telemetry.

Usage::

    python scripts/benchmark_run.py \\
        --dataset prompts.jsonl \\
        --output results.json \\
        --max-rounds 10

The dataset file should be a JSONL file with one JSON object per line,
each containing at least a ``"prompt"`` field.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from specsplit.core.config import OrchestratorConfig
from specsplit.core.telemetry import TelemetryLogger
from specsplit.workers.orchestrator.client import Orchestrator

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> list[dict]:
    """Load prompts from a JSONL file.

    Args:
        path: Path to a JSONL file where each line has ``{"prompt": "..."}``.

    Returns:
        List of parsed JSON objects.
    """
    entries = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d: %s", line_num, e)
    logger.info("Loaded %d prompts from %s", len(entries), path)
    return entries


def run_benchmark(
    dataset: list[dict],
    config: OrchestratorConfig,
) -> dict:
    """Run all prompts through the orchestrator and collect results.

    Args:
        dataset: List of dicts, each with a ``"prompt"`` key.
        config: Orchestrator configuration.

    Returns:
        A dict containing results and aggregate statistics.
    """
    orch = Orchestrator(config=config)
    orch.connect()

    results = []
    total_start = time.perf_counter()

    for i, entry in enumerate(dataset):
        prompt = entry.get("prompt", "")
        logger.info("Prompt %d/%d: %r", i + 1, len(dataset), prompt[:60])

        start = time.perf_counter()
        output = orch.run(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append({
            "prompt": prompt,
            "output": output,
            "elapsed_ms": round(elapsed_ms, 3),
        })

    total_elapsed_ms = (time.perf_counter() - total_start) * 1000

    return {
        "num_prompts": len(dataset),
        "total_elapsed_ms": round(total_elapsed_ms, 3),
        "avg_elapsed_ms": round(total_elapsed_ms / max(len(dataset), 1), 3),
        "results": results,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SpecSplit Benchmark Runner",
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
        default="benchmark_results.json",
        help="Output path for benchmark results JSON",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override max draft→verify rounds",
    )
    parser.add_argument(
        "--telemetry-output",
        type=str,
        default=None,
        help="Path to export raw telemetry spans",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )

    # Load config
    config = OrchestratorConfig()
    if args.max_rounds is not None:
        config = OrchestratorConfig(max_rounds=args.max_rounds)

    # Load & run
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("No prompts found in %s", args.dataset)
        sys.exit(1)

    results = run_benchmark(dataset, config)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", out_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"Benchmark Complete: {results['num_prompts']} prompts")
    print(f"Total time: {results['total_elapsed_ms']:.1f} ms")
    print(f"Avg per prompt: {results['avg_elapsed_ms']:.1f} ms")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
