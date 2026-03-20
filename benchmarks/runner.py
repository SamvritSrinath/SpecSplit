#!/usr/bin/env python3
"""SpecSplit benchmark runner.

Runs the orchestrator client over a JSONL prompt dataset and produces:
  - benchmarks/results/raw/          telemetry JSON per request
  - benchmarks/results/summary.csv   per-request aggregate metrics
  - benchmarks/results/per_round.csv per-round acceptance/latency records
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
ORCHESTRATOR_MODULE = "specsplit.workers.orchestrator.client"

SUMMARY_COLUMNS = [
    "run_id",
    "prompt_id",
    "category",
    "gamma",
    "prompt_text",
    "prompt_tokens",
    "generated_tokens",
    "num_rounds",
    "ttft_ms",
    "tpot_ms",
    "total_latency_ms",
    "total_network_idle_ms",
    "network_overhead_pct",
    "avg_acceptance_rate",
    "speculation_hit_rate",
    "tokens_per_second",
    "effective_tokens_per_round",
    "miss_count",
    "hit_count",
]

ROUND_COLUMNS = [
    "run_id",
    "prompt_id",
    "gamma",
    "round_idx",
    "accepted",
    "path_depth",
    "tree_nodes",
    "acceptance_rate",
    "verify_rpc_ms",
    "draft_rpc_ms",
    "overlap_savings_ms",
    "speculation_outcome",
]


@dataclass
class RunSummary:
    run_id: str = ""
    prompt_id: str = ""
    category: str = ""
    gamma: int = 0
    prompt_text: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    num_rounds: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    total_latency_ms: float = 0.0
    total_network_idle_ms: float = 0.0
    network_overhead_pct: float = 0.0
    avg_acceptance_rate: float = 0.0
    speculation_hit_rate: float = 0.0
    tokens_per_second: float = 0.0
    effective_tokens_per_round: float = 0.0
    miss_count: int = 0
    hit_count: int = 0

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class RoundRecord:
    run_id: str = ""
    prompt_id: str = ""
    gamma: int = 0
    round_idx: int = 0
    accepted: int = 0
    path_depth: int = 0
    tree_nodes: int = 0
    acceptance_rate: float = 0.0
    verify_rpc_ms: float = 0.0
    draft_rpc_ms: float = 0.0
    overlap_savings_ms: float = 0.0
    speculation_outcome: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def parse_telemetry(telemetry_path: Path, prompt_id: str, category: str, gamma: int) -> tuple[RunSummary, list[RoundRecord]]:
    with open(telemetry_path) as f:
        data = json.load(f)

    summary_data = data.get("summary", {})
    prompt_data = data.get("prompt", {})
    spans = data.get("spans", [])
    timeline = data.get("timeline_events", [])
    per_round = data.get("per_round_acceptance", [])

    run_id = data.get("run_id", "unknown")
    total_latency_ms = summary_data.get("total_latency_ms", 0.0)
    network_idle_ms = summary_data.get("total_network_idle_ms", 0.0)
    generated_tokens = summary_data.get("generated_tokens", 0)

    initial_draft_ms = next((s["wall_time_ms"] for s in spans if s.get("operation") == "initial_draft"), 0.0)
    first_overlapped = next(
        (
            s["wall_time_ms"]
            for s in spans
            if s.get("operation") == "overlapped_round" and s.get("metadata", {}).get("round_idx") == 0
        ),
        0.0,
    )
    ttft_ms = initial_draft_ms + first_overlapped

    tpot_ms = total_latency_ms / max(generated_tokens, 1)
    tokens_per_second = (generated_tokens / total_latency_ms * 1000) if total_latency_ms > 0 else 0.0
    network_overhead_pct = (network_idle_ms / total_latency_ms * 100) if total_latency_ms > 0 else 0.0

    outcomes = [e["metadata"]["outcome"] for e in timeline if e.get("event_type") == "speculation_result"]
    hit_count = outcomes.count("hit")
    miss_count = outcomes.count("miss")
    num_rounds = summary_data.get("num_rounds", len(per_round))
    effective_tokens_per_round = generated_tokens / max(num_rounds, 1)

    prompt_tokens = data.get("effective_config", {}).get("max_context_window", 0)
    for ev in timeline:
        if ev.get("event_type") == "prompt_tokenized":
            prompt_tokens = ev["metadata"].get("prompt_token_count", prompt_tokens)
            break

    summary = RunSummary(
        run_id=run_id,
        prompt_id=prompt_id,
        category=category,
        gamma=gamma,
        prompt_text=prompt_data.get("text", "")[:120],
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        num_rounds=num_rounds,
        ttft_ms=round(ttft_ms, 3),
        tpot_ms=round(tpot_ms, 3),
        total_latency_ms=round(total_latency_ms, 3),
        total_network_idle_ms=round(network_idle_ms, 3),
        network_overhead_pct=round(network_overhead_pct, 2),
        avg_acceptance_rate=round(summary_data.get("average_acceptance_rate", 0.0), 4),
        speculation_hit_rate=round(summary_data.get("speculation_hit_rate", 0.0), 4),
        tokens_per_second=round(tokens_per_second, 3),
        effective_tokens_per_round=round(effective_tokens_per_round, 3),
        miss_count=miss_count,
        hit_count=hit_count,
    )

    span_by_round: dict[int, dict] = {}
    for s in spans:
        if s.get("operation") == "overlapped_round":
            ridx = s.get("metadata", {}).get("round_idx")
            if ridx is not None:
                span_by_round[ridx] = s

    outcome_by_round: dict[int, str] = {}
    for ev in timeline:
        if ev.get("event_type") == "speculation_result":
            ridx = ev["metadata"].get("round_idx")
            if ridx is not None:
                outcome_by_round[ridx] = ev["metadata"].get("outcome", "unknown")

    round_records: list[RoundRecord] = []
    for r in per_round:
        ridx = r["round"]
        span_meta = span_by_round.get(ridx, {}).get("metadata", {})
        round_records.append(
            RoundRecord(
                run_id=run_id,
                prompt_id=prompt_id,
                gamma=gamma,
                round_idx=ridx,
                accepted=r.get("accepted", 0),
                path_depth=r.get("path_depth", 0),
                tree_nodes=r.get("tree_nodes", 0),
                acceptance_rate=round(r.get("acceptance_rate", 0.0), 4),
                verify_rpc_ms=round(span_meta.get("verify_rpc_ms", 0.0), 3),
                draft_rpc_ms=round(span_meta.get("draft_rpc_ms", 0.0), 3),
                overlap_savings_ms=round(span_meta.get("overlap_savings_ms", 0.0), 3),
                speculation_outcome=outcome_by_round.get(ridx, "unknown"),
            )
        )

    return summary, round_records


def build_command(
    prompt: str,
    gamma: int,
    max_rounds: int,
    max_output_tokens: int,
    draft_temperature: float,
    verify_temperature: float,
    use_target_cache: bool,
    telemetry_output_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        ORCHESTRATOR_MODULE,
        "--prompt",
        prompt,
        "--max-rounds",
        str(max_rounds),
        "--max-output-tokens",
        str(max_output_tokens),
        "--max-draft-tokens",
        str(gamma),
        "--draft-temperature",
        str(draft_temperature),
        "--verify-temperature",
        str(verify_temperature),
        "--telemetry-output",
        str(telemetry_output_path),
    ]
    if use_target_cache:
        cmd.append("--use-target-cache")
    return cmd


def run_single(
    prompt_entry: dict,
    draft_addr: str,
    target_addr: str,
    tokenizer_model: str,
    gamma: int,
    max_rounds: int,
    max_output_tokens: int,
    draft_temperature: float,
    verify_temperature: float,
    use_target_cache: bool,
    raw_dir: Path,
    timeout_s: int = 300,
) -> Path | None:
    prompt = prompt_entry["prompt"]
    prompt_id = prompt_entry.get("id", "unknown")
    safe_prompt_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in prompt_id)
    telemetry_path = raw_dir / f"gamma{gamma}_{safe_prompt_id}_{int(time.time() * 1000)}.json"

    env = os.environ.copy()
    env["SPECSPLIT_ORCH_DRAFT_ADDRESS"] = draft_addr
    env["SPECSPLIT_ORCH_TARGET_ADDRESS"] = target_addr
    env["SPECSPLIT_ORCH_TOKENIZER_MODEL"] = tokenizer_model

    cmd = build_command(
        prompt=prompt,
        gamma=gamma,
        max_rounds=max_rounds,
        max_output_tokens=max_output_tokens,
        draft_temperature=draft_temperature,
        verify_temperature=verify_temperature,
        use_target_cache=use_target_cache,
        telemetry_output_path=telemetry_path,
    )

    logger.info("[gamma=%d] Running prompt_id=%s: %r", gamma, prompt_id, prompt[:60])
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        logger.error("[gamma=%d] TIMEOUT after %ds for prompt_id=%s", gamma, timeout_s, prompt_id)
        return None
    except Exception as exc:
        logger.error("[gamma=%d] Subprocess error for prompt_id=%s: %s", gamma, prompt_id, exc)
        return None

    if result.returncode != 0:
        logger.error(
            "[gamma=%d] CLI returned exit code %d for prompt_id=%s:\n%s",
            gamma,
            result.returncode,
            prompt_id,
            result.stderr[-1000:],
        )
        return None

    logger.info("[gamma=%d] prompt_id=%s completed in %.1fs", gamma, prompt_id, time.monotonic() - t0)
    if not telemetry_path.exists():
        logger.error("Telemetry output missing for prompt_id=%s at %s", prompt_id, telemetry_path)
        return None
    return telemetry_path


def load_prompts(path: Path, categories: list[str] | None = None) -> list[dict]:
    entries = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: %s", i, exc)
                continue
            if categories and obj.get("category") not in categories:
                continue
            entries.append(obj)
    logger.info("Loaded %d prompts from %s", len(entries), path)
    return entries


def sample_per_category(entries: list[dict], per_category_limit: int) -> list[dict]:
    """Keep at most N prompts per category while preserving original order."""
    if per_category_limit <= 0:
        return entries
    counts: dict[str, int] = {}
    sampled: list[dict] = []
    for entry in entries:
        category = entry.get("category", "unknown")
        cur = counts.get(category, 0)
        if cur >= per_category_limit:
            continue
        counts[category] = cur + 1
        sampled.append(entry)
    return sampled


def already_ran(summary_csv: Path, prompt_id: str, gamma: int) -> bool:
    if not summary_csv.exists():
        return False
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            if row.get("prompt_id") == prompt_id and int(row.get("gamma", -1)) == gamma:
                return True
    return False


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpecSplit benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/specsplit_bench.jsonl"))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=0,
        help="If >0, keep up to N prompts per category (applied before --limit).",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gamma", type=int, nargs="+", default=[5])
    parser.add_argument("--draft-addr", type=str, default=os.environ.get("SPECSPLIT_ORCH_DRAFT_ADDRESS", ""))
    parser.add_argument("--target-addr", type=str, default=os.environ.get("SPECSPLIT_ORCH_TARGET_ADDRESS", ""))
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=os.environ.get("SPECSPLIT_ORCH_TOKENIZER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--draft-temperature", type=float, default=0.0)
    parser.add_argument("--verify-temperature", type=float, default=0.0)
    parser.add_argument("--no-target-cache", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if not args.draft_addr or not args.target_addr:
        logger.error(
            "Draft and target addresses must be set via --draft-addr/--target-addr or env vars "
            "SPECSPLIT_ORCH_DRAFT_ADDRESS/SPECSPLIT_ORCH_TARGET_ADDRESS."
        )
        sys.exit(1)

    results_dir: Path = args.results_dir
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "summary.csv"
    rounds_csv = results_dir / "per_round.csv"

    prompts = load_prompts(args.prompts, categories=args.categories)
    if args.per_category_limit > 0:
        prompts = sample_per_category(prompts, args.per_category_limit)
        logger.info("Using per-category sample: up to %d prompts/category -> %d prompts", args.per_category_limit, len(prompts))
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        logger.error("No prompts loaded — check --prompts and --categories.")
        sys.exit(1)

    gammas = sorted(args.gamma)
    use_target_cache = not args.no_target_cache
    logger.info("Starting benchmark: %d prompts x %d gamma(s)", len(prompts), len(gammas))

    run_count = 0
    skip_count = 0
    error_count = 0

    for gamma in gammas:
        logger.info("=== gamma=%d sweep ===", gamma)
        for entry in prompts:
            prompt_id = entry.get("id", "unknown")
            category = entry.get("category", "unknown")

            if args.resume and already_ran(summary_csv, prompt_id, gamma):
                skip_count += 1
                logger.info("Skipping gamma=%d prompt_id=%s (already in results)", gamma, prompt_id)
                continue

            telemetry_path = run_single(
                prompt_entry=entry,
                draft_addr=args.draft_addr,
                target_addr=args.target_addr,
                tokenizer_model=args.tokenizer,
                gamma=gamma,
                max_rounds=args.max_rounds,
                max_output_tokens=args.max_output_tokens,
                draft_temperature=args.draft_temperature,
                verify_temperature=args.verify_temperature,
                use_target_cache=use_target_cache,
                raw_dir=raw_dir,
                timeout_s=args.timeout,
            )
            if telemetry_path is None:
                error_count += 1
            else:
                try:
                    summary, round_records = parse_telemetry(telemetry_path, prompt_id, category, gamma)
                    append_csv(summary_csv, [summary.to_row()], SUMMARY_COLUMNS)
                    append_csv(rounds_csv, [r.to_row() for r in round_records], ROUND_COLUMNS)
                    run_count += 1
                except Exception as exc:
                    logger.error("Failed to parse telemetry %s: %s", telemetry_path, exc)
                    error_count += 1

            if args.delay > 0:
                time.sleep(args.delay)

    logger.info("Done. %d completed, %d skipped, %d errors.", run_count, skip_count, error_count)
    logger.info("summary CSV: %s", summary_csv)
    logger.info("per-round CSV: %s", rounds_csv)
    logger.info("raw telemetry: %s", raw_dir)


if __name__ == "__main__":
    main()
