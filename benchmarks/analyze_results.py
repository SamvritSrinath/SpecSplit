#!/usr/bin/env python3
"""SpecSplit benchmark result analyzer.

Reads benchmark CSVs produced by `benchmarks/runner.py` and generates:
  - Console summary tables (acceptance rate, TPOT, hit rate by gamma/category)
  - Matplotlib figures saved under `<results-dir>/figures/`
  - Optional LaTeX macros for reports

Usage:
    python3 benchmarks/analyze_results.py
    python3 benchmarks/analyze_results.py --results-dir benchmarks/results
    python3 benchmarks/analyze_results.py --results-dir benchmarks/results --no-plots
    python3 benchmarks/analyze_results.py --latex
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print("[WARN] matplotlib not installed - plots will be skipped.")


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(results_dir: Path) -> list[dict]:
    path = results_dir / "summary.csv"
    if not path.exists():
        print(f"ERROR: summary.csv not found in {results_dir}")
        sys.exit(1)
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for col in ["gamma", "prompt_tokens", "generated_tokens", "num_rounds", "miss_count", "hit_count"]:
                row[col] = int(row.get(col, 0) or 0)
            for col in [
                "ttft_ms",
                "tpot_ms",
                "total_latency_ms",
                "total_network_idle_ms",
                "network_overhead_pct",
                "avg_acceptance_rate",
                "speculation_hit_rate",
                "tokens_per_second",
                "effective_tokens_per_round",
            ]:
                row[col] = float(row.get(col, 0.0) or 0.0)
            rows.append(row)
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def load_rounds(results_dir: Path) -> list[dict]:
    path = results_dir / "per_round.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for col in ["gamma", "round_idx", "accepted", "path_depth", "tree_nodes"]:
                row[col] = int(row.get(col, 0) or 0)
            for col in ["acceptance_rate", "verify_rpc_ms", "draft_rpc_ms", "overlap_savings_ms"]:
                row[col] = float(row.get(col, 0.0) or 0.0)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------

def print_gamma_table(rows: list[dict]) -> None:
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)

    sep = "=" * 120
    print(f"\n{sep}")
    print("GAMMA SWEEP ANALYSIS")
    print(sep)
    print(
        f"{'g':>4}  {'N':>4}  {'Accept%':>8}  {'+/-':>6}  {'HitRate%':>9}  {'+/-':>6}  "
        f"{'TPOT(ms)':>9}  {'+/-':>6}  {'NetIdle%':>9}  {'Tok/s':>7}  {'Tok/Rnd':>8}  {'TTFT(ms)':>9}"
    )
    print("-" * 120)
    for g in sorted(by_gamma):
        ss = by_gamma[g]
        n = len(ss)
        acc = [s["avg_acceptance_rate"] * 100 for s in ss]
        hit = [s["speculation_hit_rate"] * 100 for s in ss]
        tpot_vals = [s["tpot_ms"] for s in ss]
        ni = [s["network_overhead_pct"] for s in ss]
        tps = [s["tokens_per_second"] for s in ss]
        etr = [s["effective_tokens_per_round"] for s in ss]
        ttft_vals = [s["ttft_ms"] for s in ss]
        print(
            f"{g:>4}  {n:>4}  "
            f"{mean(acc):>8.1f}  {stdev(acc):>6.1f}  "
            f"{mean(hit):>9.1f}  {stdev(hit):>6.1f}  "
            f"{mean(tpot_vals):>9.1f}  {stdev(tpot_vals):>6.1f}  "
            f"{mean(ni):>9.1f}  "
            f"{mean(tps):>7.2f}  "
            f"{mean(etr):>8.2f}  "
            f"{mean(ttft_vals):>9.1f}"
        )
    print(sep + "\n")


def print_category_table(rows: list[dict]) -> None:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    if len(by_cat) <= 1:
        return

    sep = "=" * 90
    print(f"\n{sep}")
    print("CATEGORY BREAKDOWN (all gammas pooled)")
    print(sep)
    print(f"{'Category':<25}  {'N':>4}  {'Accept%':>8}  {'HitRate%':>9}  {'TPOT(ms)':>9}  {'Tok/s':>7}  {'Tok/Rnd':>8}")
    print("-" * 90)
    for cat in sorted(by_cat):
        ss = by_cat[cat]
        print(
            f"{cat:<25}  {len(ss):>4}  "
            f"{mean([s['avg_acceptance_rate'] for s in ss]) * 100:>8.1f}  "
            f"{mean([s['speculation_hit_rate'] for s in ss]) * 100:>9.1f}  "
            f"{mean([s['tpot_ms'] for s in ss]):>9.1f}  "
            f"{mean([s['tokens_per_second'] for s in ss]):>7.2f}  "
            f"{mean([s['effective_tokens_per_round'] for s in ss]):>8.2f}"
        )
    print(sep + "\n")


def print_miss_analysis(rows: list[dict]) -> None:
    total_rounds = sum(r["num_rounds"] for r in rows)
    total_misses = sum(r["miss_count"] for r in rows)
    total_hits = sum(r["hit_count"] for r in rows)

    print("\n=== SPECULATION MISS ANALYSIS ===")
    print(f"  Total rounds    : {total_rounds}")
    print(f"  Total hits      : {total_hits}  ({total_hits / max(total_rounds, 1) * 100:.1f}%)")
    print(f"  Total misses    : {total_misses}  ({total_misses / max(total_rounds, 1) * 100:.1f}%)")
    print("  Avg miss cost   : adds ~1 draft RTT per miss")
    acc_vals = [r["avg_acceptance_rate"] * 100 for r in rows]
    print("\n  Acceptance rate distribution:")
    print(f"    p25 = {pct(acc_vals, 25):.1f}%")
    print(f"    p50 = {pct(acc_vals, 50):.1f}%")
    print(f"    p75 = {pct(acc_vals, 75):.1f}%")
    print(f"    p95 = {pct(acc_vals, 95):.1f}%")
    print()


def print_latex_snippet(rows: list[dict]) -> None:
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)
    print("\n% ======= SpecSplit LaTeX macros (auto-generated by benchmarks/analyze_results.py) =======")
    for g in sorted(by_gamma):
        ss = by_gamma[g]
        prefix = f"SpS{g}"
        print(f"\\newcommand{{\\{prefix}AcceptRate}}{{{mean([s['avg_acceptance_rate']*100 for s in ss]):.1f}\\%}}")
        print(f"\\newcommand{{\\{prefix}HitRate}}{{{mean([s['speculation_hit_rate']*100 for s in ss]):.1f}\\%}}")
        print(f"\\newcommand{{\\{prefix}TPOT}}{{{mean([s['tpot_ms'] for s in ss]):.1f}ms}}")
        print(f"\\newcommand{{\\{prefix}TokPerSec}}{{{mean([s['tokens_per_second'] for s in ss]):.2f}}}")
        print(f"\\newcommand{{\\{prefix}NetOverhead}}{{{mean([s['network_overhead_pct'] for s in ss]):.1f}\\%}}")
    print("% ================================================================================\n")


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
DARK   = "#3D2B1F"
MID    = "#7A5C44"
LIGHT  = "#C9A87C"
ACCENT = "#8B2500"
COLORS = [DARK, ACCENT, MID, LIGHT, "#4A6741"]


# ---------------------------------------------------------------------------
# Individual figures (original set)
# ---------------------------------------------------------------------------

def fig_acceptance_vs_gamma(rows: list[dict], out: Path) -> None:
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)
    gammas = sorted(by_gamma)
    acc_means = [mean([s["avg_acceptance_rate"] * 100 for s in by_gamma[g]]) for g in gammas]
    acc_stds = [stdev([s["avg_acceptance_rate"] * 100 for s in by_gamma[g]]) for g in gammas]
    hit_means = [mean([s["speculation_hit_rate"] * 100 for s in by_gamma[g]]) for g in gammas]
    hit_stds = [stdev([s["speculation_hit_rate"] * 100 for s in by_gamma[g]]) for g in gammas]

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(len(gammas)))
    ax.errorbar(xs, acc_means, yerr=acc_stds, fmt="-o", color=DARK, capsize=4, label="Token Acceptance Rate", linewidth=2, markersize=7)
    ax.errorbar(xs, hit_means, yerr=hit_stds, fmt="--s", color=ACCENT, capsize=4, label="Speculation Hit Rate", linewidth=2, markersize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"γ={g}" for g in gammas])
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Draft Tree Depth (γ)")
    ax.set_title("Acceptance and Hit Rate vs. Draft Tree Depth", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_tpot_vs_gamma(rows: list[dict], out: Path) -> None:
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)
    gammas = sorted(by_gamma)
    tpot_means = [mean([s["tpot_ms"] for s in by_gamma[g]]) for g in gammas]
    tpot_stds = [stdev([s["tpot_ms"] for s in by_gamma[g]]) for g in gammas]
    etr_means = [mean([s["effective_tokens_per_round"] for s in by_gamma[g]]) for g in gammas]


    fig, ax1 = plt.subplots(figsize=(7, 4))
    xs = list(range(len(gammas)))
    ax1.errorbar(xs, tpot_means, yerr=tpot_stds, fmt="-o", color=DARK, capsize=4, label="TPOT (ms)", linewidth=2, markersize=7)
    ax1.set_ylabel("Time Per Output Token (ms)", color=DARK)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"γ={g}" for g in gammas])
    ax1.set_xlabel("Draft Tree Depth (γ)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(xs, etr_means, "--s", color=ACCENT, label="Tokens/Round", linewidth=2, markersize=7)
    ax2.set_ylabel("Effective Tokens per Round", color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ACCENT)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("TPOT and Throughput vs. Draft Tree Depth", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_acceptance_by_category(rows: list[dict], out: Path) -> None:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    if len(by_cat) <= 1:
        return
    cats = sorted(by_cat)
    acc_vals = [[s["avg_acceptance_rate"] * 100 for s in by_cat[c]] for c in cats]
    hit_vals = [[s["speculation_hit_rate"] * 100 for s in by_cat[c]] for c in cats]


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bp_kwargs = dict(
        patch_artist=True,
        boxprops=dict(facecolor=LIGHT, color=DARK),
        whiskerprops=dict(color=MID),
        capprops=dict(color=MID),
        medianprops=dict(color=ACCENT, linewidth=2),
        flierprops=dict(marker="o", markerfacecolor=ACCENT, markersize=5, linestyle="none"),
    )
    axes[0].boxplot(acc_vals, **bp_kwargs)
    axes[0].set_xticklabels(cats, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Token Acceptance Rate (%)")
    axes[0].set_title("Acceptance Rate by Category", fontweight="bold")
    axes[0].grid(True, axis="y")

    axes[1].boxplot(hit_vals, **bp_kwargs)
    axes[1].set_xticklabels(cats, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Speculation Hit Rate (%)")
    axes[1].set_title("Speculation Hit Rate by Category", fontweight="bold")
    axes[1].grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_per_round_acceptance(round_rows: list[dict], gamma: int, out: Path) -> None:
    target = [r for r in round_rows if r["gamma"] == gamma]
    if not target:
        return
    hit_rounds = [r["accepted"] for r in target if r["speculation_outcome"] == "hit"]
    miss_rounds = [r["accepted"] for r in target if r["speculation_outcome"] == "miss"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = [0, 1, 2, 3, 4, 5, 6]
    ax.hist(hit_rounds, bins=bins, alpha=0.7, color=DARK, label="Pipeline HIT", align="left")
    ax.hist(miss_rounds, bins=bins, alpha=0.7, color=ACCENT, label="Pipeline MISS", align="left")
    ax.set_xlabel(f"Tokens Accepted per Round (γ={gamma})")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Round Acceptance Distribution (γ={gamma})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_network_overhead(rows: list[dict], out: Path) -> None:
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)
    gammas = sorted(by_gamma)
    overhead_means = [mean([s["network_overhead_pct"] for s in by_gamma[g]]) for g in gammas]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"γ={g}" for g in gammas], overhead_means, color=COLORS[: len(gammas)])
    ax.axhline(y=5.88, color="red", linestyle="--", linewidth=1.5, label="Theoretical (2ms / 34ms ≈ 5.9%)")
    ax.set_ylabel("Network Idle Time (% of total latency)")
    ax.set_title("Network Overhead vs. Gamma", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y")
    for bar, val in zip(bars, overhead_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


# ---------------------------------------------------------------------------
# NEW figures
# ---------------------------------------------------------------------------

def fig_ttft_tpot_scatter(rows: list[dict], out: Path) -> None:
    """TTFT vs TPOT scatter, colored by acceptance rate, sized by tokens generated."""
    fig, ax = plt.subplots(figsize=(7, 5))

    gammas = sorted(set(r["gamma"] for r in rows))
    markers = ["o", "s", "^", "D", "P"]
    for i, g in enumerate(gammas):
        subset = [r for r in rows if r["gamma"] == g]
        ttft   = [r["ttft_ms"] for r in subset]
        tpot   = [r["tpot_ms"] for r in subset]
        acc    = [r["avg_acceptance_rate"] for r in subset]
        sizes  = [max(30, r["generated_tokens"] * 2) for r in subset]

        sc = ax.scatter(ttft, tpot, c=acc, cmap="YlOrBr", vmin=0, vmax=1,
                        s=sizes, marker=markers[i % len(markers)],
                        edgecolors=DARK, linewidths=0.5, alpha=0.85,
                        label=f"γ={g}", zorder=3)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Avg Token Acceptance Rate", fontsize=9)
    ax.set_xlabel("Time to First Token (ms)")
    ax.set_ylabel("Time per Output Token (ms)")
    ax.set_title("TTFT vs. TPOT\n(marker = γ, size = tokens generated, color = acceptance rate)",
                    fontsize=11, fontweight="bold")
    ax.legend(title="γ", fontsize=8, title_fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_acceptance_vs_hit_rate(rows: list[dict], out: Path) -> None:
    """Token acceptance rate vs speculation hit rate — reveals the coupling between them."""
    fig, ax = plt.subplots(figsize=(6, 5))

    gammas = sorted(set(r["gamma"] for r in rows))
    markers = ["o", "s", "^", "D", "P"]
    for i, g in enumerate(gammas):
        subset = [r for r in rows if r["gamma"] == g]
        acc    = [r["avg_acceptance_rate"] * 100 for r in subset]
        hit    = [r["speculation_hit_rate"] * 100 for r in subset]
        ax.scatter(acc, hit, color=COLORS[i % len(COLORS)],
                    marker=markers[i % len(markers)], s=70,
                    edgecolors=DARK, linewidths=0.5, alpha=0.85, label=f"γ={g}", zorder=3)

    # Diagonal reference line
    lo, hi = 0, 105
    ax.plot([lo, hi], [lo, hi], "--", color=MID, linewidth=1, alpha=0.5, label="Accept = Hit (ideal)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Token Acceptance Rate (%)")
    ax.set_ylabel("Speculation Hit Rate (%)")
    ax.set_title("Acceptance Rate vs. Speculation Hit Rate", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_per_prompt_heatmap(rows: list[dict], out: Path) -> None:
    """
    Heatmap of acceptance rate per (prompt_id, gamma).
    Useful for showing which prompts are hard vs easy regardless of γ.
    Only generated when multiple gammas are present.
    """
    gammas = sorted(set(r["gamma"] for r in rows))
    prompt_ids = sorted(set(r["prompt_id"] for r in rows))
    if len(gammas) <= 1 or len(prompt_ids) < 2:
        return

    # Build matrix: rows=prompts, cols=gammas
    matrix = []
    for pid in prompt_ids:
        row_vals = []
        for g in gammas:
            matches = [r["avg_acceptance_rate"] for r in rows if r["prompt_id"] == pid and r["gamma"] == g]
            row_vals.append(mean(matches) if matches else float("nan"))
        matrix.append(row_vals)

    fig_h = max(4, len(prompt_ids) * 0.35)
    fig, ax = plt.subplots(figsize=(max(5, len(gammas) * 1.2), fig_h))

    import numpy as np
    mat = np.array(matrix, dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrBr", vmin=0, vmax=1,
                    interpolation="nearest")

    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f"γ={g}" for g in gammas], fontsize=9)
    ax.set_yticks(range(len(prompt_ids)))
    ax.set_yticklabels(prompt_ids, fontsize=7)
    ax.set_xlabel("Draft Tree Depth (γ)")
    ax.set_title("Token Acceptance Rate per Prompt × γ", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Acceptance Rate", fontsize=9)

    # Annotate cells
    for i in range(len(prompt_ids)):
        for j in range(len(gammas)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.6 else DARK)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close("all")


def fig_hit_miss_breakdown(rows: list[dict], out: Path) -> None:
    """
    Stacked bar: hits vs misses per gamma.
    Also overlays TPOT on a secondary axis to show the cost relationship.
    """
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)
    gammas = sorted(by_gamma)

    total_hits   = [sum(r["hit_count"] for r in by_gamma[g]) for g in gammas]
    total_misses = [sum(r["miss_count"] for r in by_gamma[g]) for g in gammas]
    tpot_means   = [mean([r["tpot_ms"] for r in by_gamma[g]]) for g in gammas]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    xs = list(range(len(gammas)))
    width = 0.5

    ax1.bar(xs, total_hits,   width, label="Speculation HITs",   color=DARK,   alpha=0.85)
    ax1.bar(xs, total_misses, width, bottom=total_hits, label="Speculation MISSes", color=ACCENT, alpha=0.85)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"γ={g}" for g in gammas])
    ax1.set_ylabel("Total Pipeline Rounds")
    ax1.set_title("Speculation Hit/Miss Breakdown with TPOT", fontsize=12, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(xs, tpot_means, "D--", color=MID, linewidth=2, markersize=8, label="Avg TPOT (ms)")
    ax2.set_ylabel("Avg TPOT (ms)", color=MID)
    ax2.tick_params(axis="y", labelcolor=MID)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(MID)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_round_timeline(round_rows: list[dict], prompt_id: str, gamma: int, out: Path) -> None:
    """
    Per-round timeline for a single (prompt_id, gamma): verify RPC time,
    draft RPC time, and accepted tokens. Shows the overlap savings in context.
    """
    target = [r for r in round_rows if r["prompt_id"] == prompt_id and r["gamma"] == gamma]
    if not target:
        print(f"  [skip] No round data for prompt_id={prompt_id} gamma={gamma}")
        return
    target = sorted(target, key=lambda r: r["round_idx"])

    rounds      = [r["round_idx"] for r in target]
    verify_ms   = [r["verify_rpc_ms"] for r in target]
    draft_ms    = [r["draft_rpc_ms"] for r in target]
    accepted    = [r["accepted"] for r in target]
    outcomes    = [r["speculation_outcome"] for r in target]


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: RPC timings
    ax1.bar(rounds, verify_ms, label="Verify RPC (ms)", color=DARK, alpha=0.8, width=0.4, align="center")
    ax1.bar([r + 0.42 for r in rounds], draft_ms, label="Draft RPC (ms)",
            color=LIGHT, alpha=0.9, width=0.4, align="center")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"Round-by-Round Pipeline Timing\n(prompt: {prompt_id}, γ={gamma})",
                    fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.5)

    # Mark misses on top panel
    for r in target:
        if r["speculation_outcome"] == "miss":
            ax1.axvline(r["round_idx"], color=ACCENT, linestyle=":", alpha=0.6, linewidth=1)

    # Bottom: accepted tokens per round, colored by outcome
    bar_colors = [DARK if o == "hit" else ACCENT for o in outcomes]
    ax2.bar(rounds, accepted, color=bar_colors, alpha=0.85, width=0.7)
    ax2.set_ylabel("Tokens Accepted")
    ax2.set_xlabel("Round Index")
    ax2.set_ylim(0, max(accepted) + 1 if accepted else 6)
    ax2.grid(True, axis="y", alpha=0.5)

    # Legend for bottom
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=DARK, label="HIT"), Patch(color=ACCENT, label="MISS")],
                fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_verify_vs_draft_time(round_rows: list[dict], out: Path) -> None:
    """
    Scatter of verify_rpc_ms vs draft_rpc_ms for all rounds.
    Draws the verify = N * draft lines to show how much overlap budget exists.
    """
    if not round_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    gammas = sorted(set(r["gamma"] for r in round_rows))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    all_verify = [r["verify_rpc_ms"] for r in round_rows if r["verify_rpc_ms"] > 0]
    all_draft  = [r["draft_rpc_ms"]  for r in round_rows if r["draft_rpc_ms"]  > 0]

    for i, g in enumerate(gammas):
        subset = [r for r in round_rows if r["gamma"] == g and r["verify_rpc_ms"] > 0]
        ax.scatter(
            [r["draft_rpc_ms"] for r in subset],
            [r["verify_rpc_ms"] for r in subset],
            marker=markers[i % len(markers)],
            s=25,
            label=f"γ={g}",
            zorder=3,
        )

    # Reference lines: verify = k * draft
    if all_draft and all_verify:
        import numpy as np
        d_max = max(all_draft) * 1.05
        ds = np.linspace(0, d_max, 200)
        for mult, lbl in [(1, "1×"), (2, "2×"), (4, "4×")]:
            ax.plot(ds, mult * ds, linestyle="--", linewidth=1,
                    color=MID, alpha=0.5 + 0.1 * mult)
            ax.text(d_max * 0.92, mult * d_max * 0.92, lbl,
                    fontsize=8, color=MID, alpha=0.8)

    ax.set_xlabel("Draft RPC Time (ms)")
    ax.set_ylabel("Verify RPC Time (ms)")
    ax.set_title("Verify vs. Draft RPC Time per Round\n(points above 1× line = verify dominates → overlap useful)",
                    fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_verify_vs_draft_time_loglog(round_rows: list[dict], out: Path) -> None:
    """
    Log-log scatter of verify_rpc_ms vs draft_rpc_ms for all rounds.
    Useful when points form clustered bands at different latency scales.
    """
    if not round_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    gammas = sorted(set(r["gamma"] for r in round_rows))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    positive_rows = [r for r in round_rows if r["verify_rpc_ms"] > 0 and r["draft_rpc_ms"] > 0]
    if not positive_rows:
        return

    for i, g in enumerate(gammas):
        subset = [r for r in positive_rows if r["gamma"] == g]
        ax.scatter(
            [r["draft_rpc_ms"] for r in subset],
            [r["verify_rpc_ms"] for r in subset],
            marker=markers[i % len(markers)],
            s=25,
            label=f"γ={g}",
            zorder=3,
        )

    import numpy as np

    all_draft = np.array([r["draft_rpc_ms"] for r in positive_rows])
    all_verify = np.array([r["verify_rpc_ms"] for r in positive_rows])

    x_lo, x_hi = np.percentile(all_draft, [2, 98])
    y_lo, y_hi = np.percentile(all_verify, [2, 98])
    x_min = max(x_lo * 0.9, 1e-3)
    x_max = max(x_hi * 1.1, x_min * 10)
    y_min = max(y_lo * 0.9, 1e-3)
    y_max = max(y_hi * 1.1, y_min * 10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ds = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    for mult, lbl in [(1, "1×"), (2, "2×"), (4, "4×")]:
        ax.plot(ds, mult * ds, linestyle="--", linewidth=1, color=MID, alpha=0.55)
        text_x = x_max / 1.35
        text_y = mult * text_x
        if y_min < text_y < y_max:
            ax.text(text_x, text_y, lbl, fontsize=8, color=MID, alpha=0.8)

    ax.set_xlabel("Draft RPC Time (ms, log scale)")
    ax.set_ylabel("Verify RPC Time (ms, log scale)")
    ax.set_title(
        "Verify vs. Draft RPC Time per Round (log-log)\n(points above 1× line = verify dominates)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close("all")


def fig_tokens_per_second_by_prompt(rows: list[dict], out: Path) -> None:
    """
    Horizontal bar chart of tokens/sec per prompt, grouped by gamma.
    Shows which prompts benefit most from speculation.
    """
    prompt_ids = sorted(set(r["prompt_id"] for r in rows))
    gammas = sorted(set(r["gamma"] for r in rows))

    fig_h = max(4, len(prompt_ids) * 0.45 * len(gammas) * 0.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    n_gammas = len(gammas)
    bar_height = 0.8 / max(n_gammas, 1)
    offsets = [(i - (n_gammas - 1) / 2) * bar_height for i in range(n_gammas)]

    for i, g in enumerate(gammas):
        tps_vals = []
        for pid in prompt_ids:
            matches = [r["tokens_per_second"] for r in rows if r["prompt_id"] == pid and r["gamma"] == g]
            tps_vals.append(mean(matches) if matches else 0.0)
        ys = [j + offsets[i] for j in range(len(prompt_ids))]
        ax.barh(ys, tps_vals, height=bar_height * 0.9,
                color=COLORS[i % len(COLORS)], alpha=0.85, label=f"γ={g}")

    ax.set_yticks(range(len(prompt_ids)))
    ax.set_yticklabels(prompt_ids, fontsize=8)
    ax.set_xlabel("Tokens per Second")
    ax.set_title("Throughput (tok/s) per Prompt", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close("all")


def fig_acceptance_rate_distribution(rows: list[dict], out: Path) -> None:
    """
    Violin + strip plot of acceptance rate distribution per gamma.
    More informative than a single mean when N is small.
    """
    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)
    gammas = sorted(by_gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, (field, label, scale) in enumerate([
        ("avg_acceptance_rate", "Token Acceptance Rate (%)", 100),
        ("speculation_hit_rate", "Speculation Hit Rate (%)", 100),
    ]):
        ax = axes[ax_idx]
        data = [[r[field] * scale for r in by_gamma[g]] for g in gammas]
        positions = list(range(len(gammas)))

        # Violin if enough data, else box
        try:
            import numpy as np
            parts = ax.violinplot(data, positions=positions, showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(LIGHT)
                pc.set_edgecolor(DARK)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color(ACCENT)
            parts["cmedians"].set_linewidth(2)
            for key in ("cbars", "cmins", "cmaxes"):
                if key in parts:
                    parts[key].set_color(MID)
        except Exception:
            ax.boxplot(data, positions=positions, patch_artist=True,
                        boxprops=dict(facecolor=LIGHT, color=DARK),
                        medianprops=dict(color=ACCENT, linewidth=2))

        # Strip points
        import numpy as np
        for j, (g, vals) in enumerate(zip(gammas, data)):
            jitter = np.random.default_rng(42).uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(np.array([j] * len(vals)) + jitter, vals,
                        color=DARK, s=25, alpha=0.7, zorder=3, edgecolors="none")

        ax.set_xticks(positions)
        ax.set_xticklabels([f"γ={g}" for g in gammas])
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(-5, 110)
        ax.grid(True, axis="y", alpha=0.5)

    fig.suptitle("Rate Distributions by Draft Tree Depth", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close("all")


def fig_dashboard(rows: list[dict], round_rows: list[dict], out: Path) -> None:
    """
    Single-page 2×3 dashboard suitable for the paper appendix or a README.
    Combines: (1) acceptance vs gamma, (2) hit rate vs gamma, (3) TPOT violin,
    (4) hit/miss stacked bar, (5) tokens/s scatter, (6) per-round acceptance dist.
    """

    by_gamma: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)
    gammas = sorted(by_gamma)
    if not gammas:
        return

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    # --- Panel 1: Acceptance rate by gamma (bar + error) ---
    ax1 = fig.add_subplot(gs[0, 0])
    acc_m = [mean([r["avg_acceptance_rate"] * 100 for r in by_gamma[g]]) for g in gammas]
    acc_s = [stdev([r["avg_acceptance_rate"] * 100 for r in by_gamma[g]]) for g in gammas]
    xs = list(range(len(gammas)))
    bars = ax1.bar(xs, acc_m, color=DARK, alpha=0.85, width=0.55)
    ax1.errorbar(xs, acc_m, yerr=acc_s, fmt="none", color=MID, capsize=4, linewidth=1.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"γ={g}" for g in gammas], fontsize=8)
    ax1.set_ylabel("Acceptance Rate (%)")
    ax1.set_title("Token Acceptance Rate", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.grid(True, axis="y", alpha=0.5)
    for bar, val in zip(bars, acc_m):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=7)

    # --- Panel 2: Speculation hit rate by gamma ---
    ax2 = fig.add_subplot(gs[0, 1])
    hit_m = [mean([r["speculation_hit_rate"] * 100 for r in by_gamma[g]]) for g in gammas]
    hit_s = [stdev([r["speculation_hit_rate"] * 100 for r in by_gamma[g]]) for g in gammas]
    bars2 = ax2.bar(xs, hit_m, color=ACCENT, alpha=0.85, width=0.55)
    ax2.errorbar(xs, hit_m, yerr=hit_s, fmt="none", color=MID, capsize=4, linewidth=1.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"γ={g}" for g in gammas], fontsize=8)
    ax2.set_ylabel("Hit Rate (%)")
    ax2.set_title("Speculation Hit Rate", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis="y", alpha=0.5)
    for bar, val in zip(bars2, hit_m):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=7)

    # --- Panel 3: TPOT by gamma ---
    ax3 = fig.add_subplot(gs[0, 2])
    tpot_data = [[r["tpot_ms"] for r in by_gamma[g]] for g in gammas]
    try:
        parts = ax3.violinplot(tpot_data, positions=xs, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(LIGHT)
            pc.set_edgecolor(DARK)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color(ACCENT)
        parts["cmedians"].set_linewidth(2)
    except Exception:
        ax3.boxplot(tpot_data, positions=xs, patch_artist=True,
                    boxprops=dict(facecolor=LIGHT, color=DARK),
                    medianprops=dict(color=ACCENT, linewidth=2))
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"γ={g}" for g in gammas], fontsize=8)
    ax3.set_ylabel("TPOT (ms)")
    ax3.set_title("Time Per Output Token", fontweight="bold", fontsize=10)
    ax3.grid(True, axis="y", alpha=0.5)

    # --- Panel 4: Hit/Miss stacked bars ---
    ax4 = fig.add_subplot(gs[1, 0])
    hits_total   = [sum(r["hit_count"]  for r in by_gamma[g]) for g in gammas]
    misses_total = [sum(r["miss_count"] for r in by_gamma[g]) for g in gammas]
    ax4.bar(xs, hits_total,   color=DARK,   alpha=0.85, label="HITs",   width=0.55)
    ax4.bar(xs, misses_total, color=ACCENT, alpha=0.85, label="MISSes",
            bottom=hits_total, width=0.55)
    ax4.set_xticks(xs)
    ax4.set_xticklabels([f"γ={g}" for g in gammas], fontsize=8)
    ax4.set_ylabel("Rounds")
    ax4.set_title("Hit / Miss Count", fontweight="bold", fontsize=10)
    ax4.legend(fontsize=7)
    ax4.grid(True, axis="y", alpha=0.5)

    # --- Panel 5: Tokens/s scatter per prompt ---
    ax5 = fig.add_subplot(gs[1, 1])
    for i, g in enumerate(gammas):
        subset = [r for r in rows if r["gamma"] == g]
        tps = [r["tokens_per_second"] for r in subset]
        acc = [r["avg_acceptance_rate"] for r in subset]
        ax5.scatter(acc, tps, color=COLORS[i % len(COLORS)], s=40, alpha=0.8,
                    label=f"γ={g}", edgecolors="none", zorder=3)
    ax5.set_xlabel("Acceptance Rate")
    ax5.set_ylabel("Tokens per Second")
    ax5.set_title("Acceptance Rate vs. Throughput", fontweight="bold", fontsize=10)
    ax5.legend(fontsize=7)
    ax5.grid(True)

    # --- Panel 6: Per-round acceptance histogram (largest gamma) ---
    ax6 = fig.add_subplot(gs[1, 2])
    if round_rows:
        g_max = max(gammas)
        target = [r for r in round_rows if r["gamma"] == g_max]
        hit_r  = [r["accepted"] for r in target if r["speculation_outcome"] == "hit"]
        miss_r = [r["accepted"] for r in target if r["speculation_outcome"] == "miss"]
        bins   = list(range(0, g_max + 3))
        ax6.hist(hit_r,  bins=bins, alpha=0.7, color=DARK,   label="HIT",  align="left")
        ax6.hist(miss_r, bins=bins, alpha=0.7, color=ACCENT, label="MISS", align="left")
        ax6.set_xlabel(f"Tokens Accepted (γ={g_max})")
        ax6.set_ylabel("Count")
        ax6.set_title("Per-Round Acceptance Dist.", fontweight="bold", fontsize=10)
        ax6.legend(fontsize=7)
        ax6.grid(True, axis="y", alpha=0.5)
    else:
        ax6.text(0.5, 0.5, "No per-round data", ha="center", va="center",
                    transform=ax6.transAxes, fontsize=10, color=MID)

    fig.suptitle("SpecSplit Pipeline Benchmark — Dashboard", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close("all")


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_plots(rows: list[dict], round_rows: list[dict], fig_dir: Path,
                   gamma_for_round_plot: int = 5) -> None:
    """Generate the full figure set into fig_dir."""
    if not HAVE_MPL:
        print("Skipping plots (matplotlib not available). Install with: pip install matplotlib")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures in {fig_dir}/")

    # Original set
    fig_acceptance_vs_gamma(rows,   fig_dir / "acceptance_vs_gamma.png")
    fig_tpot_vs_gamma(rows,         fig_dir / "tpot_vs_gamma.png")
    fig_acceptance_by_category(rows, fig_dir / "acceptance_by_category.png")
    fig_network_overhead(rows,      fig_dir / "network_overhead.png")
    if round_rows:
        fig_per_round_acceptance(round_rows, gamma_for_round_plot,
                                 fig_dir / f"per_round_dist_gamma{gamma_for_round_plot}.png")

    # New figures
    fig_ttft_tpot_scatter(rows,              fig_dir / "ttft_vs_tpot_scatter.png")
    fig_acceptance_vs_hit_rate(rows,         fig_dir / "acceptance_vs_hit_rate.png")
    fig_per_prompt_heatmap(rows,             fig_dir / "acceptance_heatmap.png")
    fig_hit_miss_breakdown(rows,             fig_dir / "hit_miss_breakdown.png")
    fig_tokens_per_second_by_prompt(rows,    fig_dir / "tokens_per_second_by_prompt.png")
    fig_acceptance_rate_distribution(rows,   fig_dir / "acceptance_distribution.png")
    fig_verify_vs_draft_time(round_rows,     fig_dir / "verify_vs_draft_time.png")
    fig_verify_vs_draft_time_loglog(round_rows, fig_dir / "verify_vs_draft_time_loglog.png")

    # Dashboard summary (most useful for paper / README)
    fig_dashboard(rows, round_rows,          fig_dir / "dashboard.png")

    # Per-prompt round timelines (for a couple representative prompts)
    if round_rows:
        gammas = sorted(set(r["gamma"] for r in round_rows))
        prompt_ids = sorted(set(r["prompt_id"] for r in round_rows))
        for pid in prompt_ids[:3]:
            for g in gammas[:1]:
                fig_round_timeline(
                    round_rows, pid, g,
                    fig_dir / f"round_timeline_{pid}_gamma{g}.png",
                )

    print(f"\nAll figures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--latex", action="store_true", help="Print LaTeX macros")
    p.add_argument("--gamma-for-round-plot", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_summary(args.results_dir)
    round_rows = load_rounds(args.results_dir)
    print_gamma_table(rows)
    print_category_table(rows)
    print_miss_analysis(rows)
    if args.latex:
        print_latex_snippet(rows)

    if not args.no_plots:
        generate_plots(
            rows, round_rows,
            fig_dir=args.results_dir / "figures",
            gamma_for_round_plot=args.gamma_for_round_plot,
        )


if __name__ == "__main__":
    main()
