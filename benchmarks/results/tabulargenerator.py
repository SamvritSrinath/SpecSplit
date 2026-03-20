#!/usr/bin/env python3
"""
Reads SpecSplit summary.csv and generates publication-ready LaTeX tables 
using the booktabs package for a two-column systems paper.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def load_summary(csv_path: Path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns
            for k, v in row.items():
                try:
                    row[k] = float(v) if '.' in v else int(v)
                except ValueError:
                    pass
            rows.append(row)
    return rows

def generate_gamma_table(rows):
    by_gamma = defaultdict(list)
    for r in rows:
        by_gamma[r["gamma"]].append(r)

    print("% " + "="*50)
    print("% TABLE 1: GAMMA SWEEP")
    print("% " + "="*50)
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{System performance across varying Draft Lengths ($\\gamma$).}")
    print("\\label{tab:gamma_sweep}")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{@{}lccccc@{}}")
    print("\\toprule")
    print("\\textbf{$\\gamma$} & \\textbf{Accept \\%} & \\textbf{Hit \\%} & \\textbf{TPOT (ms)} & \\textbf{Tok/s} & \\textbf{Tok/Round} \\\\ \\midrule")

    for gamma in sorted(by_gamma.keys()):
        group = by_gamma[gamma]
        acc = statistics.mean([r["avg_acceptance_rate"] for r in group]) * 100
        hit = statistics.mean([r["speculation_hit_rate"] for r in group]) * 100
        tpot = statistics.mean([r["tpot_ms"] for r in group])
        tps = statistics.mean([r["tokens_per_second"] for r in group])
        tpr = statistics.mean([r["effective_tokens_per_round"] for r in group])

        print(f"{gamma} & {acc:.1f}\\% & {hit:.1f}\\% & {tpot:.1f} & {tps:.2f} & {tpr:.2f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}\n")

def generate_category_table(rows):
    # Filter for the optimal gamma if desired, or aggregate all.
    # Here we aggregate all, but you could filter by: rows = [r for r in rows if r["gamma"] == 5]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    print("% " + "="*50)
    print("% TABLE 2: CATEGORY BREAKDOWN")
    print("% " + "="*50)
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Acceptance and Hit Rates grouped by workload category.}")
    print("\\label{tab:category_breakdown}")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("\\textbf{Category} & \\textbf{Accept \\%} & \\textbf{Hit \\%} & \\textbf{TPOT (ms)} & \\textbf{Tok/s} \\\\ \\midrule")

    # Calculate stats and sort by Acceptance Rate (descending)
    stats = []
    for cat, group in by_cat.items():
        acc = statistics.mean([r["avg_acceptance_rate"] for r in group]) * 100
        hit = statistics.mean([r["speculation_hit_rate"] for r in group]) * 100
        tpot = statistics.mean([r["tpot_ms"] for r in group])
        tps = statistics.mean([r["tokens_per_second"] for r in group])

        # Clean up category names for the paper (e.g., "factual_qa" -> "Factual QA")
        clean_cat = cat.replace("_", " ").title()
        if clean_cat == "Factual Qa": clean_cat = "Factual QA"

        stats.append((clean_cat, acc, hit, tpot, tps))

    stats.sort(key=lambda x: x[1], reverse=True)

    for s in stats:
        # Bold the top performing row for emphasis
        if s == stats[0]:
            print(f"\\textbf{{{s[0]}}} & \\textbf{{{s[1]:.1f}\\%}} & \\textbf{{{s[2]:.1f}\\%}} & \\textbf{{{s[3]:.1f}}} & \\textbf{{{s[4]:.2f}}} \\\\")
        else:
            print(f"{s[0]} & {s[1]:.1f}\\% & {s[2]:.1f}\\% & {s[3]:.1f} & {s[4]:.2f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}\n")

if __name__ == "__main__":
    csv_file = Path("benchmarks/results/summary.csv")
    if not csv_file.exists():
        print("Error: summary.csv not found in the current directory.")
    else:
        data = load_summary(csv_file)
        generate_gamma_table(data)
        generate_category_table(data)
