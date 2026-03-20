#!/usr/bin/env python3
"""Renamed entrypoint for the old benchmark harness.

Preferred modern runner:
    python3 benchmarks/runner.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    old_script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_run.py"
    runpy.run_path(str(old_script), run_name="__main__")


if __name__ == "__main__":
    main()
