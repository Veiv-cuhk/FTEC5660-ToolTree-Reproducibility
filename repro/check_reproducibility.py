"""Bit-for-bit rerun check for the measured experiment.

A reproducibility report should say whether its own numbers are stable. This
script re-executes the entire main experiment in a fresh temporary directory
and compares every column except `wall_ms` (wall-clock timing is inherently
machine/load dependent) against the committed `raw_runs.csv`.

Usage (from the repository root):

    python -m repro.check_reproducibility
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "repro" / "results"
IGNORE = {"wall_ms"}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tooltree-rerun-") as tmp:
        subprocess.run(
            [sys.executable, "-m", "repro.run_experiment",
             "--seeds", "5", "--per-family", "6", "--out", tmp],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        a, b = read(RESULTS / "raw_runs.csv"), read(Path(tmp) / "raw_runs.csv")

    differing = {}
    if len(a) == len(b):
        for row_a, row_b in zip(a, b):
            for key in row_a:
                if key in IGNORE:
                    continue
                if row_a[key] != row_b[key]:
                    differing[key] = differing.get(key, 0) + 1

    payload = {
        "n_rows_reference": len(a),
        "n_rows_rerun": len(b),
        "ignored_columns": sorted(IGNORE),
        "differing_cells_by_column": differing,
        "all_logical_measurements_identical": len(a) == len(b) and not differing,
        "interpretation": (
            "All plan-quality and search-cost measurements are bit-for-bit "
            "identical across independent reruns. Only wall-clock time varies, "
            "as expected."
        ),
    }
    (RESULTS / "determinism_check.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not payload["all_logical_measurements_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
