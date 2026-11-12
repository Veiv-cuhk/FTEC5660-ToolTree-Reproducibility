"""Run the paired reproduction experiment and write raw per-run measurements.

Usage (from the repository root):

    # main protocol (early stopping disabled so pruning's effect is visible)
    python -m repro.run_experiment --seeds 5 --per-family 6

    # sensitivity protocol (shipped early-stop default, applied to BOTH arms)
    python -m repro.run_experiment --seeds 5 --per-family 6 --early-stop-patience 10 --out repro/results/sensitivity_defaultstop

Outputs `raw_runs.csv` (one row per task x seed x condition) and
`run_manifest.json` (environment / provenance information) in the chosen
output directory.

Design: 24 synthetic tasks x 5 seeds x 2 conditions = 240 measured searches.
Seeds are PAIRED -- baseline and ablation see identical score jitter and UCT
tie-break randomness, so the only variable is tau_post.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from repro.harness import run_one
from repro.tasks import generate_tasks

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "baseline": ROOT / "repro" / "configs" / "baseline.yaml",
    "ablation_no_post_pruning": ROOT / "repro" / "configs" / "ablation_no_post_pruning.yaml",
}


UPSTREAM_COMMIT = "fed7a608ca54ba2e75078f9a1f409c9aa7fab18f"


def _upstream_commit() -> str:
    """The pinned upstream ToolTree commit imported into this fork.

    Note: `git log -1` here would report our own vendor-import commit, not the
    upstream hash, so the upstream commit is pinned explicitly (and recorded
    in README.md and docs/UPSTREAM_README.md).
    """
    return UPSTREAM_COMMIT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="number of paired trials")
    ap.add_argument("--per-family", type=int, default=6, help="tasks per family")
    ap.add_argument(
        "--out", type=str, default=str(ROOT / "repro" / "results"), help="output dir"
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help=(
            "sensitivity protocol: override early_stop_patience in BOTH "
            "conditions (e.g. 10 for the shipped default). This is a protocol "
            "knob, not the experimental modification."
        ),
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(args.per_family)
    rows = []
    override = (
        {"early_stop_patience": args.early_stop_patience}
        if args.early_stop_patience is not None
        else None
    )

    for condition, config_path in CONFIGS.items():
        for seed in range(1, args.seeds + 1):
            for idx, task in enumerate(tasks):
                rows.append(
                    run_one(task, str(config_path), seed, condition, idx, override)
                )

    fieldnames = list(rows[0].keys())
    with (out / "raw_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, "-m", "repro.run_experiment", *sys.argv[1:]]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "upstream_head": _upstream_commit(),
        "upstream_project": "SYang2000/ICLR_2026_ToolTree",
        "configs": {k: str(v.relative_to(ROOT)) for k, v in CONFIGS.items()},
        "n_tasks": len(tasks),
        "task_families": sorted({t["family"] for t in tasks}),
        "n_seeds": args.seeds,
        "n_conditions": len(CONFIGS),
        "n_runs": len(rows),
        "jitter_sigma": 0.03,
        "paired_seeds": True,
        "protocol_override": override,
        "llm_used": False,
        "llm_note": (
            "Planner/judge replaced by deterministic stubs adapted from the "
            "upstream tests/test_tier1_fixes.py. No API key required; no "
            "network calls at run time."
        ),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(rows)} measurement rows to {out / 'raw_runs.csv'}")
    print(f"Wrote provenance manifest to {out / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
