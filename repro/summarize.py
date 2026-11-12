"""Aggregate raw measurements into tables and figures for the report.

Usage (from the repository root):

    python -m repro.summarize

Reads:
    repro/results/raw_runs.csv                          (main protocol)
    repro/results/sensitivity_defaultstop/raw_runs.csv  (sensitivity protocol)
    repro/results/claim_verification.json               (reproduced claims)

Writes:
    repro/results/summary_by_seed.csv
    repro/results/summary_overall.json
    repro/results/table_results.md
    repro/results/figure_cost_quality.png
    repro/results/figure_by_family.png
    repro/results/sensitivity_defaultstop/table_results.md

All values are computed from the measured rows; nothing is hard-coded.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "repro" / "results"

BASELINE = "baseline"
ABLATION = "ablation_no_post_pruning"
CONDITIONS = (BASELINE, ABLATION)

QUALITY = ["plan_success", "tool_f1", "plan_f1", "gold_recall", "plan_length"]
COST = [
    "pre_eval_calls",
    "post_eval_calls",
    "planner_calls",
    "rollouts_used",
    "nodes_created",
    "post_pruned_nodes",
    "wall_ms",
    "best_q",
]
ALL_METRICS = QUALITY + COST
LABELS = {
    "plan_success": "Plan success (exact gold tool set)",
    "tool_f1": "Tool F1 (repo metric)",
    "plan_f1": "Plan F1 (repo metric)",
    "gold_recall": "Gold-tool recall",
    "plan_length": "Returned plan length",
    "pre_eval_calls": "Pre-evaluation calls",
    "post_eval_calls": "Post-evaluation calls",
    "planner_calls": "Planner (arg-draft) calls",
    "rollouts_used": "Rollouts used",
    "nodes_created": "Search-tree nodes",
    "post_pruned_nodes": "Post-pruned nodes",
    "wall_ms": "Wall time (ms)",
    "best_q": "Best Q value",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (st.mean(values), st.pstdev(values))


def summarise(rows: list[dict]) -> dict:
    """Per-condition metrics: mean over all runs, and mean +/- sd over seeds."""
    by_cond_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_cond_seed[(r["condition"], int(r["seed"]))].append(r)

    seed_means: dict[str, dict[int, dict[str, float]]] = {
        c: {} for c in CONDITIONS
    }
    for (cond, seed), group in by_cond_seed.items():
        seed_means[cond][seed] = {
            m: st.mean(float(r[m]) for r in group) for m in ALL_METRICS
        }

    overall: dict[str, dict[str, dict[str, float]]] = {}
    for cond in CONDITIONS:
        overall[cond] = {}
        for m in ALL_METRICS:
            across = [seed_means[cond][s][m] for s in sorted(seed_means[cond])]
            mu, sd = mean_sd(across)
            overall[cond][m] = {"mean": mu, "sd": sd, "n_seeds": len(across)}

    deltas: dict[str, dict[str, float]] = {}
    for m in ALL_METRICS:
        b, a = overall[BASELINE][m]["mean"], overall[ABLATION][m]["mean"]
        deltas[m] = {
            "absolute": a - b,
            "percent": (100.0 * (a - b) / b) if b else float("nan"),
        }
    return {"seed_means": seed_means, "overall": overall, "deltas": deltas}


def write_seed_csv(summary: dict, path: Path) -> None:
    fields = ["condition", "seed"] + ALL_METRICS
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond in CONDITIONS:
            for seed in sorted(summary["seed_means"][cond]):
                row = {"condition": cond, "seed": seed}
                row.update(summary["seed_means"][cond][seed])
                w.writerow(row)


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def table_markdown(summary: dict, title: str) -> str:
    o, d = summary["overall"], summary["deltas"]
    lines = [
        f"### {title}",
        "",
        f"Each cell is **mean +/- s.d. across {int(o[BASELINE]['tool_f1']['n_seeds'])} paired seeds**; "
        "each seed averages over 24 synthetic tasks. `Delta` = ablation - baseline.",
        "",
        "| Metric | Baseline (post-pruning ON) | Ablation (post-pruning OFF) | Delta | Delta % |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in ALL_METRICS:
        b, a = o[BASELINE][m], o[ABLATION][m]
        pct = d[m]["percent"]
        pct_s = "n/a" if pct != pct else f"{pct:+.1f}%"
        lines.append(
            f"| {LABELS[m]} | {_fmt(b['mean'])} +/- {_fmt(b['sd'])} | "
            f"{_fmt(a['mean'])} +/- {_fmt(a['sd'])} | "
            f"{d[m]['absolute']:+.3f} | {pct_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def family_table(rows: list[dict]) -> str:
    fams = sorted({r["family"] for r in rows})
    lines = [
        "### Per-family breakdown (main protocol)",
        "",
        "| Family | Tool F1 (ON) | Tool F1 (OFF) | Post-eval calls (ON) | Post-eval calls (OFF) | Nodes (ON) | Nodes (OFF) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in fams:
        vals = {}
        for cond in CONDITIONS:
            g = [r for r in rows if r["family"] == fam and r["condition"] == cond]
            vals[cond] = {m: st.mean(float(r[m]) for r in g) for m in
                          ("tool_f1", "post_eval_calls", "nodes_created")}
        b, a = vals[BASELINE], vals[ABLATION]
        lines.append(
            f"| {fam} | {b['tool_f1']:.3f} | {a['tool_f1']:.3f} | "
            f"{b['post_eval_calls']:.1f} | {a['post_eval_calls']:.1f} | "
            f"{b['nodes_created']:.1f} | {a['nodes_created']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def claims_table() -> str:
    payload = json.loads((RESULTS / "claim_verification.json").read_text(encoding="utf-8"))
    lines = [
        "### Reproduction of the repository's reported/pinned behaviors",
        "",
        "| Claim (source) | Reported | Measured here | Status |",
        "|---|---|---|---|",
    ]
    for c in payload["claims"]:
        rep = "; ".join(f"{k}={v}" for k, v in c["reported"].items())
        mea = "; ".join(f"{k}={v}" for k, v in c["measured"].items()
                        if k not in ("exit_code", "summary"))
        status = "Reproduced" if c["reproduced"] else "NOT reproduced"
        lines.append(f"| {c['claim']} (`{c['source']}`) | {rep} | {mea} | {status} |")
    lines.append("")
    return "\n".join(lines)


def figure_cost_quality(summary: dict, path: Path) -> None:
    o = summary["overall"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {BASELINE: "#2F6DB5", ABLATION: "#D9534F"}

    def grouped(ax, metrics, title, ylabel, pct=False):
        xs = range(len(metrics))
        w = 0.36
        for i, cond in enumerate(CONDITIONS):
            mus = [o[cond][m]["mean"] for m in metrics]
            sds = [o[cond][m]["sd"] for m in metrics]
            pos = [x + (i - 0.5) * w for x in xs]
            ax.bar(pos, mus, w, yerr=sds, capsize=3, label=cond,
                   color=colors[cond], alpha=0.9,
                   error_kw={"elinewidth": 1, "ecolor": "#333333"})
        ax.set_xticks(list(xs))
        ax.set_xticklabels([LABELS[m].split(" (")[0] for m in metrics],
                           rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    grouped(axes[0], ["post_eval_calls", "rollouts_used", "nodes_created"],
            "Search cost", "count / run")
    grouped(axes[1], ["plan_success", "tool_f1", "plan_f1"],
            "Plan quality", "score (0-1)")
    grouped(axes[2], ["pre_eval_calls", "planner_calls", "wall_ms"],
            "Other cost proxies", "count or ms / run")
    fig.suptitle("ToolTree: effect of disabling post-evaluation pruning (tau_post: 0.4 -> 0.0)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_by_family(rows: list[dict], path: Path) -> None:
    fams = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (metric, title, ylabel) in zip(axes, [
        ("tool_f1", "Plan quality by task family", "Tool F1"),
        ("post_eval_calls", "Search cost by task family", "post-eval calls / run"),
    ]):
        xs = range(len(fams))
        w = 0.36
        for i, cond in enumerate(CONDITIONS):
            mus = [st.mean(float(r[metric]) for r in rows
                           if r["family"] == f and r["condition"] == cond)
                   for f in fams]
            pos = [x + (i - 0.5) * w for x in xs]
            ax.bar(pos, mus, w, label=cond,
                   color="#2F6DB5" if cond == BASELINE else "#D9534F", alpha=0.9)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(fams, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    main_rows = read_csv(RESULTS / "raw_runs.csv")
    sens_dir = RESULTS / "sensitivity_defaultstop"
    sens_rows = read_csv(sens_dir / "raw_runs.csv")

    main_summary = summarise(main_rows)
    sens_summary = summarise(sens_rows)

    write_seed_csv(main_summary, RESULTS / "summary_by_seed.csv")
    (RESULTS / "summary_overall.json").write_text(
        json.dumps({"main": main_summary, "sensitivity": sens_summary},
                   indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    figure_cost_quality(main_summary, RESULTS / "figure_cost_quality.png")
    figure_by_family(main_rows, RESULTS / "figure_by_family.png")

    report_tables = "\n".join([
        "# Measured results (generated by `python -m repro.summarize`)",
        "",
        claims_table(),
        table_markdown(main_summary, "Main protocol: baseline vs modification"),
        family_table(main_rows),
        table_markdown(sens_summary, "Sensitivity: shipped early-stop default (patience=10) in BOTH arms"),
        "",
        "All figures are generated from `raw_runs.csv`; no value in this file is hand-entered.",
    ])
    (RESULTS / "table_results.md").write_text(report_tables + "\n", encoding="utf-8")
    (sens_dir / "table_results.md").write_text(
        table_markdown(sens_summary, "Sensitivity protocol (patience=10)") + "\n",
        encoding="utf-8",
    )
    print(report_tables)


if __name__ == "__main__":
    main()
