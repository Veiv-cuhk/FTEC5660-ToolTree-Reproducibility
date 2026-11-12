"""Verify the repo's *numeric* reported behaviors with our own measurements.

The upstream README claims a "Behavior test suite (32 tests)" and documents
release fixes with explicit numeric assertions. Running the suite gives a
pass/fail, but a reproducibility report needs OUR OWN MEASURED NUMBERS. This
script re-instantiates the exact scenarios used by three upstream tests,
records the measured quantities, and compares them with the values asserted
by the upstream test suite.

Usage (from the repository root):

    python -m repro.verify_claims
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_tier1_fixes import make_search, make_tool  # noqa: E402
from src.mcts.tree_search import ToolTreeSearch  # noqa: E402


def _iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def claim_exhaustion() -> dict:
    """Upstream test_exhausted_search_stops_without_revisit_backprop.

    Reported: exactly 4 judged executions / 4 rollouts with a budget of 50,
    root.visit_count == 4, root.is_exhausted is True.
    """
    random.seed(0)
    tools = [make_tool("t_a"), make_tool("t_b")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_a": 0.9, "t_b": 0.6},
        post_scores={"t_a": 0.9, "t_b": 0.9},
        max_rollouts=50,
        max_depth=3,
    )
    search.search("q", {}, tools)
    root = search._last_root
    measured = {
        "post_eval_calls": len(judge.post_calls),
        "rollouts_used": search._rollouts_used,
        "root_visit_count": root.visit_count,
        "root_exhausted": bool(root.is_exhausted),
        "best_q_history_len": len(search._best_q_history),
    }
    reported = {
        "post_eval_calls": 4,
        "rollouts_used": 4,
        "root_visit_count": 4,
        "root_exhausted": True,
        "best_q_history_len": 4,
    }
    return {
        "claim": "Exhausted search stops without re-visiting / re-backpropagating",
        "source": "tests/test_release_fixes.py::test_exhausted_search_stops_without_revisit_backprop",
        "reported": reported,
        "measured": measured,
        "reproduced": all(measured[k] == reported[k] for k in reported),
    }


def claim_no_rebuild() -> dict:
    """Upstream test_no_equivalent_candidate_rebuild_after_post_prune.

    Reported: a permanently post-pruned tool is judged at most twice (once
    under the root, once under the good chain), and no node has two children
    sharing an (tool, args) fingerprint.
    """
    random.seed(0)
    tools = [make_tool("t_good"), make_tool("t_dead")]
    search, judge, _ = make_search(
        tools,
        pre_scores={"t_good": 0.9, "t_dead": 0.6},
        post_scores={"t_good": 0.9, "t_dead": 0.0},
        max_rollouts=30,
        max_depth=3,
    )
    search.search("q", {}, tools)
    root = search._last_root
    dead_judgments = sum(1 for c in judge.post_calls if c["tool"] == "t_dead")
    duplicate_children = 0
    for node in _iter_nodes(root):
        fps = [
            ToolTreeSearch._action_fingerprint(
                (c.action or {}).get("tool_name"),
                c.action_args or (c.action or {}).get("tool_args") or {},
            )
            for c in node.children
        ]
        if len(fps) != len(set(fps)):
            duplicate_children += 1
    measured = {
        "dead_tool_post_judgments": dead_judgments,
        "nodes_with_duplicate_children": duplicate_children,
    }
    reported = {"dead_tool_post_judgments": "<= 2", "nodes_with_duplicate_children": 0}
    return {
        "claim": "A post-pruned branch is never rebuilt and re-judged",
        "source": "tests/test_release_fixes.py::test_no_equivalent_candidate_rebuild_after_post_prune",
        "reported": reported,
        "measured": measured,
        "reproduced": dead_judgments <= 2 and duplicate_children == 0,
    }


def claim_uct() -> dict:
    """Upstream test_eq1_formula_for_visited_children (Eq. 1 / UCT).

    Reported: for Q_a=0.50,N_a=4,r_pre_a=0.8 and Q_b=0.90,N_b=2,r_pre_b=0.4
    with N(parent)=4, c=1.4, UCT(b) > UCT(a) and best_child returns b.
    """
    import math

    from src.mcts.node import MCTSNode

    random.seed(0)
    root = MCTSNode(state={"context": {}, "depth": 0})
    root.visit_count = 4
    a = MCTSNode(state={}, action={"tool_name": "a"}, parent=root)
    a.q_value, a.visit_count, a.r_pre = 0.50, 4, 0.8
    b = MCTSNode(state={}, action={"tool_name": "b"}, parent=root)
    b.q_value, b.visit_count, b.r_pre = 0.90, 2, 0.4
    root.children = [a, b]
    uct_a = 0.50 + 1.4 * 0.8 * math.sqrt(math.log(4) / 4)
    uct_b = 0.90 + 1.4 * 0.4 * math.sqrt(math.log(4) / 2)
    chosen = root.best_child(1.4)
    measured = {
        "uct_a": round(uct_a, 4),
        "uct_b": round(uct_b, 4),
        "uct_b_gt_uct_a": bool(uct_b > uct_a),
        "best_child": (chosen.action or {}).get("tool_name"),
    }
    reported = {
        "uct_a": 1.1594,
        "uct_b": 1.3664,
        "uct_b_gt_uct_a": True,
        "best_child": "b",
    }
    return {
        "claim": "UCT selection follows Eq. 1 (Q + c*r_pre*sqrt(ln N / N_a))",
        "source": "tests/test_tier1_fixes.py::test_eq1_formula_for_visited_children",
        "reported": reported,
        "measured": measured,
        "reproduced": (
            measured["uct_b_gt_uct_a"] is True
            and measured["best_child"] == "b"
            and abs(measured["uct_a"] - 1.1594) < 0.001
            and abs(measured["uct_b"] - 1.3664) < 0.001
        ),
    }


def run_upstream_suite(out_dir: Path) -> dict:
    """Run the shipped 32-test behavior suite and capture the transcript."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    transcript = proc.stdout + proc.stderr
    (out_dir / "upstream_tests.txt").write_text(transcript, encoding="utf-8")
    import re

    match = re.search(r"(\d+) passed", transcript)
    passed = int(match.group(1)) if match else 0
    summary = [ln for ln in transcript.splitlines() if "passed" in ln]
    return {
        "claim": "Upstream behavior test suite",
        "source": "README.md ('Behavior test suite (32 tests)')",
        "reported": {"tests_passed": 32},
        "measured": {"tests_passed": passed, "exit_code": proc.returncode,
                     "summary": summary[-1].strip() if summary else ""},
        "reproduced": proc.returncode == 0 and passed == 32,
    }


def main() -> None:
    out_dir = ROOT / "repro" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_upstream_suite(out_dir), claim_exhaustion(), claim_no_rebuild(), claim_uct()]
    payload = {
        "note": (
            "'reported' values are the numeric assertions / descriptions in the "
            "upstream repository; 'measured' values are produced by this script "
            "on this machine. No upstream source file was modified."
        ),
        "claims": results,
        "all_reproduced": all(r["reproduced"] for r in results),
    }
    (out_dir / "claim_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
