"""Measurement harness: drives the UNMODIFIED upstream ToolTree MCTS.

Everything that matters for the claim under test comes from upstream code:

* `src.mcts.tree_search.ToolTreeSearch` -- the official search loop,
* `src.config.load_config`             -- the official YAML config loader,
* `src.mcts.pruning.BidirectionalPruner` -- the official pruning rule,
* `src.tools.*`                        -- the official tool manager/registry,
* `src.evaluation.metrics`             -- the official F1 metrics.

Only the LLM boundary is replaced by deterministic stubs (adapted from the
upstream `tests/test_tier1_fixes.py`). No upstream source file is modified
anywhere in this study. The single experimental change is a config value
(`tau_post`), see `repro/configs/`.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from typing import Any

from src.config import MCTSConfig, load_config
from src.evaluation.metrics import plan_f1, tool_f1
from src.mcts.tree_search import ToolTreeSearch
from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolRegistry

# std-dev of the Gaussian jitter added to each tool's pre/post score. Small
# enough to keep the score landscape intact, large enough to make the "trap"
# tasks genuinely stochastic and to give non-zero variance across seeds.
JITTER_SIGMA = 0.03

AUDIT_LOGGERS = ("src.mcts.tree_search", "src.mcts.node")


class AuditCounter(logging.Handler):
    """Capture the upstream `AUDIT ...` log lines as measured counters."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        checks = {
            "pre_pruned": "AUDIT pre_prune tool=" in msg,
            "topk_dropped": "AUDIT topk_drop tool=" in msg,
            "expand_kept": "AUDIT expand_keep tool=" in msg,
            "post_pruned": "post_pruned=True" in msg,
            "schema_gated": "AUDIT schema_gate" in msg,
            "dedup_prior": "AUDIT dedup_prior_action" in msg,
            "dedup_child": "AUDIT dedup_existing_child" in msg,
            "search_exhausted": "AUDIT search_exhausted" in msg,
            "early_stop": "AUDIT early_stop triggered" in msg,
        }
        for key, hit in checks.items():
            if hit:
                self.counts[key] += 1


class StubJudge:
    """Deterministic stand-in for `src.evaluation.judge.LLMJudge`.

    Scores are fixed per tool name, so repeated evaluations of the same
    action are consistent (matching the repo's content-keyed judge
    memoization). A zero-mean Gaussian jitter is applied once per run.
    """

    def __init__(self, pre_scores: dict, post_scores: dict) -> None:
        self.pre_scores = pre_scores
        self.post_scores = post_scores
        self.pre_calls: list[dict] = []
        self.post_calls: list[dict] = []

    def pre_evaluate(self, query, context, tool_card, arg_draft):
        name = tool_card.get("name")
        self.pre_calls.append({"tool": name})
        return self.pre_scores.get(name, 0.5), "stub"

    def post_evaluate(self, query, context_before, tool_card, args_used, tool_output):
        name = tool_card.get("name")
        self.post_calls.append({"tool": name})
        return self.post_scores.get(name, 0.5), "stub"


class StubPlanner:
    """Deterministic argument drafts; identical to the upstream test stub."""

    def __init__(self) -> None:
        self.json_calls = 0

    def generate_json(self, messages):
        self.json_calls += 1
        return {"query": "draft"}

    def generate(self, messages, **kwargs):
        return "{}"


def _jitter(scores: dict, rng: random.Random) -> dict[str, float]:
    return {
        name: min(1.0, max(0.0, value + rng.gauss(0.0, JITTER_SIGMA)))
        for name, value in scores.items()
    }


def _count_nodes(root) -> int:
    stack, total = [root], 0
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def _post_pruned_nodes(root) -> int:
    stack, total = [root], 0
    while stack:
        node = stack.pop()
        if node.r_post is not None and not node.is_expandable:
            total += 1
        stack.extend(node.children)
    return total


def run_one(
    task: dict[str, Any],
    config_path: str,
    seed: int,
    condition: str,
    task_index: int,
    cfg_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one task and return a flat measurement row.

    The seed is paired across conditions: `condition` is deliberately NOT
    mixed into the RNG seed, so baseline and ablation see bit-identical
    score noise and UCT tie-break jitter. The only difference is tau_post.

    `cfg_override` applies equally to BOTH conditions; it is used for the
    sensitivity protocol (shipped early-stop default) and is never used to
    introduce the experimental modification.
    """
    cfg: MCTSConfig = load_config(config_path).mcts
    if cfg_override:
        for key, value in cfg_override.items():
            if not hasattr(cfg, key):
                raise KeyError(f"Unknown MCTSConfig field: {key}")
            setattr(cfg, key, value)
    run_seed = seed * 1_000_003 + task_index * 7919 + 17
    rng = random.Random(run_seed)
    random.seed(run_seed)  # UCT tie-break jitter inside MCTSNode.best_child

    judge = StubJudge(
        _jitter(task["pre_scores"], rng),
        _jitter(task["post_scores"], rng),
    )
    planner = StubPlanner()
    manager = ToolManager(ToolRegistry())
    search = ToolTreeSearch(cfg, manager, judge, planner)

    audit = AuditCounter()
    prev_levels: dict[str, int] = {}
    for name in AUDIT_LOGGERS:
        lg = logging.getLogger(name)
        prev_levels[name] = lg.level
        lg.setLevel(logging.INFO)
        lg.addHandler(audit)

    try:
        t0 = time.perf_counter()
        trajectory, best_q = search.search(task["query"], {}, task["tools"])
        wall_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        for name in AUDIT_LOGGERS:
            lg = logging.getLogger(name)
            lg.removeHandler(audit)
            lg.setLevel(prev_levels[name])

    root = search._last_root
    predicted_tools = [step.get("action") for step in trajectory]
    predicted_plan = [
        {"tool": step.get("action"), "args": step.get("args") or {}}
        for step in trajectory
    ]
    gold_tools = task["gold_tools"]
    nodes = _count_nodes(root)

    # Upstream grades plans with multiset F1 (src/evaluation/metrics.py),
    # which is deliberately order-insensitive. Our primary success indicator
    # follows that convention. `order_exact` is kept as a stricter,
    # transparent secondary statistic (the MCTS tie-break among unvisited
    # children is random, so tool order is not a stable target).
    from collections import Counter as _Counter

    plan_success = float(_Counter(predicted_tools) == _Counter(gold_tools))

    return {
        "condition": condition,
        "seed": seed,
        "task_id": task["task_id"],
        "family": task["family"],
        "tau_pre": cfg.tau_pre,
        "tau_post": cfg.tau_post,
        "early_stop_patience": cfg.early_stop_patience,
        "plan_success": plan_success,
        "order_exact": float(predicted_tools == gold_tools),
        "tool_f1": tool_f1(predicted_tools, gold_tools),
        "plan_f1": plan_f1(predicted_plan, task["gold_plan"]),
        "gold_recall": (
            len(set(gold_tools) & set(predicted_tools)) / len(gold_tools)
        ),
        "plan_length": len(predicted_tools),
        "best_q": round(float(best_q), 6),
        "pre_eval_calls": len(judge.pre_calls),
        "post_eval_calls": len(judge.post_calls),
        "planner_calls": planner.json_calls,
        "rollouts_used": search._rollouts_used,
        "nodes_created": nodes,
        "post_pruned_nodes": _post_pruned_nodes(root),
        "pre_pruned_log": audit.counts["pre_pruned"],
        "topk_dropped_log": audit.counts["topk_dropped"],
        "post_pruned_log": audit.counts["post_pruned"],
        "schema_gated_log": audit.counts["schema_gated"],
        "dedup_log": audit.counts["dedup_prior"] + audit.counts["dedup_child"],
        "search_exhausted": audit.counts["search_exhausted"],
        "early_stop": audit.counts["early_stop"],
        "wall_ms": round(wall_ms, 3),
        "predicted_plan": " > ".join(str(t) for t in predicted_tools),
        "gold_plan": " > ".join(gold_tools),
    }
