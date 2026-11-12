"""Deterministic synthetic tool-planning tasks for the offline reproduction.

Why synthetic tasks?
--------------------
The upstream ToolTree release ships NO benchmark data (see the "Benchmarks"
section of the upstream README): GTA / m&m / ToolBench / RestBench must be
downloaded separately, and all of them need a live LLM endpoint for the
planner, the dual judge and the answer predictor. Reproducing a headline
benchmark number therefore requires (a) a hosted LLM key, (b) multi-GB
datasets, and (c) hundreds of LLM calls -- outside the laptop/time budget of
this assignment.

Section 2 of the assignment explicitly allows reproducing ONE small, clearly
defined task/experiment, and the eligibility checklist allows "a clear claimed
capability with an evaluation script" as the reproduction target. We therefore
reproduce the *mechanism-level* claim that ToolTree's bidirectional pruning
concentrates the search budget on promising tool chains. To measure it we need
a controlled environment in which the gold tool chain and the judge scores are
known exactly. This module builds such an environment: a small closed-set of
tool-planning tasks with explicit gold chains and fixed pre-/post-evaluation
scores.

The tasks are NOT claimed to represent GTA or any real benchmark. They are a
unit-test-style instrument for measuring search cost and plan quality under a
controlled, fully specified score landscape. All randomness is seeded, so the
entire study is bit-for-bit reproducible on any machine.

The stub judge / stub planner used to drive the search are adapted from the
upstream repository's own `tests/test_tier1_fixes.py` ("Pure stubs -- no LLM
server needed"). We use the SAME MCTS implementation, the SAME configuration
loader, the SAME pruning code and the SAME metric functions as upstream; only
the LLM boundary is replaced by deterministic stubs. This is documented as a
controlled deviation in the report (Section 3, "Setup notes").
"""

from __future__ import annotations

import random
from typing import Any

# A pool of plausible tool names, modelled on the closed-set GTA tool cards
# described in the upstream README (14 executable tools, e.g. OCR, Calculator).
TOOL_POOL: tuple[str, ...] = (
    "OCR",
    "Calculator",
    "WebSearch",
    "WeatherLookup",
    "Translator",
    "ImageCaption",
    "SQLQuery",
    "MapRoute",
    "Calendar",
    "CurrencyConvert",
)

# Draft arguments produced by the deterministic stub planner. The upstream
# StubPlanner returns {"query": "draft"} for every tool, so gold plans use the
# same shape; this keeps the repo's plan_f1 / argument_f1 comparable.
STUB_ARGS: dict[str, str] = {"query": "draft"}

FAMILIES: tuple[str, ...] = ("clean", "trap", "noisy", "deep")


def _tool_card(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"The {name} tool.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "output_schema": {"type": "object"},
    }


def _build_family(family: str, variant: int) -> dict[str, Any]:
    """Build one deterministic task. `family` fixes the difficulty regime."""
    # NOTE: do NOT use Python's hash() here -- string hashing is randomized
    # per process (PYTHONHASHSEED), which would break bit-for-bit
    # reproducibility. FAMILIES.index() gives a stable process-independent id.
    rng = random.Random(20261113 + FAMILIES.index(family) * 1000 + variant)
    pool = list(TOOL_POOL)
    rng.shuffle(pool)

    n_gold = 3 if family == "deep" else 2
    n_distract = 4 if family == "noisy" else 3
    gold = pool[:n_gold]
    distractors = pool[n_gold : n_gold + n_distract]

    pre_scores: dict[str, float] = {}
    post_scores: dict[str, float] = {}

    # Gold chain: high pre-evaluation (survives tau_pre=0.3) and high
    # post-evaluation (survives tau_post=0.4).
    for j, name in enumerate(gold):
        pre_scores[name] = round(0.92 - 0.05 * j, 3)
        post_scores[name] = round(0.90 - 0.05 * j, 3)

    # Distractors: admissible (r_pre >= tau_pre) but unproductive after
    # execution (r_post < tau_post), i.e. exactly the branches that
    # post-evaluation pruning is designed to eliminate.
    for j, name in enumerate(distractors):
        if family == "noisy":
            pre = rng.uniform(0.31, 0.42)
        else:
            pre = rng.uniform(0.35, 0.72)
        post = rng.uniform(0.05, 0.30)
        pre_scores[name] = round(pre, 3)
        post_scores[name] = round(post, 3)

    if family == "trap":
        # A "high-prior trap": the most attractive candidate before execution
        # (r_pre=0.96) but only marginally below tau_post. With jitter it can
        # cross the threshold, making the task genuinely stochastic.
        trap = distractors[0]
        pre_scores[trap] = 0.96
        post_scores[trap] = 0.36

    tools = [_tool_card(n) for n in gold + distractors]
    # The order of the tool list does not change the search (upstream _expand
    # sorts surviving candidates by r_pre), but shuffling keeps the instrument
    # free of any accidental order-dependent artifacts.
    rng.shuffle(tools)

    return {
        "task_id": f"{family}-{variant:02d}",
        "family": family,
        "query": (
            f"Complete a {n_gold}-step tool plan ending with "
            f"{gold[-1]} using the available tools."
        ),
        "tools": tools,
        "gold_tools": list(gold),
        "gold_plan": [{"tool": n, "args": dict(STUB_ARGS)} for n in gold],
        "pre_scores": pre_scores,
        "post_scores": post_scores,
    }


def generate_tasks(per_family: int = 6) -> list[dict[str, Any]]:
    """Return `4 * per_family` fully specified, deterministic tasks."""
    if per_family < 1:
        raise ValueError("per_family must be >= 1")
    return [
        _build_family(family, variant)
        for family in FAMILIES
        for variant in range(per_family)
    ]
