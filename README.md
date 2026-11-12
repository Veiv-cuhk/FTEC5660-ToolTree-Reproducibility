# Reproducibility Study: ToolTree (FTEC5660)

**Author:** JIANG Jiawei (Student ID 1155271183)
**Course:** Agentic AI for Business and FinTech (FTEC5660) — Reproducibility Work
**Due:** 13 November 2026
**Upstream project:** [SYang2000/ICLR_2026_ToolTree](https://github.com/SYang2000/ICLR_2026_ToolTree), imported at commit `fed7a608ca54ba2e75078f9a1f409c9aa7fab18f` (2026-07-05)
**Paper:** Yang et al., *ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning*, ICLR 2026 ([arXiv:2603.12740](https://arxiv.org/abs/2603.12740))

---

## 1. What this repository is

This is a fork of the official ToolTree implementation plus a **small, isolated,
fully offline reproducibility study**. No upstream source file is modified.

Only what the assignment asks for is kept here: the runnable upstream snapshot,
the reproducibility harness with its measured outputs, this README, and the
report. Slides, speaker notes, recording notes and demo-site assets are **not**
part of the submission; the recorded presentation is hosted as a public Release
asset (Section 8).

ToolTree is an agentic system: given a query and a set of tools, it plans a
multi-step tool trajectory with a Monte Carlo Tree Search loop, scoring candidate
tool calls before execution (*pre-evaluation*) and after execution
(*post-evaluation*), and pruning unpromising branches in both directions. It is
item **No. 17 (Tier A)** of the instructor-provided paper list.

**Target claim selected for reproduction.** The upstream README claims that
"bidirectional pruning ... eliminat[es] unpromising branches, concentrating
computational budget on promising tool chains", and the release ships a
32-test behaviour suite with explicit numeric assertions about search
termination and pruning. We reproduce (i) those pinned numeric behaviours and
(ii) the budget-concentration mechanism, measured on a controlled task set.

**What is deliberately NOT reproduced.** The paper's headline result
("~10% average gain across 4 benchmarks") and Figure 3's efficiency numbers
require the GTA / m&m / ToolBench / RestBench datasets, a live LLM endpoint for
planner + dual judge + answer predictor, and hundreds of LLM calls. The
upstream repo ships no benchmark data. Section 2 of the assignment explicitly
permits reproducing one small, clearly defined experiment, so those headline
numbers are out of scope and are reported as such in the report.

---

## 2. Install

Requirements: Python 3.10+ (developed and verified on Python 3.13.14, Windows),
git. **No LLM API key, no GPU, no dataset download and no network access is
needed at run time.**

```bash
git clone https://github.com/Veiv-cuhk/FTEC5660-ToolTree-Reproducibility.git
cd FTEC5660-ToolTree-Reproducibility
python -m pip install -r requirements.txt -r repro/requirements.txt
```

`requirements.txt` is upstream (openai, pyyaml, numpy); `repro/requirements.txt`
adds `pytest` and `matplotlib` for the behaviour suite and the figures.

---

## 3. Run / reproduce every number

All commands are executed **from the repository root**.

```bash
# (A) Reproduce the upstream reported/pinned behaviours (32 tests + 3 numeric claims)
python -m pytest tests/ -q
python -m repro.verify_claims

# (B) Run the paired main experiment: 24 tasks x 5 seeds x 2 conditions = 240 searches
python -m repro.run_experiment --seeds 5 --per-family 6

# (C) Sensitivity protocol: shipped early-stop default in BOTH arms
python -m repro.run_experiment --seeds 5 --per-family 6 \
    --early-stop-patience 10 --out repro/results/sensitivity_defaultstop

# (D) Rebuild every table and figure from the measured rows
python -m repro.summarize

# (E) Verify the experiment is bit-for-bit reproducible
python -m repro.check_reproducibility
```

Outputs are written to `repro/results/`:

| File | Contents |
|---|---|
| `upstream_tests.txt` | verbatim transcript of the shipped 32-test suite |
| `claim_verification.json` | reported vs measured values for 4 pinned claims |
| `raw_runs.csv` | 240 rows, one per task x seed x condition (all raw measurements) |
| `run_manifest.json` | Python/platform versions, upstream commit, configs, seed design |
| `summary_by_seed.csv` | per-condition, per-seed means |
| `summary_overall.json` | mean +/- s.d. across seeds and paired deltas |
| `table_results.md` | report-ready Markdown tables (generated, not hand-typed) |
| `figure_cost_quality.png`, `figure_by_family.png` | report figures |
| `determinism_check.json` | independent rerun comparison |
| `sensitivity_defaultstop/` | the same outputs for the sensitivity protocol |

---

## 4. Methodology in one paragraph

The upstream release ships no benchmark data and needs a live LLM for three
separate roles. We therefore replace **only the LLM boundary** with the
deterministic stubs that the upstream repository itself uses in
`tests/test_tier1_fixes.py` ("Pure stubs -- no LLM server needed"), and drive a
small, fully specified synthetic closed-set of 24 tool-planning tasks with known
gold tool chains and known pre/post evaluation scores. The MCTS loop, UCT
selection, pruning rule, tool manager, config loader and F1 metrics are all
unmodified upstream code. Seeds are **paired**: baseline and ablation see
identical score jitter and tie-break randomness, so the only independent
variable is `tau_post`. Five seeds x 24 tasks give mean and variance. A
Gaussian jitter (sigma = 0.03) on judge scores makes the trials genuinely
stochastic. Full justification and limitations are in the report.

---

## 5. The modification (one isolated change)

`repro/configs/ablation_no_post_pruning.yaml` differs from
`repro/configs/baseline.yaml` in exactly one value:

```yaml
# baseline
mcts: { ..., tau_post: 0.4, ... }   # post-evaluation pruning ACTIVE
# ablation
mcts: { ..., tau_post: 0.0, ... }   # post-evaluation pruning DISABLED
```

Because `r_post` is always >= 0, setting `tau_post = 0.0` makes the condition
`r_post < tau_post` unreachable: the post-execution half of bidirectional
pruning is removed while pre-pruning, the MCTS loop, early stopping, tools,
judge, planner, tasks and seeds are held fixed. This is an **ablation of one
mechanism**, measurable in both plan quality and search cost.

`early_stop_patience: 100` appears in both configs. That is a **measurement
protocol choice**, not the modification: with the shipped default of 10 both
arms stop after ~12 rollouts and the pruning effect is masked. The
sensitivity run (command C) restores the shipped default in both arms and is
reported alongside the main result.

---

## 6. Headline measured results

Mean +/- s.d. across 5 paired seeds; each seed averages 24 tasks. Full tables:
`repro/results/table_results.md`.

| Metric | Baseline (pruning ON) | Ablation (pruning OFF) | Change |
|---|---:|---:|---:|
| Plan success (exact gold tool set) | 0.983 +/- 0.020 | 0.133 +/- 0.061 | -86.4% |
| Tool F1 (upstream metric) | 0.994 +/- 0.007 | 0.635 +/- 0.046 | -36.1% |
| Post-evaluation calls / run | 25.325 +/- 0.469 | 40.000 +/- 0.000 | +57.9% |
| Search-tree nodes / run | 27.408 +/- 0.564 | 52.425 +/- 0.533 | +91.3% |
| Wall time (ms) / run | 4.868 +/- 0.262 | 8.511 +/- 0.195 | +74.8% |

With pruning ON, 88/120 runs terminate by tree exhaustion; with pruning OFF,
0/120 do and **all 120 hit the 40-rollout budget cap**. Disabling
post-pruning therefore roughly doubles the search cost and collapses plan
quality — a direct confirmation of the repo's budget-concentration claim.

---

## 7. Commit map (small, reviewable changes)

The history is eight small, single-purpose commits rather than one bulk upload:

| # | Commit |
|---|---|
| 1 | import the unmodified upstream ToolTree snapshot @ `fed7a608` |
| 2 | deterministic synthetic task set + offline measurement harness |
| 3 | script that re-measures the upstream pinned numeric behaviours |
| 4 | baseline config, paired-seed runner, aggregation and determinism check |
| 5 | **the modification** -- `tau_post: 0.4 -> 0.0` in one YAML file, one value |
| 6 | measured outputs: raw CSV, summaries, tables, figures, test transcript |
| 7 | this README with install + run steps and the submission package |
| 8 | 8-page LaTeX report, its source, build script and figures |

The experimental change is isolated in commit 5 (one file, one value). No commit
mixes source code with unrelated artefacts. Only the files listed in Section 8
and the code needed to reproduce the numbers have ever been committed; no slide
deck, speaker script or Word document is present in the repository or its
history.

No secrets, API keys, `.env` files or tokens are committed. `.gitignore` excludes
`.env`, `*.key`, `*.token`, build caches and Python temporary files.

---

## 8. Submission package

| Item | Location |
|---|---|
| Brief report (8 pages, <= 10) | `report/FTEC5660_Reproducibility_Report_JIANG_Jiawei_1155271183.pdf` — typeset with LaTeX in the style of the assignment brief |
| GitHub link | `https://github.com/Veiv-cuhk/FTEC5660-ToolTree-Reproducibility` (public, small reviewable commits, no secrets; clone URL in the Install section) |
| Presentation link (9:13; limit 15) | Recorded Zoom presentation, presenter on camera, hosted as a public Release asset in this repository (no login required). Release page: `https://github.com/Veiv-cuhk/FTEC5660-ToolTree-Reproducibility/releases/tag/v1.0`. Direct MP4: `https://github.com/Veiv-cuhk/FTEC5660-ToolTree-Reproducibility/releases/download/v1.0/FTEC5660_Presentation_JIANG_Jiawei_1155271183.mp4` |

The report can be rebuilt from its LaTeX source with `bash report/build.sh` from
the repository root (needs `pdflatex`; TeX Live 2024 was used, run twice). LaTeX
auxiliary files are git-ignored.

---

## 9. Scope, honesty statement, and reproduction limits

* All numbers in `repro/results/` come from actual executions of the upstream
  code on this machine. Nothing is hand-entered or estimated.
* All logical measurements are bit-for-bit identical across independent
  reruns (`determinism_check.json`); only wall-clock timing varies.
* Results are **not directly comparable to the paper's benchmark numbers**:
  the underlying LLM is replaced by deterministic stubs and the tasks are
  synthetic. This is stated explicitly in the report, as the assignment
  requires for controlled-variable changes.
* The study isolates a mechanism, not a benchmark score. Conclusions about
  GTA / m&m accuracy cannot be drawn from it.

---

## 10. Citation and license

Upstream code is MIT licensed (see `LICENSE`). Please cite the original paper:

```bibtex
@inproceedings{yang2026tooltree,
  title={ToolTree: Efficient {LLM} Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning},
  author={Shuo Yang and Caren Han and Yihao Ding and Shuhe Wang and Eduard Hovy},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
