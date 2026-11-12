### Sensitivity protocol (patience=10)

Each cell is **mean +/- s.d. across 5 paired seeds**; each seed averages over 24 synthetic tasks. `Delta` = ablation - baseline.

| Metric | Baseline (post-pruning ON) | Ablation (post-pruning OFF) | Delta | Delta % |
|---|---:|---:|---:|---:|
| Plan success (exact gold tool set) | 0.675 +/- 0.017 | 0.658 +/- 0.055 | -0.017 | -2.5% |
| Tool F1 (repo metric) | 0.935 +/- 0.003 | 0.904 +/- 0.028 | -0.031 | -3.3% |
| Plan F1 (repo metric) | 0.935 +/- 0.003 | 0.904 +/- 0.028 | -0.031 | -3.3% |
| Gold-tool recall | 0.917 +/- 0.000 | 0.871 +/- 0.028 | -0.046 | -5.0% |
| Returned plan length | 2.075 +/- 0.017 | 2.000 +/- 0.000 | -0.075 | -3.6% |
| Pre-evaluation calls | 16.525 +/- 0.180 | 17.358 +/- 0.174 | +0.833 | +5.0% |
| Post-evaluation calls | 11.900 +/- 0.208 | 11.783 +/- 0.180 | -0.117 | -1.0% |
| Planner (arg-draft) calls | 19.258 +/- 0.277 | 20.017 +/- 0.228 | +0.758 | +3.9% |
| Rollouts used | 11.900 +/- 0.208 | 11.783 +/- 0.180 | -0.117 | -1.0% |
| Search-tree nodes | 16.867 +/- 0.135 | 17.700 +/- 0.110 | +0.833 | +4.9% |
| Post-pruned nodes | 7.133 +/- 0.232 | 0.000 +/- 0.000 | -7.133 | -100.0% |
| Wall time (ms) | 2.096 +/- 0.100 | 2.223 +/- 0.126 | +0.128 | +6.1% |
| Best Q value | 0.812 +/- 0.012 | 0.874 +/- 0.012 | +0.062 | +7.7% |

