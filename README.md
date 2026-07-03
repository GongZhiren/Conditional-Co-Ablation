<div align="center">

# Conditional Co-Ablation (CoAx)
### Recovering Self-Repair Backups in Transformer Circuits

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2607.01940)
[![Project page](https://img.shields.io/badge/project-page-1f6fb0.svg)](https://gongzhiren.github.io/Conditional-Co-Ablation-website)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)

*A label-free, output-grounded score that recovers the dormant **backup** components a circuit
falls back on under intervention --- the redundancy a first-order score is provably blind to.*

</div>

---

## Why

Mechanistic interpretability scores a component by the effect of ablating it **in isolation**.
That first-order view breaks under **self-repair**: when a primary component is removed, a
dormant *backup* takes over, so the primary looks unimportant (the model repaired the damage)
and the backup looks unimportant too (it is silent on the intact model). A single-ablation score
misreads both sides of the redundancy — and the same blind spot is inherited by attribution,
capability knockout, and pruning.

**CoAx** asks a conditional question instead: *once the primary circuit is removed, how much does
each remaining unit's ablation effect grow?* Writing `E(δz_u | S)` for the mean (Fisher-metric)
energy of ablating unit `u` **after** a primary seed `S` is ablated,

```
comp_u(S)  =  E(δz_u | S)  −  E(δz_u | ∅)
```

is large for a dormant backup (silent alone, load-bearing once its primary is gone) and near
zero otherwise. It is label-free, gradient-free, and costs `O(#heads)` forward passes.

## Install

```bash
pip install -r requirements.txt      # torch, transformers, numpy, scikit-learn
```

GPT-2 is downloaded from the Hugging Face hub on first run. Everything reproduces on CPU in a
few minutes; a single GPU takes seconds. Pass `--model <path>` to use a local checkpoint.

## Quick start

```bash
python reproduce_ioi.py              # the headline: single ablation 0.33 -> CoAx 0.91 backup ROC-AUC
```

## Reproducing the paper

Each script prints a table and maps to one result in the paper. Numbers below were reproduced on
GPT-2-small (96 prompts unless noted); multi-set experiments average over the paper's four seeds.

| script | paper result | key numbers (reproduced) |
|---|---|---|
| `experiments/backup_recovery.py` | Table 1 — backup ROC-AUC, all methods | single `0.33` · AtP `0.59` · GIM(seeded) `0.62` · AtP\* GradDrop `0.79` · **CoAx `0.91`** · co-activation `0.93` |
| `experiments/knockout.py` | Table 4 / Fig 1c — capability knockout | −prim `0.97` (self-repair) · **+CoAx `0.74`** ≈ oracle `0.72` · +own `0.38` (overshoots) · +random `0.91` |
| `experiments/attribution.py` | Table 3 — masked-attribution recovery | primaries `+0.23` · +random `+0.67` · +doc `+1.22` · **+CoAx `+1.08`** (≈5× the masked effect) |
| `experiments/mechanism.py` | Fig 3 — are they real backups? | wake-up norm `1.00→1.15`, answer contribution `+0.05→+0.11`; **freezing the backups removes 53%** of the self-repair (random ~0%) |
| `experiments/synthetic.py` | Table 2 / Prop. 2 — controlled redundancy | clean-state scores `0.42 / 0.51` (≤ chance) · **CoAx `0.92`** *(no model needed; runs in a second)* |

```bash
python experiments/backup_recovery.py     # Table 1  (add --fast to skip the gradient baselines)
python experiments/knockout.py            # Table 4
python experiments/attribution.py         # Table 3
python experiments/mechanism.py           # Fig 3  (wake-up + counterfactual patching)
python experiments/synthetic.py           # Table 2 (Proposition 2)
```

The labels (the eight documented backups) are used **only to evaluate** the rankings — the CoAx
score never sees them.

## Use CoAx on your own model and circuit

```python
from coax import Model, CoAx

model = Model("gpt2")                       # any HF decoder-only LM
coax  = CoAx(model, prompts, top_r=192)     # `prompts`: a list of task strings

# Condition on the primary heads a first-order analysis found, then read the growth:
scores  = coax.conditional_compensation(seed=[(9, 9), (10, 0), (9, 6)])
backups = scores["compensation"]            # per-head CoAx score; rank it, high = backup
ratios  = coax.wakeup_ratios(seed=[(9, 9), (10, 0), (9, 6)])   # the wake-up signature

# The symmetric second-order interaction is available directly, too:
S = coax.pairwise_synergy(units=range(model.num_units))
```

The primary seed can come from any first-order method (attribution patching, EAP-IG, AtP\*, or a
manual analysis); CoAx **completes** that circuit by returning the backups it hides.

## Repository layout

```
coax/
  model.py       frozen HF model + per-head ablation via plain forward hooks
  scoring.py     Fisher-centered features; single energy, conditional CoAx score, synergy, wake-up
  baselines.py   AtP, AtP* GradDrop, GIM (gradient) and co-activation (input-side control)
  ioi.py         the GPT-2 IOI task, the documented circuit (Wang et al. 2022), and task metrics
experiments/     one script per paper result (table above)
reproduce_ioi.py the headline backup-recovery run
```

Datasets, model weights, and pre-computed results are intentionally **not** shipped: everything is
recomputed from scratch. The cross-architecture **induction** generalization and the
**structured-pruning** sweeps in the paper use the same `coax` API across additional models and
WikiText-2; they are omitted here to keep the repo self-contained and single-model.

## Citation

```bibtex
@misc{gong2026coax,
  title         = {Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits},
  author        = {Gong, Zhiren and Zeng, Zihao and Yuen, Chau and Lim, Wei Yang Bryan},
  year          = {2026},
  eprint        = {2607.01940},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Project page: https://gongzhiren.github.io/Conditional-Co-Ablation-website}
}
```

## License

Released under the [MIT License](LICENSE).
