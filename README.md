<div align="center">

# Conditional Co-Ablation (CoAx)
### Recovering Self-Repair Backups in Transformer Circuits

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
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
misreads both sides of the redundancy — and the same blind spot is inherited by every tool built
on that primitive: attribution, capability knockout, and pruning.

**CoAx** asks a conditional question instead: *once the primary circuit is removed, how much does
each remaining unit's ablation effect grow?* A backup has little effect alone but a large
conditional effect; an irrelevant unit has neither.

<div align="center">

| signal | what it sees | backup ROC-AUC on GPT-2 IOI |
|:--|:--|:--:|
| single ablation (1st order) | effect of a unit **alone** | `0.33`  *(below chance — backups are dormant)* |
| **CoAx** (2nd order) | **growth** of that effect once the primaries are gone | **`0.91`** |

</div>

## The idea in one equation

Every effect is measured in the output distribution's own (Fisher) metric on the clean top-*r*
logits. Writing `E(δz_u | S)` for the mean energy of ablating unit `u` **after** a primary seed
`S` is ablated, the CoAx score is the *growth* of that energy under conditioning:

```
comp_u(S)  =  E(δz_u | S)  −  E(δz_u | ∅)
```

Large for a dormant backup (silent alone, load-bearing once its primary is gone), near zero
otherwise. It costs `O(#heads)` forward passes per seed — the same order as one single-ablation
scan, and no gradients or labels.

## Install

```bash
pip install -r requirements.txt   # torch, transformers, numpy, scikit-learn
```

## Reproduce the headline

```bash
python reproduce_ioi.py                 # GPT-2-small from the Hugging Face hub
python reproduce_ioi.py --num-prompts 96   # the paper's discovery setting
```

Expected output (reproduces on CPU in a few minutes; a GPU takes seconds):

```
============================================================
  Backup-name-mover recovery (8 documented backups, 141 candidates)
============================================================
  single ablation (1st order)   backup ROC-AUC = 0.33
  CoAx conditional (2nd order)  backup ROC-AUC = 0.91
============================================================
  CoAx top-k recall of documented backups: 3/8 @ top-8, 4/8 @ top-10, ...

  Top-10 heads by CoAx score (* = documented backup):
     1. * L11.H2   comp=+0.037
     2. * L10.H2   comp=+0.008
     ...
```

The labels (the eight documented backups) are used **only to evaluate** the ranking — the score
itself never sees them.

## Use CoAx on your own model and circuit

```python
from coax import Model, CoAx

model = Model("gpt2")                      # any HF decoder-only LM
coax  = CoAx(model, prompts, top_r=192)    # `prompts`: a list of task strings

# Condition on the primary heads your first-order analysis found, then read the growth:
scores = coax.conditional_compensation(seed=[(9, 9), (10, 0), (9, 6)])
backups = scores["compensation"]           # per-head CoAx score; rank it, high = backup

# The symmetric second-order interaction is available directly, too:
S = coax.pairwise_synergy(units=range(model.num_units))
```

`CoAx` is unit-agnostic; the primary seed can come from any first-order method (attribution
patching, EAP-IG, AtP\*, or a manual analysis). CoAx **completes** that circuit by returning the
backups it hides.

## Repository layout

```
coax/
  model.py      frozen HF model + per-head ablation via plain forward hooks
  scoring.py    Fisher-centered features; single energy, conditional CoAx score, pairwise synergy
  ioi.py        the GPT-2 IOI task + the documented circuit (Wang et al., 2022)
reproduce_ioi.py  end-to-end headline experiment
```

Datasets, model weights, and pre-computed results are intentionally **not** shipped: the script
downloads GPT-2 on first run and recomputes everything from scratch.

## Citation

```bibtex
@inproceedings{gong2026coax,
  title     = {Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits},
  author    = {Gong, Zhiren and Zeng, Zihao and Yuen, Chau and Lim, Wei Yang Bryan},
  year      = {2026},
  note      = {Project page: https://gongzhiren.github.io/Conditional-Co-Ablation-website}
}
```

## License

Released under the [MIT License](LICENSE).
