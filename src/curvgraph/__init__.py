"""curvgraph — Second-order co-ablation curvature for circuit discovery.

The package implements a second-order co-ablation method for
mechanistic circuit discovery: pairwise synergy and a conditional/greedy co-ablation signal
that recover the COMPENSATION and BACKUP/SELF-REPAIR
structure single-component ablation, co-activation clustering, and weight-subspace affinity
all miss. Fully self-contained: the calibration / ablation / model infrastructure it builds
on is bundled in ``curvgraph._core`` (no code outside this repository is required).

Core method code:
  coablation.py  the second-order method: 1st-order affinity + pairwise synergy I_uv +
                 conditional compensation comp_u (backup / self-repair discovery)
  circuits.py    GPT-2 circuit ground truth, single-ablation kernel, induction detection,
                 causal scrubbing, controls (co-activation, random); head-level plumbing
The public experiment entry points live under ``experiments/paper``.
"""
from __future__ import annotations

from .coablation import CoAblation
from ._core.config import load_config, model_config, validate_config
from ._core.model import ModelBundle, load_model_bundle

__version__ = "2.0.0"
__all__ = [
    "CoAblation",
    "ModelBundle",
    "load_config",
    "load_model_bundle",
    "model_config",
    "validate_config",
]
