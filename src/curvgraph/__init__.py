"""curvgraph — Conditional co-ablation for intervention-conditioned circuit completion.

The package implements CoAx, whose two-state effect change aggregates all interaction orders
linking a candidate to a supplied primary set. Its signed-growth score identifies compensation
and backup/self-repair structure that intact-state single-component scores can miss. Pairwise
synergy remains available as an auxiliary diagnostic. The calibration, ablation, and model
infrastructure is bundled in ``curvgraph._core``.

Core method code:
  coablation.py  conditional signed growth plus auxiliary pairwise synergy
  circuits.py    GPT-2 circuit ground truth, single-ablation kernel, induction detection,
                 causal scrubbing, controls (co-activation, random); head-level plumbing
The public experiment entry points live under ``experiments/paper``.
"""
from __future__ import annotations

from .coablation import CoAblation
from ._core.config import load_config, model_config, validate_config
from ._core.model import ModelBundle, load_model_bundle

__version__ = "2.1.0"
__all__ = [
    "CoAblation",
    "ModelBundle",
    "load_config",
    "load_model_bundle",
    "model_config",
    "validate_config",
]
