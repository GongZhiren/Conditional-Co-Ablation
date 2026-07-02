"""CoAx --- Conditional Co-Ablation for recovering self-repair backups in transformer circuits.

A label-free, output-grounded score that recovers the dormant *backup* components a circuit
falls back on under intervention --- the redundancy a first-order (node-additive) score is
provably blind to. See the paper for the method and results; see ``reproduce_ioi.py`` for the
headline GPT-2-small IOI experiment.
"""
from __future__ import annotations

from .model import Model
from .scoring import CoAx

__all__ = ["Model", "CoAx"]
__version__ = "1.0.0"
