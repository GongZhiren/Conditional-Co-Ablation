"""Freeze LayerNorm scale to its clean value, to separate the two self-repair mechanisms.

Ablating a head shrinks the residual stream, so every downstream LayerNorm divides by a
smaller denominator and *uniformly* amplifies whatever is left. That amplification is one of
the documented self-repair mechanisms (LayerNorm rescaling, Rushing & Nanda 2024) and it is
NOT a backup head taking over -- yet it inflates the conditional ablation energy of every
remaining unit, so it is a confound for any conditional score including CoAx.

This module removes it. We record each norm's denominator on the clean forward, then replay
the ablated forwards with that denominator held fixed. What survives is the part of the
conditional growth that comes from components actually re-routing, not from renormalization.

Modes:
  "scale" : freeze the denominator only (the rescaling channel; the default and the one that
            matches the mechanism named in the literature).
  "full"  : freeze the centering mean as well (a stricter, more surgical variant).

Architecture coverage: nn.LayerNorm (GPT-2, GPT-Neo, Pythia) and RMSNorm-style norms
(Llama, Qwen, Gemma, OLMo), detected by class name.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


def _is_rms(mod: nn.Module) -> bool:
    return "rms" in type(mod).__name__.lower()


def _norm_eps(mod: nn.Module) -> float:
    for attr in ("eps", "variance_epsilon"):
        v = getattr(mod, attr, None)
        if v is not None:
            return float(v)
    return 1e-5


def collect_norms(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Every normalization layer whose denominator responds to the residual-stream norm."""
    out: List[Tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm) or _is_rms(mod):
            out.append((name, mod))
    return out


def _stats(x: torch.Tensor, mod: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """(mean, denominator) exactly as the module would compute them, in float32."""
    xf = x.float()
    eps = _norm_eps(mod)
    if _is_rms(mod):
        mean = torch.zeros_like(xf[..., :1])
        denom = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    else:
        mean = xf.mean(-1, keepdim=True)
        denom = torch.sqrt(xf.var(-1, unbiased=False, keepdim=True) + eps)
    return mean, denom


class LayerNormFreezer:
    """Records clean per-norm statistics per calibration bucket, and replays them.

    Usage:
        fz = LayerNormFreezer(model, mode="scale")
        fz.record(bucket_key, lambda: forward(ids))     # clean pass, once per bucket
        handles = fz.hooks(bucket_key)                  # then any ablated pass is frozen
        ...
        for h in handles: h.remove()
    """

    def __init__(self, model: nn.Module, mode: str = "scale"):
        if mode not in ("scale", "full"):
            raise ValueError(f"mode must be 'scale' or 'full', got {mode!r}")
        self.mode = mode
        self.norms = collect_norms(model)
        if not self.norms:
            raise RuntimeError("no LayerNorm/RMSNorm modules found; cannot freeze")
        self.cache: Dict[object, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    # ------------------------------------------------------------------ recording
    def record(self, key: object, forward_fn: Callable[[], object]) -> None:
        """Run one clean forward and cache (mean, denom) for every norm."""
        store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        handles = []

        def mk(name: str, mod: nn.Module):
            def hook(_m, inputs):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    mean, denom = _stats(inputs[0].detach(), mod)
                    store[name] = (mean, denom)
            return hook

        for name, mod in self.norms:
            handles.append(mod.register_forward_pre_hook(mk(name, mod)))
        try:
            with torch.no_grad():
                forward_fn()
        finally:
            for h in handles:
                h.remove()
        self.cache[key] = store

    # -------------------------------------------------------------------- replay
    def hooks(self, key: object) -> List[torch.utils.hooks.RemovableHandle]:
        """Forward hooks that recompute each norm's output with the cached denominator.

        Returns handles the caller must remove. If a norm was not recorded (shape mismatch,
        skipped module) it is left untouched, so the pass degrades to the model's own norm
        rather than silently producing garbage.
        """
        store = self.cache.get(key)
        if store is None:
            raise KeyError(f"no clean statistics recorded for bucket {key!r}")
        handles = []

        def mk(name: str, mod: nn.Module):
            def hook(_m, inputs, output):
                cached = store.get(name)
                if cached is None or not inputs or not isinstance(inputs[0], torch.Tensor):
                    return output
                mean_c, denom_c = cached
                x = inputs[0]
                if x.shape[:-1] != denom_c.shape[:-1]:
                    return output  # different calibration bucket: leave the module alone
                xf = x.float()
                if self.mode == "full":
                    centered = xf - mean_c
                elif _is_rms(mod):
                    centered = xf
                else:
                    centered = xf - xf.mean(-1, keepdim=True)
                y = centered / denom_c
                w = getattr(mod, "weight", None)
                b = getattr(mod, "bias", None)
                if w is not None:
                    y = y * w.float()
                if b is not None:
                    y = y + b.float()
                return y.to(output.dtype if isinstance(output, torch.Tensor) else x.dtype)
            return hook

        for name, mod in self.norms:
            handles.append(mod.register_forward_hook(mk(name, mod)))
        return handles

    # ---------------------------------------------------------------- self-check
    def verify(self, key: object, forward_fn: Callable[[], torch.Tensor],
               tol: float = 1e-3) -> float:
        """Replaying the clean statistics on a CLEAN forward must reproduce the clean output.

        This validates the reimplementation of the norm (eps, weight/bias, RMS vs LayerNorm,
        dtype) before any frozen number is trusted. Returns the max absolute logit deviation
        and raises if it exceeds `tol`.
        """
        with torch.no_grad():
            ref = forward_fn().float()
            handles = self.hooks(key)
            try:
                got = forward_fn().float()
            finally:
                for h in handles:
                    h.remove()
        dev = float((ref - got).abs().max().item())
        if dev > tol:
            raise RuntimeError(
                f"LayerNorm freeze self-check failed: max |Δlogit| = {dev:.3e} > {tol:.0e}. "
                "The frozen-norm reimplementation does not match the model's own norm.")
        return dev
