"""Baselines for backup discovery, on the same frozen model and IOI metric as CoAx.

All of these rank attention heads by a *first-order* (node-additive) signal, and all of them
miss the dormant backups to varying degrees (Table 1 of the paper):

* ``coactivation_scores``    input-side control: correlation of head-output norm with the
                             primaries (forward-only; ranks the backups high but conflates
                             roles, so it over-ablates downstream).
* ``atp_scores``             attribution patching, |grad x activation| (Syed et al. 2023).
* ``graddrop_scores``        AtP* with GradDrop (Kramar et al. 2024): L attributions, each
                             cutting one block's indirect gradient path (the self-repair fix).
* ``gim_conditional_scores`` GIM-style: gradient x activation on the *primary-ablated* model
                             (Edin et al. 2025) --- the fair, same-seed gradient comparison.

The gradient methods need a differentiable forward, so they bypass the frozen no-grad path.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------- gradient plumbing
def _ioi_metric(model, ex: Dict[str, str]) -> torch.Tensor:
    ids = model.tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].to(model.device)
    io = model.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
    s = model.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
    logits = model.model(input_ids=ids, use_cache=False).logits[0, -1, :]
    return logits[io] - logits[s]


def _capture_hooks(model):
    """Pre-hooks on every attention output projection, storing its (grad-retaining) input
    (the concatenated per-head output z)."""
    store: Dict[int, torch.Tensor] = {}
    handles = []

    def mk(li):
        def hook(_m, inp):
            x = inp[0]
            if x.dim() == 3 and x.requires_grad:
                x.retain_grad()
                store[li] = x
        return hook

    for li in range(model.num_layers):
        handles.append(model._out_proj(li).register_forward_pre_hook(mk(li)))
    return handles, store


def _grad_times_act(model, store) -> np.ndarray:
    """|z . dz| summed per head from the current backward pass."""
    nH, hd = model.num_heads, model.head_dim
    score = np.zeros(model.num_units)
    for li, x in store.items():
        if x.grad is None:
            continue
        attr = (x * x.grad).detach()[0]                     # [T, nH*hd]
        for hi in range(nH):
            score[model.head_index(li, hi)] += float(attr[:, hi * hd:(hi + 1) * hd].sum())
    return score


# --------------------------------------------------------------- baselines
def atp_scores(model, examples: Sequence[Dict[str, str]]) -> np.ndarray:
    """Attribution patching: mean over prompts of |grad x activation| per head."""
    score = np.zeros(model.num_units)
    for ex in examples:
        handles, store = _capture_hooks(model)
        try:
            model.model.zero_grad(set_to_none=True)
            _ioi_metric(model, ex).backward()
            score += _grad_times_act(model, store)
        finally:
            for h in handles:
                h.remove()
    return np.abs(score / max(1, len(examples)))


def graddrop_scores(model, examples: Sequence[Dict[str, str]]) -> np.ndarray:
    """AtP* GradDrop: average |grad x activation| over L runs, each zeroing the gradient
    through one block's output (cutting its indirect path --- the self-repair cancellation)."""
    L = model.num_layers
    acc = np.zeros(model.num_units)
    for drop_l in range(L):
        per_drop = np.zeros(model.num_units)
        for ex in examples:
            handles, store = _capture_hooks(model)
            def drop_hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(h) and h.requires_grad:
                    h.register_hook(lambda g: torch.zeros_like(g))
                return out
            blk = model._layers[drop_l].register_forward_hook(drop_hook)
            try:
                model.model.zero_grad(set_to_none=True)
                _ioi_metric(model, ex).backward()
                per_drop += _grad_times_act(model, store)
            finally:
                for h in handles:
                    h.remove()
                blk.remove()
        acc += np.abs(per_drop / max(1, len(examples)))
    return acc / max(1, L)


def gim_conditional_scores(model, examples: Sequence[Dict[str, str]],
                           seed: Sequence[Tuple[int, int]]) -> np.ndarray:
    """GIM-style, same-seed gradient baseline: |grad x activation| measured on the
    primary-ablated model (so the dormant backups' gradient is no longer muted)."""
    seed_units = {model.head_index(l, h) for (l, h) in seed}
    score = np.zeros(model.num_units)
    for ex in examples:
        abl = model._ablation_hooks(list(seed))               # zero the primaries in-graph
        handles, store = _capture_hooks(model)
        try:
            model.model.zero_grad(set_to_none=True)
            _ioi_metric(model, ex).backward()
            score += _grad_times_act(model, store)
        finally:
            for h in handles:
                h.remove()
            for h in abl:
                h.remove()
    for u in seed_units:
        score[u] = 0.0
    return np.abs(score / max(1, len(examples)))


def coactivation_scores(model, prompts: Sequence[str],
                        primaries: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Input-side control: rank each head by the mean |correlation| of its attention-output
    norm with the primary heads' norms, across all calibration positions (forward-only)."""
    nH, hd = model.num_heads, model.head_dim
    norms: Dict[int, List[np.ndarray]] = {u: [] for u in range(model.num_units)}
    handles = []

    def mk(li):
        def hook(_m, inp):
            x = inp[0]
            if x.dim() != 3:
                return
            for hi in range(nH):
                sl = x[..., hi * hd:(hi + 1) * hd].norm(dim=-1).reshape(-1).float().cpu().numpy()
                norms[model.head_index(li, hi)].append(sl)
        return hook

    for li in range(model.num_layers):
        handles.append(model._out_proj(li).register_forward_pre_hook(mk(li)))
    try:
        for p in prompts:
            ids = model.tokenizer(p, return_tensors="pt")["input_ids"]
            model.logits(ids)
    finally:
        for h in handles:
            h.remove()

    mat = np.stack([np.concatenate(norms[u]) for u in range(model.num_units)], axis=0)
    corr = np.nan_to_num(np.corrcoef(mat))
    prim = [model.head_index(l, h) for (l, h) in primaries]
    return np.abs(corr[:, prim]).mean(axis=1)
