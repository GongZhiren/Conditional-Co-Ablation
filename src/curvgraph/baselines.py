"""First-order circuit-attribution baselines + faithfulness utilities.

These are the field-standard comparisons our second-order co-ablation method must beat: every
published circuit-discovery method (ACDC, attribution/edge-attribution patching, AtP*) is
first-order -- it scores a head by its single-component (gradient or ablation) effect and
selects greedily, so it structurally misses redundant / backup heads whose solo effect is
muted by self-repair. We implement two:

  * attribution_patching (ATP): grad of a behavioral metric w.r.t. a head's output, times the
    output -- the first-order Taylor estimate of zero-ablating that head. (Nanda 2023; the
    node-level form behind EAP / AtP*.)
  * integrated_gradient_attribution (EAP-IG-style): the same, path-integrated from a corrupt
    baseline to the clean input, which is the more faithful variant (Hanna et al. 2024).

Plus faithfulness_curve: normalized logit-diff recovered vs. circuit size, the metric the
field (MIB, Wang 2022) actually reports -- not just same-circuit AUC.

Kept separate from coablation.py so the novel method stays clean.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from contextlib import contextmanager

import numpy as np
import torch

from curvgraph._core.model import ModelBundle, bundle_device, layer_modules
from . import circuits as C


def projection_kernel_affinity(bundle: ModelBundle, head_set) -> np.ndarray:
    """Weight-subspace head affinity (Yamagiwa et al. 2026, Projection Kernel): the normalized
    projection metric between the per-head VALUE subspaces. For heads a,b with orthonormal bases
    U_a,U_b of their value column-spaces, affinity = ||U_a^T U_b||_F^2 / head_dim in [0,1]. This is
    the input/weight-side control our output-side co-ablation must beat (GPT-2 c_attn layout)."""
    nH, hd = bundle.num_heads, bundle.head_dim
    layers = layer_modules(bundle)
    bases = {}
    for u in head_set:
        li, hi = C.head_layer_head(int(u), nH)
        attn = C._attn_module(layers[li])
        cattn = getattr(attn, "c_attn", None)
        if cattn is None or not hasattr(cattn, "weight"):
            return np.eye(len(head_set))                       # arch without GPT-2 c_attn: skip
        W = cattn.weight.detach().float()                      # Conv1D: [d_model, 3*d_model]
        d = W.shape[0]
        Wv = W[:, 2 * d:]                                       # value block [d_model, d_model]
        Wv_h = Wv[:, hi * hd:(hi + 1) * hd]                    # [d_model, head_dim]
        q, _ = torch.linalg.qr(Wv_h)                           # orthonormal basis [d_model, head_dim]
        bases[int(u)] = q
    m = len(head_set)
    A = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            Ua, Ub = bases[int(head_set[i])], bases[int(head_set[j])]
            val = float((Ua.T @ Ub).pow(2).sum().item()) / hd
            A[i, j] = A[j, i] = val
    return A


def _head_output_hooks(bundle: ModelBundle):
    """Register forward hooks on every layer's attention output projection, capturing its INPUT
    (the concatenated per-head outputs z) with grad retained. Returns (handles, store) where
    store[layer] gets filled on the forward pass."""
    layers = layer_modules(bundle)
    store: Dict[int, torch.Tensor] = {}
    handles = []

    def mk(li):
        def hook(_m, inp):
            x = inp[0]
            if x.dim() == 3:
                x.retain_grad()
                store[li] = x
        return hook

    for li in range(bundle.num_layers):
        cp = C._cproj(C._attn_module(layers[li]))
        if cp is not None:
            handles.append(cp.register_forward_pre_hook(mk(li)))
    return handles, store


def _ioi_metric(bundle: ModelBundle, ex: Dict[str, str]) -> torch.Tensor:
    """Differentiable IOI logit-diff (logit[IO]-logit[S]) at the last position for one prompt."""
    dev = bundle_device(bundle)
    ids = bundle.tokenizer(ex["prompt"], return_tensors="pt").to(dev)["input_ids"]
    io_id = bundle.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
    s_id = bundle.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
    logits = bundle.model(input_ids=ids, use_cache=False).logits[0, -1, :]
    return logits[io_id] - logits[s_id]


def head_attribution_patching(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                              metric=_ioi_metric) -> np.ndarray:
    """ATP head saliency: |sum_pos z . d(metric)/d z| per head, averaged over prompts. This is
    the first-order (gradient x activation) estimate of the head's effect on the behavior."""
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    score = np.zeros(nU)
    for ex in prompts:
        handles, store = _head_output_hooks(bundle)
        try:
            bundle.model.zero_grad(set_to_none=True)
            m = metric(bundle, ex)
            m.backward()
            for li, x in store.items():
                if x.grad is None:
                    continue
                attr = (x * x.grad).detach()[0]                 # [T, nH*hd]
                for hi in range(nH):
                    s = attr[:, hi * hd:(hi + 1) * hd].sum().item()
                    score[C.head_index(li, hi, nH)] += s
        finally:
            for h in handles:
                h.remove()
    return np.abs(score / max(1, len(prompts)))


def head_attribution_graddrop(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                              metric=_ioi_metric) -> np.ndarray:
    """AtP*-style attribution with GradDrop (Kramar et al. 2024). Plain attribution patching gives
    false negatives when a node's direct and indirect gradient paths cancel -- exactly the
    self-repair cancellation that hides backups. GradDrop runs L attributions, each zeroing the
    gradient flowing through one transformer block's output (cutting that block's indirect path),
    and takes the mean of the per-node absolute scores. We implement the GradDrop fix (the part
    relevant to backup discovery); the QK-fix targets Q/K saturation, not head-output nodes."""
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    L = bundle.num_layers
    layers = layer_modules(bundle)
    acc = np.zeros(nU)
    for drop_l in range(L):
        per_drop = np.zeros(nU)
        for ex in prompts:
            handles, store = _head_output_hooks(bundle)
            # forward hook on the dropped block to zero the gradient on its output (cut its path)
            blk_handles = []
            def mk_drop():
                def fh(_m, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if torch.is_tensor(h) and h.requires_grad:
                        h.register_hook(lambda g: torch.zeros_like(g))
                    return out
                return fh
            blk_handles.append(layers[drop_l].register_forward_hook(mk_drop()))
            try:
                bundle.model.zero_grad(set_to_none=True)
                m = metric(bundle, ex)
                m.backward()
                for li, x in store.items():
                    if x.grad is None:
                        continue
                    attr = (x * x.grad).detach()[0]
                    for hi in range(nH):
                        per_drop[C.head_index(li, hi, nH)] += attr[:, hi * hd:(hi + 1) * hd].sum().item()
            finally:
                for h in handles:
                    h.remove()
                for h in blk_handles:
                    h.remove()
        acc += np.abs(per_drop / max(1, len(prompts)))
    return acc / max(1, L)


def conditional_attribution_patching(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                                     ablate_heads: Sequence[Tuple[int, int]],
                                     metric=_ioi_metric) -> np.ndarray:
    """GIM-style self-repair-aware attribution (our fair adaptation for backup DISCOVERY).

    GIM (Edin et al. 2025) corrects gradient attribution for self-repair during backprop; as
    published it returns corrected *scores*, not a backup *set*. The principle adapts to backup
    discovery directly: measure the gradient attribution of the behavior ON THE PRIMARY-ABLATED
    model. With the primaries ablated, the dormant backups become load-bearing, so their gradient
    is no longer muted -- a first-order analogue of our conditional co-ablation. We rank candidate
    heads by this conditional gradient x activation. (Heads in the ablated set get score 0.)"""
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    abl_units = set(C.head_index(l, h, nH) for (l, h) in ablate_heads)
    score = np.zeros(nU)
    for ex in prompts:
        abl_handles = C.register_heads_ablation(bundle, ablate_heads)   # zero primaries in-graph
        handles, store = _head_output_hooks(bundle)
        try:
            bundle.model.zero_grad(set_to_none=True)
            m = metric(bundle, ex)
            m.backward()
            for li, x in store.items():
                if x.grad is None:
                    continue
                attr = (x * x.grad).detach()[0]
                for hi in range(nH):
                    u = C.head_index(li, hi, nH)
                    if u in abl_units:
                        continue
                    score[u] += attr[:, hi * hd:(hi + 1) * hd].sum().item()
        finally:
            for h in handles:
                h.remove()
            for h in abl_handles:
                h.remove()
    return np.abs(score / max(1, len(prompts)))


def integrated_gradient_attribution(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                                    steps: int = 5, metric=_ioi_metric) -> np.ndarray:
    """EAP-IG-style: integrate grad x (clean activation) along alpha in (0,1] scaling the head
    outputs from a zero (corrupt) baseline to clean. More faithful than plain ATP."""
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    score = np.zeros(nU)
    layers = layer_modules(bundle)
    for ex in prompts:
        # accumulate grad over alpha steps while scaling the o_proj input by alpha.
        acc = {li: None for li in range(bundle.num_layers)}
        clean = {li: None for li in range(bundle.num_layers)}
        for step in range(1, steps + 1):
            alpha = step / steps
            store: Dict[int, torch.Tensor] = {}
            handles = []

            def mk(li):
                def hook(_m, inp):
                    x = inp[0]
                    if x.dim() == 3:
                        xs = (alpha * x)
                        xs.retain_grad()
                        store[li] = xs
                        if clean[li] is None:
                            clean[li] = x.detach()[0]
                        return (xs,) + tuple(inp[1:])
                    return inp
                return hook

            for li in range(bundle.num_layers):
                cp = C._cproj(C._attn_module(layers[li]))
                if cp is not None:
                    handles.append(cp.register_forward_pre_hook(mk(li)))
            try:
                bundle.model.zero_grad(set_to_none=True)
                m = metric(bundle, ex)
                m.backward()
                for li, xs in store.items():
                    if xs.grad is not None:
                        g = xs.grad.detach()[0]
                        acc[li] = g if acc[li] is None else acc[li] + g
            finally:
                for h in handles:
                    h.remove()
        for li in range(bundle.num_layers):
            if acc[li] is None or clean[li] is None:
                continue
            attr = (clean[li] * acc[li] / steps)                # [T, nH*hd]
            for hi in range(nH):
                score[C.head_index(li, hi, nH)] += attr[:, hi * hd:(hi + 1) * hd].sum().item()
    return np.abs(score / max(1, len(prompts)))


def faithfulness_curve(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                       ranked_units: Sequence[int], sizes: Sequence[int],
                       behavior=None) -> Dict[str, List[float]]:
    """Normalized logit-diff recovered when the circuit = top-k ranked units (ablate the
    COMPLEMENT, keep the circuit). norm = (m(circuit)-m(empty))/(m(full)-m(empty)).
    Returns sizes + normalized faithfulness; AUC is the trapezoidal area (the MIB CPR idea)."""
    behavior = behavior or (lambda heads: C.ioi_logit_diff(bundle, prompts, ablate_heads=heads))
    nH = bundle.num_heads
    nU = bundle.num_layers * nH
    all_heads = [C.head_layer_head(u, nH) for u in range(nU)]
    m_full = behavior(None)
    m_empty = behavior(all_heads)                                # everything ablated
    denom = (m_full - m_empty) or 1e-9
    out = {"sizes": [], "faithfulness": []}
    for k in sizes:
        keep = set(int(u) for u in ranked_units[:k])
        ablate = [lh for u, lh in enumerate(all_heads) if u not in keep]
        m_c = behavior(ablate)
        out["sizes"].append(int(k))
        out["faithfulness"].append(float((m_c - m_empty) / denom))
    xs = np.array(out["sizes"], dtype=float) / max(1, nU)
    out["auc"] = float(np.trapz(out["faithfulness"], xs) / max(1e-9, xs[-1] - xs[0])) if len(xs) > 1 else None
    return out


@contextmanager
def _gim_context(model, **gim_kwargs):
    """The released GIM context, with one device-placement fix and no algorithmic change.

    `gim.context.norm._swap_norms_with_detach` constructs its replacement modules without a
    `device=`/`dtype=` argument and only copies the parameter VALUES, so on a CUDA model the
    swapped norms keep CPU parameters and the forward pass raises a device mismatch. We enter the
    released context unchanged and then move any swapped module onto the model's device and dtype.
    The forward mathematics, the detached statistics, the softmax temperature and the Q/K/V
    gradient scales are exactly the package's.
    """
    from gim import GIM
    from gim.context.norm import LayerNormDetach, RMSNormDetach

    ref = next(model.parameters())
    with GIM(model, **gim_kwargs):
        for mod in model.modules():
            if isinstance(mod, (LayerNormDetach, RMSNormDetach)):
                mod.to(device=ref.device, dtype=ref.dtype)
        yield


def head_attribution_gim(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                         metric=_ioi_metric, **gim_kwargs) -> np.ndarray:
    """Head attribution under the OFFICIAL GIM backward pass (gim-explain package).

    GIM is not a separate attribution algorithm but a set of backward-pass modifications --
    detached LayerNorm/RMSNorm statistics, a temperature-adjusted softmax backward, and rescaled
    Q/K/V gradients -- applied as a context manager. That makes the comparison exact rather than
    approximate: we compute the SAME gradient-times-activation head score on the SAME prompts
    against the SAME metric, and the only thing that differs is the backward pass. Any gap is
    therefore attributable to the score's form, not to a difference in setup.

    Defaults are the package defaults, which are the paper's (T=2, q=k=0.25, v=0.5, freeze_norm).
    """
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    score = np.zeros(nU)
    for ex in prompts:
        handles, store = _head_output_hooks(bundle)
        try:
            bundle.model.zero_grad(set_to_none=True)
            with _gim_context(bundle.model, **gim_kwargs):
                m = metric(bundle, ex)
                m.backward()
            for li, x in store.items():
                if x.grad is None:
                    continue
                attr = (x * x.grad).detach()[0]
                for hi in range(nH):
                    score[C.head_index(li, hi, nH)] += \
                        attr[:, hi * hd:(hi + 1) * hd].sum().item()
        finally:
            for h in handles:
                h.remove()
    return np.abs(score / max(1, len(prompts)))


def head_attribution_gim_conditional(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                                     ablate_heads: Sequence[Tuple[int, int]],
                                     metric=_ioi_metric, **gim_kwargs) -> np.ndarray:
    """Official GIM computed on the PRIMARY-ABLATED model -- the matched-information baseline.

    CoAx is given the primary seed, so a comparison against a seed-free score is not matched in
    information. This variant hands GIM exactly the same seed: the gradients are taken on the model
    with the primaries already ablated, so both methods see the same intervention and differ only
    in what they read off it.
    """
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    score = np.zeros(nU)
    abl = C.register_heads_ablation_grouped(bundle, list(ablate_heads))
    try:
        for ex in prompts:
            handles, store = _head_output_hooks(bundle)
            try:
                bundle.model.zero_grad(set_to_none=True)
                with _gim_context(bundle.model, **gim_kwargs):
                    m = metric(bundle, ex)
                    m.backward()
                for li, x in store.items():
                    if x.grad is None:
                        continue
                    attr = (x * x.grad).detach()[0]
                    for hi in range(nH):
                        score[C.head_index(li, hi, nH)] += \
                            attr[:, hi * hd:(hi + 1) * hd].sum().item()
            finally:
                for h in handles:
                    h.remove()
    finally:
        for h in abl:
            h.remove()
    out = np.abs(score / max(1, len(prompts)))
    for (l, h) in ablate_heads:
        out[C.head_index(l, h, nH)] = 0.0
    return out
