"""First-order circuit-attribution baselines + faithfulness utilities.

These are first-order comparisons for the second-order co-ablation method. They score a head by
its single-component gradient or ablation effect, which can be muted for redundant components
under self-repair. We implement:

  * attribution_patching (ATP): grad of a behavioral metric w.r.t. a head's output, times the
    output -- the first-order Taylor estimate of zero-ablating that head. (Nanda 2023; the
    node-level form behind EAP / AtP*.)
  * integrated_gradient_attribution: activation-space EAP-IG from a zero head-output baseline
    to the clean activation (Hanna et al. 2024).
  * head_attribution_graddrop: the AtP* GradDrop correction, adapted to the same zero-ablation
    head-output target (Kramar et al. 2024).

Plus faithfulness_curve: normalized logit-diff recovered vs. circuit size, the metric the
field (MIB, Wang 2022) actually reports -- not just same-circuit AUC.

Kept separate from coablation.py so the novel method stays clean.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    num_examples = 0
    for ex in prompts:
        handles, store = _head_output_hooks(bundle)
        try:
            bundle.model.zero_grad(set_to_none=True)
            m = metric(bundle, ex)
            m.backward()
            batch_size = next(iter(store.values())).shape[0]
            per_example = np.zeros((batch_size, nU))
            for li, x in store.items():
                if x.grad is None:
                    continue
                attr = (x * x.grad).detach()                    # [B, T, nH*hd]
                values = attr.reshape(*attr.shape[:-1], nH, hd).sum(dim=(1, 3))
                start = li * nH
                per_example[:, start:start + nH] += values.float().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        # AtP is E_x[|I_hat(n; x)|], not |E_x[I_hat(n; x)]|.  Taking the
        # magnitude before the dataset average avoids cancellation across examples.
        score += np.abs(per_example).sum(axis=0)
        num_examples += per_example.shape[0]
    return score / max(1, num_examples)


def _register_residual_contribution_graddrop(layer):
    """Keep a residual block's forward value but detach only its residual contribution."""
    state: Dict[str, torch.Tensor] = {}

    def capture_residual(_module, inputs):
        state["residual"] = inputs[0]

    def detach_contribution(_module, _inputs, output):
        residual = state["residual"]
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        # With an Accelerate-sharded model, the pre-hook can observe the block input
        # before it is transferred to the block's execution device, while the forward
        # hook observes the output on that execution device.  Moving the identity
        # residual to the output device preserves the same computation and autograd
        # path while avoiding a cross-device subtraction at shard boundaries.
        if residual.device != hidden.device:
            residual = residual.to(hidden.device)
        dropped = residual + (hidden - residual).detach()
        if isinstance(output, tuple):
            return (dropped,) + tuple(output[1:])
        if isinstance(output, list):
            return [dropped] + list(output[1:])
        return dropped

    return [
        layer.register_forward_pre_hook(capture_residual),
        layer.register_forward_hook(detach_contribution),
    ]


def head_attribution_graddrop(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                              metric=_ioi_metric,
                              ablate_heads: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """AtP* GradDrop for head-output nodes, using a zero-ablation perturbation.

    For each layer, the forward value is kept exactly unchanged while the Jacobian of that
    layer's *residual contribution* is detached.  The identity residual path remains live; this
    is the distinction between GradDrop and stopping the gradient at the whole block output.
    Absolute per-example estimates are summed over the ``L`` dropped layers and divided by
    ``L - 1``, as in Kramar et al. (2024).  The Q/K saturation correction is not applicable to
    the post-attention head-output nodes scored here.  When ``ablate_heads`` is supplied, the
    same estimator is evaluated in that intervened model and the ablated units remain at zero.
    """
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    L = bundle.num_layers
    layers = layer_modules(bundle)
    ablated_by_layer: Dict[int, List[int]] = {}
    for layer, head in (ablate_heads or []):
        ablated_by_layer.setdefault(layer, []).append(head)
    acc = np.zeros(nU, dtype=np.float64)
    num_examples = 0
    for ex in prompts:
        per_example = None
        for drop_l in range(L):
            ablation_handles = C.register_heads_ablation(bundle, ablate_heads) \
                if ablate_heads else []
            handles, store = _head_output_hooks(bundle)
            blk_handles = _register_residual_contribution_graddrop(layers[drop_l])
            try:
                bundle.model.zero_grad(set_to_none=True)
                m = metric(bundle, ex)
                m.backward()
                batch_size = next(iter(store.values())).shape[0]
                if per_example is None:
                    per_example = np.zeros((batch_size, nU), dtype=np.float64)
                per_drop = np.zeros((batch_size, nU), dtype=np.float64)
                for li, x in store.items():
                    if x.grad is None:
                        continue
                    attr = (x * x.grad).detach()
                    values = attr.reshape(*attr.shape[:-1], nH, hd).sum(dim=(1, 3))
                    values = values.float().cpu().numpy()
                    if li in ablated_by_layer:
                        values[:, ablated_by_layer[li]] = 0.0
                    start = li * nH
                    per_drop[:, start:start + nH] += values
            finally:
                for h in handles:
                    h.remove()
                for h in blk_handles:
                    h.remove()
                for h in ablation_handles:
                    h.remove()
            per_example += np.abs(per_drop)
        if per_example is not None:
            acc += per_example.sum(axis=0) / max(1, L - 1)
            num_examples += per_example.shape[0]
    return acc / max(1, num_examples)


def conditional_attribution_patching(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                                     ablate_heads: Sequence[Tuple[int, int]],
                                     metric=_ioi_metric) -> np.ndarray:
    """Attribution patching evaluated in the primary-ablated graph.

    This is the removed-state gradient control for backup discovery: it uses the same conditional
    context as CoAx, but remains a first-order local estimate rather than a finite intervention
    difference. The ablation is installed in-graph, so primary-head contributions and gradients
    remain zero throughout backward. Heads in the ablated set receive score zero.
    """
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    ablated_by_layer: Dict[int, List[int]] = {}
    for layer, head in ablate_heads:
        ablated_by_layer.setdefault(layer, []).append(head)
    score = np.zeros(nU)
    num_examples = 0
    for ex in prompts:
        abl_handles = C.register_heads_ablation(bundle, ablate_heads)   # zero primaries in-graph
        handles, store = _head_output_hooks(bundle)
        try:
            bundle.model.zero_grad(set_to_none=True)
            m = metric(bundle, ex)
            m.backward()
            batch_size = next(iter(store.values())).shape[0]
            per_example = np.zeros((batch_size, nU))
            for li, x in store.items():
                if x.grad is None:
                    continue
                attr = (x * x.grad).detach()
                values = attr.reshape(*attr.shape[:-1], nH, hd).sum(dim=(1, 3))
                values = values.float().cpu().numpy()
                if li in ablated_by_layer:
                    values[:, ablated_by_layer[li]] = 0.0
                start = li * nH
                per_example[:, start:start + nH] += values
        finally:
            for h in handles:
                h.remove()
            for h in abl_handles:
                h.remove()
        score += np.abs(per_example).sum(axis=0)
        num_examples += per_example.shape[0]
    return score / max(1, num_examples)


def integrated_gradient_attribution(bundle: ModelBundle, prompts: Sequence[Dict[str, str]],
                                    steps: int = 5, metric=_ioi_metric) -> np.ndarray:
    """Activation-space EAP-IG for zero-ablation of head-output nodes.

    This is the standard component-activation variant: cache the true clean head outputs, then
    interpolate one attention layer at a time from zero to its clean concatenated-head output.
    Other layers are recomputed normally.  It therefore costs ``steps * num_layers``
    forward/backward passes per example; simultaneously scaling every layer is a different path
    and is not activation-space EAP-IG.
    """
    nH, hd = bundle.num_heads, bundle.head_dim
    nU = bundle.num_layers * nH
    score = np.zeros(nU, dtype=np.float64)
    layers = layer_modules(bundle)
    for ex in prompts:
        clean: Dict[int, torch.Tensor] = {}
        clean_handles = []

        def capture_clean(li):
            def hook(_module, inputs):
                x = inputs[0]
                if x.dim() == 3:
                    clean[li] = x.detach().clone()
            return hook

        for li, layer in enumerate(layers):
            cp = C._cproj(C._attn_module(layer))
            if cp is not None:
                clean_handles.append(cp.register_forward_pre_hook(capture_clean(li)))
        try:
            with torch.no_grad():
                metric(bundle, ex)
        finally:
            for handle in clean_handles:
                handle.remove()

        for li, layer in enumerate(layers):
            if li not in clean:
                continue
            grad_sum = torch.zeros_like(clean[li], dtype=torch.float32)
            cp = C._cproj(C._attn_module(layer))
            for step in range(1, steps + 1):
                alpha = step / steps
                store: Dict[str, torch.Tensor] = {}

                def interpolate(_module, inputs, alpha=alpha, clean_value=clean[li]):
                    x = inputs[0]
                    xs = alpha * clean_value.to(device=x.device, dtype=x.dtype) + x * 0.0
                    xs.retain_grad()
                    store["value"] = xs
                    return (xs,) + tuple(inputs[1:])

                handle = cp.register_forward_pre_hook(interpolate)
                try:
                    bundle.model.zero_grad(set_to_none=True)
                    m = metric(bundle, ex)
                    m.backward()
                    xs = store.get("value")
                    if xs is not None and xs.grad is not None:
                        grad_sum += xs.grad.detach().float()
                finally:
                    handle.remove()
            attr = -clean[li].float() * (grad_sum / max(1, steps))
            for hi in range(nH):
                score[C.head_index(li, hi, nH)] += \
                    attr[..., hi * hd:(hi + 1) * hd].sum().item()
    # Match the official EAP-IG aggregation: accumulate signed dataset scores,
    # then rank nodes by the magnitude of that aggregate.
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
