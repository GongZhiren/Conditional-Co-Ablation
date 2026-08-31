"""Activation freezing: the causal validator used to LABEL compensators independently of CoAx.

The CoAx score is a label-free Fisher energy. To evaluate it we need backup labels produced by
something that shares none of its machinery, or the evaluation is circular. This module supplies
that validator using activation freezing, a standard intervention in the self-repair literature:

    ablate the primary set S, but FREEZE a candidate u to the activation it had on the clean
    model, so u cannot respond to S's removal.

If u is a genuine compensator, denying it the chance to wake up removes part of the repair and
the task metric falls further. If u is irrelevant, freezing it changes nothing. The readout

    repair_removed(u | S) = metric(ablate S) - metric(ablate S, freeze u)

is computed entirely from the task metric and activations -- no Fisher geometry, no conditional
energy, no CoAx quantity of any kind.

Everything is batched by sequence length, matching curvgraph.coablation, so scoring all units is
O(|U|) batched forwards rather than O(|U| x prompts) single ones.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from curvgraph._core.model import ModelBundle, layer_modules
from . import circuits as C


def capture_final_head_slices(bundle: ModelBundle, ids: torch.Tensor,
                              heads: Sequence[Tuple[int, int]],
                              ablate: Optional[Sequence[Tuple[int, int]]] = None
                              ) -> Dict[Tuple[int, int], torch.Tensor]:
    """Capture each requested head's pre-projection slice at the final position."""
    head_dim = bundle.head_dim
    layers = layer_modules(bundle)
    by_layer: Dict[int, List[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    captured: Dict[Tuple[int, int], torch.Tensor] = {}

    def make_hook(layer_index: int, head_indices: List[int]):
        def hook(_module, inputs):
            x = inputs[0].detach()
            for head_index in head_indices:
                start = head_index * head_dim
                captured[(layer_index, head_index)] = x[0, -1, start:start + head_dim].clone()
        return hook

    handles = []
    for layer_index, head_indices in by_layer.items():
        projection = C._cproj(C._attn_module(layers[layer_index]))
        handles.append(projection.register_forward_pre_hook(make_hook(layer_index, head_indices)))
    handles += C.register_heads_ablation(bundle, ablate) if ablate else []
    try:
        C._forward_logits(bundle, ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def ioi_logit_diff_with_patch(bundle: ModelBundle, ids: torch.Tensor, io_token: int, s_token: int,
                              ablate: Optional[Sequence[Tuple[int, int]]] = None,
                              patch: Optional[Dict[Tuple[int, int], torch.Tensor]] = None) -> float:
    """IOI final-position margin with optional ablation and clean-activation freezing."""
    head_dim = bundle.head_dim
    layers = layer_modules(bundle)
    handles = C.register_heads_ablation(bundle, ablate) if ablate else []
    if patch:
        by_layer: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
        for (layer, head), value in patch.items():
            by_layer.setdefault(int(layer), []).append((int(head), value))

        def make_hook(items):
            def hook(_module, inputs):
                x = inputs[0].clone()
                for head_index, value in items:
                    start = head_index * head_dim
                    x[0, -1, start:start + head_dim] = value.to(dtype=x.dtype, device=x.device)
                return (x,) + tuple(inputs[1:])
            return hook

        for layer_index, items in by_layer.items():
            projection = C._cproj(C._attn_module(layers[layer_index]))
            handles.append(projection.register_forward_pre_hook(make_hook(items)))
    try:
        logits = C._forward_logits(bundle, ids)[0, -1]
        return float(logits[io_token] - logits[s_token])
    finally:
        for handle in handles:
            handle.remove()


def length_buckets(seqs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Group equal-length token sequences into stacked [B, T] batches."""
    buckets: Dict[int, List[torch.Tensor]] = {}
    for ids in seqs:
        buckets.setdefault(int(ids.shape[1]), []).append(ids.reshape(1, -1))
    return [torch.cat(buckets[T], dim=0) for T in sorted(buckets)]


def capture_head_activations(bundle: ModelBundle, ids: torch.Tensor,
                             heads: Sequence[Tuple[int, int]],
                             ablate: Optional[Sequence[Tuple[int, int]]] = None
                             ) -> Dict[Tuple[int, int], torch.Tensor]:
    """Each head's attention-output-projection input slice at EVERY position: {(l,h): [B,T,hd]}.

    Capturing all positions (not only the final one) means a later freeze pins the head's whole
    trajectory to its clean value, so the head cannot compensate anywhere in the sequence.
    """
    hd = bundle.head_dim
    layers = layer_modules(bundle)
    by_layer: Dict[int, List[int]] = {}
    for (l, h) in heads:
        by_layer.setdefault(int(l), []).append(int(h))
    cap: Dict[Tuple[int, int], torch.Tensor] = {}

    def mk(li: int, hids: List[int]):
        def hook(_m, inp):
            if not inp or inp[0].dim() != 3:
                return
            x = inp[0].detach()
            for hi in hids:
                cap[(li, hi)] = x[..., hi * hd:(hi + 1) * hd].clone()
        return hook

    handles = []
    for li, hids in by_layer.items():
        cp = C._cproj(C._attn_module(layers[li]))
        if cp is not None:
            handles.append(cp.register_forward_pre_hook(mk(li, hids)))
    abl = C.register_heads_ablation_grouped(bundle, ablate) if ablate else []
    try:
        C._forward_logits(bundle, ids)
    finally:
        for h in handles + abl:
            h.remove()
    return cap


def register_freeze(bundle: ModelBundle, frozen: Dict[Tuple[int, int], torch.Tensor]):
    """Overwrite each listed head's output slice with a supplied [B,T,hd] tensor.

    Registered AFTER the ablation hooks on the same module, so a head that is both ablated and
    frozen ends up frozen -- callers should keep the two sets disjoint.
    """
    hd = bundle.head_dim
    layers = layer_modules(bundle)
    by_layer: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    for (l, h), v in frozen.items():
        by_layer.setdefault(int(l), []).append((int(h), v))
    handles = []

    def mk(items):
        def hook(_m, inp):
            if not inp or inp[0].dim() != 3:
                return inp
            x = inp[0].clone()
            for hi, v in items:
                vv = v.to(x.dtype).to(x.device)
                if vv.shape[:2] != x.shape[:2]:
                    continue  # different bucket; leave untouched rather than mis-broadcast
                x[..., hi * hd:(hi + 1) * hd] = vv
            return (x,) + tuple(inp[1:])
        return hook

    for li, items in by_layer.items():
        cp = C._cproj(C._attn_module(layers[li]))
        if cp is not None:
            handles.append(cp.register_forward_pre_hook(mk(items)))
    return handles


def batched_final_logits(bundle: ModelBundle, ids: torch.Tensor,
                         ablate: Optional[Sequence[Tuple[int, int]]] = None,
                         frozen: Optional[Dict[Tuple[int, int], torch.Tensor]] = None
                         ) -> torch.Tensor:
    """Final-position logits [B, V] under the given ablation and freeze."""
    handles = C.register_heads_ablation_grouped(bundle, ablate) if ablate else []
    if frozen:
        handles += register_freeze(bundle, frozen)
    try:
        return C._forward_logits(bundle, ids)[:, -1, :].detach()
    finally:
        for h in handles:
            h.remove()


def batched_logit_diff(bundle: ModelBundle, ids: torch.Tensor,
                       io_ids: torch.Tensor, s_ids: torch.Tensor,
                       ablate: Optional[Sequence[Tuple[int, int]]] = None,
                       frozen: Optional[Dict[Tuple[int, int], torch.Tensor]] = None) -> float:
    """Mean (logit[IO] - logit[S]) at the final position over a [B, T] batch."""
    logits = batched_final_logits(bundle, ids, ablate, frozen)
    io = logits.gather(-1, io_ids.view(-1, 1)).squeeze(-1)
    s = logits.gather(-1, s_ids.view(-1, 1)).squeeze(-1)
    return float((io - s).mean().item())


def batched_kl_from_clean(bundle: ModelBundle, ids: torch.Tensor,
                          clean_logits: torch.Tensor,
                          ablate: Optional[Sequence[Tuple[int, int]]] = None,
                          frozen: Optional[Dict[Tuple[int, int], torch.Tensor]] = None) -> float:
    """Mean KL(p_clean || p_intervened) at the final position -- distribution-level damage.

    The task-metric readout (logit-difference) shares its objective with the gradient baselines,
    which are differentiated against exactly that scalar. This full-distribution damage measure
    does not privilege any particular answer direction, so running the validator under both makes
    the metric-dependence of the comparison explicit instead of implicit.
    """
    logits = batched_final_logits(bundle, ids, ablate, frozen)
    logp0 = torch.log_softmax(clean_logits.float(), dim=-1)
    logp = torch.log_softmax(logits.float(), dim=-1)
    kl = (logp0.exp() * (logp0 - logp)).sum(-1)
    return float(kl.mean().item())


class FreezeValidator:
    """Score every unit by how much of the repair its wake-up carries, for a given primary set.

    Usage:
        v = FreezeValidator(bundle, prompts)
        res = v.repair_removed(primary_heads)      # {unit: repair_removed}, plus reference points
    """

    def __init__(self, bundle: ModelBundle, prompts: Sequence[Dict[str, str]], device=None,
                 metric_kind: str = "logit_diff"):
        if metric_kind not in ("logit_diff", "kl"):
            raise ValueError(f"metric_kind must be 'logit_diff' or 'kl', got {metric_kind!r}")
        self.metric_kind = metric_kind
        self.bundle = bundle
        self.nH = bundle.num_heads
        self.nU = bundle.num_layers * bundle.num_heads
        dev = device or next(bundle.model.parameters()).device
        # Bucket prompts by tokenized length, carrying the IO/S answer ids alongside.
        rows: Dict[int, List[Tuple[torch.Tensor, int, int]]] = {}
        for ex in prompts:
            ids = bundle.tokenizer(ex["prompt"], return_tensors="pt").to(dev)["input_ids"]
            if ids.shape[1] < 4:
                continue
            io = bundle.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
            s = bundle.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
            rows.setdefault(int(ids.shape[1]), []).append((ids, io, s))
        self.batches = []
        for T in sorted(rows):
            grp = rows[T]
            self.batches.append({
                "ids": torch.cat([g[0] for g in grp], dim=0),
                "io": torch.tensor([g[1] for g in grp], device=dev),
                "s": torch.tensor([g[2] for g in grp], device=dev),
                "n": len(grp),
            })
        if not self.batches:
            raise RuntimeError("FreezeValidator got no usable prompts")
        if metric_kind == "kl":
            for b in self.batches:
                b["clean_logits"] = batched_final_logits(bundle, b["ids"])

    def _metric(self, ablate=None, frozen_per_batch=None) -> float:
        """Prompt-count-weighted mean of the configured damage readout across buckets.

        `logit_diff` is signed task performance (higher = intact), so repair_removed is
        metric(S) - metric(S, freeze u). `kl` is damage (higher = worse), so the sign is
        flipped in repair_removed() to keep "positive = u was repairing" in both cases.
        """
        tot, n = 0.0, 0
        for i, b in enumerate(self.batches):
            fz = None if frozen_per_batch is None else frozen_per_batch[i]
            if self.metric_kind == "kl":
                val = batched_kl_from_clean(self.bundle, b["ids"], b["clean_logits"],
                                            ablate=ablate, frozen=fz)
            else:
                val = batched_logit_diff(self.bundle, b["ids"], b["io"], b["s"],
                                         ablate=ablate, frozen=fz)
            tot += val * b["n"]
            n += b["n"]
        return tot / max(1, n)

    def repair_removed_set(self, primary: Sequence[Tuple[int, int]],
                           units: Sequence[int]) -> float:
        """Repair removed by freezing a WHOLE SET jointly -- the top-k causal gain readout.

        Freezing units one at a time and summing would re-introduce exactly the additivity
        assumption this paper argues against, so a selection is scored by freezing all of its
        members together.
        """
        prim_units = set(C.head_index(l, h, self.nH) for (l, h) in primary)
        sel = [C.head_layer_head(u, self.nH) for u in units
               if int(u) not in prim_units]
        if not sel:
            return 0.0
        m_S = self._metric(ablate=list(primary))
        clean_acts = [capture_head_activations(self.bundle, b["ids"], sel)
                      for b in self.batches]
        frozen = [{lh: acts[lh] for lh in sel if lh in acts} for acts in clean_acts]
        m_frz = self._metric(ablate=list(primary), frozen_per_batch=frozen)
        return (m_frz - m_S) if self.metric_kind == "kl" else (m_S - m_frz)

    def repair_removed(self, primary: Sequence[Tuple[int, int]],
                       units: Optional[Sequence[int]] = None,
                       progress: Optional[Callable[[int, int], None]] = None) -> Dict:
        """repair_removed(u|S) = metric(ablate S) - metric(ablate S, freeze u to clean).

        Positive means u was actively helping to repair the damage from ablating S.
        """
        prim_units = set(C.head_index(l, h, self.nH) for (l, h) in primary)
        pool = [u for u in (units if units is not None else range(self.nU))
                if u not in prim_units]

        m_clean = self._metric()
        m_S = self._metric(ablate=list(primary))

        # Clean activations for every candidate, per bucket, captured once.
        all_heads = [C.head_layer_head(u, self.nH) for u in pool]
        clean_acts = [capture_head_activations(self.bundle, b["ids"], all_heads)
                      for b in self.batches]

        out: Dict[int, float] = {}
        for j, u in enumerate(pool):
            lh = C.head_layer_head(u, self.nH)
            frozen = [{lh: acts[lh]} for acts in clean_acts]
            m_frz = self._metric(ablate=list(primary), frozen_per_batch=frozen)
            # positive == "freezing u made things worse", i.e. u was carrying repair
            out[u] = (m_frz - m_S) if self.metric_kind == "kl" else (m_S - m_frz)
            if progress is not None:
                progress(j + 1, len(pool))
        return {"clean": m_clean, "seed_ablated": m_S, "repair_removed": out}
