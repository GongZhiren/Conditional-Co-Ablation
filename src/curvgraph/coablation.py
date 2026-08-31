"""Second-order co-ablation curvature for circuit discovery.

The single-ablation kernel H_uv = E<dz_u, dz_v>_F cannot see
*compensation*: when two units substitute for each other, ablating one alone barely moves
the output, so their first-order affinity is small. This module adds the genuinely
second-order signals that capture compensation:

  * pairwise synergy   I_uv = dz_{u,v} - dz_u - dz_v   (Fisher-weighted, centered)
      -> large for MUTUALLY-compensating / cooperating pairs (e.g. name-mover cliques).
  * conditional compensation  comp_u(S) = ||dz_u | S ablated|| - ||dz_u | {}||
      -> large for PARALLEL-SUBSTITUTE backups (dormant until the primaries S are gone).

Both are computed with plain HF hooks via curvgraph.circuits primitives. Teacher top-r and
any conditioning-set baseline are computed once and reused, so the cost is O(M) forwards for
the conditional route and O(M^2) only for the explicit pairwise route (use a candidate set
or the greedy route for large models).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from curvgraph._core.model import ModelBundle, bundle_device, layer_modules
from . import circuits as C


def coactivation_affinity(bundle: ModelBundle, seqs, head_set: Sequence[int]) -> np.ndarray:
    """Head co-activation correlation on the SAME inputs (fair input-side control)."""
    nH, hd = bundle.num_heads, bundle.head_dim
    layers = layer_modules(bundle)
    cap = {u: [] for u in head_set}
    sset = set(head_set)
    handles = []

    def mk(li):
        def hook(_m, inp):
            x = inp[0].detach()
            if x.dim() != 3:
                return
            for hi in range(nH):
                u = C.head_index(li, hi, nH)
                if u in sset:
                    cap[u].append(x[..., hi * hd:(hi + 1) * hd].norm(dim=-1).reshape(-1).float().cpu().numpy())
        return hook

    for li in range(bundle.num_layers):
        cp = C._cproj(C._attn_module(layers[li]))
        if cp is not None:
            handles.append(cp.register_forward_pre_hook(mk(li)))
    try:
        for ids in seqs:
            C._forward_logits(bundle, ids)
    finally:
        for h in handles:
            h.remove()
    mat = np.stack([np.concatenate(cap[u]) for u in head_set], axis=0)
    return np.nan_to_num(np.corrcoef(mat))


def auc_same_circuit(aff: np.ndarray, head_set: Sequence[int], members,
                     restrict_to=None) -> Optional[float]:
    """Pair-level same-circuit ROC-AUC. If restrict_to is given, only pairs whose BOTH
    endpoints are in restrict_to are scored (the hard 'functional specificity' test:
    negatives = other active circuit heads, not easy active-vs-inactive separation)."""
    from sklearn.metrics import roc_auc_score
    idx = {u: i for i, u in enumerate(head_set)}
    mem = np.zeros(len(head_set), dtype=bool)
    for u in members:
        if u in idx:
            mem[idx[u]] = True
    keep = np.ones(len(head_set), dtype=bool)
    if restrict_to is not None:
        rset = set(restrict_to)
        keep = np.array([u in rset for u in head_set])
    iu = np.triu_indices(len(head_set), 1)
    pk = keep[iu[0]] & keep[iu[1]]
    same = (mem[iu[0]] & mem[iu[1]])[pk]
    vals = np.nan_to_num(aff[iu][pk], nan=0.0, posinf=0.0, neginf=0.0)
    if not same.any() or not (~same).any():
        return None
    return float(roc_auc_score(same.astype(int), vals))


class CoAblation:
    def __init__(self, bundle: ModelBundle, sequences: Sequence[torch.Tensor], top_r: int = 256,
                 ablation_mode: str = "zero", feature_mode: str = "fisher_centered",
                 freeze_ln: str = "none", position_mode: str = "all"):
        self.bundle = bundle
        self.nH = bundle.num_heads
        self.nU = bundle.num_layers * bundle.num_heads
        self.seqs = list(sequences)
        if not self.seqs:
            raise RuntimeError("CoAblation needs at least one tokenized sequence.")
        # ablation_mode: "zero" (default) or "mean" (overwrite a head's output slice with its
        # mean activation -- the expectation of resample ablation). Used for the robustness check.
        self.ablation_mode = ablation_mode
        # feature_mode: design-ablation knob for the perturbation feature.
        #   fisher_centered (default) : sqrt(p0) * (delta - E_p0[delta])   -- the Fisher metric
        #   fisher_uncentered          : sqrt(p0) * delta
        #   l2_centered                : delta - E_p0[delta]   (no Fisher weighting)
        #   l2_uncentered              : delta                  (plain logit-gap L2)
        self.feature_mode = feature_mode
        # freeze_ln: "none" (default) | "scale" | "full". When set, every ablated forward reuses
        # the CLEAN LayerNorm denominator, which removes the LayerNorm-rescaling channel of
        # self-repair from the score and leaves only genuine component re-routing (lnfreeze.py).
        self.freeze_ln = freeze_ln
        self._ln = None
        # position_mode: which output positions the Fisher energy is measured over.
        #   "all"  (library default) : logits[:, :-1] -- every next-token prediction within
        #            the window, excluding the answer slot.
        #   "last" : logits[:, -1:] -- only the answer slot. This is the paper's main IOI
        #            discovery setting for prompts ending where the answer is produced.
        #   "full" : logits[:, :] -- every position including the answer slot.
        # None of the three uses labels, a task metric, or the answer token's identity; they
        # differ only in which output positions the Fisher energy is averaged over.
        if position_mode not in ("all", "last", "full"):
            raise ValueError(f"position_mode must be 'all', 'last' or 'full', got {position_mode!r}")
        self.position_mode = position_mode
        self._means = C.head_mean_vectors(bundle, self.seqs) if ablation_mode == "mean" else None
        # Bucket sequences by length so each bucket is one stacked [B, T] forward instead of B
        # per-sequence forwards (~B x speedup; rows are independent under the causal mask, so the
        # batched logits are numerically identical to the per-sequence loop).
        buckets: Dict[int, List[torch.Tensor]] = {}
        for ids in self.seqs:
            buckets.setdefault(int(ids.shape[1]), []).append(ids.reshape(1, -1))
        self.teacher: List[Dict[str, torch.Tensor]] = []
        for T in sorted(buckets):
            ids_b = torch.cat(buckets[T], dim=0)               # [B, T]
            logits = self._slice(C._forward_logits(bundle, ids_b))
            # top-r by logit == top-r by prob (softmax is monotone); softmax over the gathered
            # top-r is exactly the renormalized full-softmax top-r, so this avoids materializing
            # a full-vocab softmax (critical for 256k-vocab models like Gemma-2).
            tl, ti = torch.topk(logits, k=min(top_r, logits.shape[-1]), dim=-1)
            tp = torch.softmax(tl, dim=-1)
            self.teacher.append({"ids": ids_b, "ti": ti, "tp": tp, "tl": tl})
        if freeze_ln != "none":
            from .lnfreeze import LayerNormFreezer
            self._ln = LayerNormFreezer(bundle.model, mode=freeze_ln)
            for i, t in enumerate(self.teacher):
                self._ln.record(i, lambda ids=t["ids"]: C._forward_logits(bundle, ids))
            # Replaying clean stats on a clean pass must be a no-op; fail loudly if it is not.
            self._ln.verify(0, lambda ids=self.teacher[0]["ids"]: C._forward_logits(bundle, ids))

    def _slice(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the output positions the energy is measured over (see position_mode)."""
        if self.position_mode == "last":
            return logits[:, -1:, :]
        if self.position_mode == "full":
            return logits
        return logits[:, :-1, :]

    def _lh(self, u: int) -> Tuple[int, int]:
        return C.head_layer_head(int(u), self.nH)

    def _top_logits(self, ablate_heads) -> List[torch.Tensor]:
        """Per-bucket gathered logits on the teacher top-r support, under the given ablation."""
        if not ablate_heads:
            handles = []
        elif self._means is None:
            # zero ablation: one hook per layer (O(#layers)), not one per head -- keeps the
            # sequential-pruning hot path fast when the conditioning set grows to hundreds of heads.
            handles = C.register_heads_ablation_grouped(self.bundle, ablate_heads)
        else:
            handles = C.register_heads_ablation(self.bundle, ablate_heads, replacements=self._means)
        out = []
        try:
            for i, t in enumerate(self.teacher):
                ln_handles = self._ln.hooks(i) if self._ln is not None else []
                try:
                    s = self._slice(C._forward_logits(self.bundle, t["ids"]))
                finally:
                    for h in ln_handles:
                        h.remove()
                out.append(torch.gather(s, -1, t["ti"]))
        finally:
            for h in handles:
                h.remove()
        return out

    def _weight(self, delta, t, probs=None):
        """Apply the configured feature transform (centering and/or Fisher weighting).

        `probs` overrides the clean distribution p_0 used to define the Fisher metric. Passing the
        SEED-ABLATED distribution p_S measures the conditional effect in the local KL geometry
        around the intervened model rather than in the clean one. The default (clean p_0) is a
        deliberate choice: it gives every candidate and every intervention a COMMON output
        coordinate system, so conditional energies are comparable across seeds; a per-intervention
        metric would put each one in its own chart. The audit script compares the two.
        """
        p = t["tp"] if probs is None else probs
        if self.feature_mode in ("fisher_centered", "l2_centered"):
            delta = delta - (p * delta).sum(-1, keepdim=True)
        if self.feature_mode in ("fisher_centered", "fisher_uncentered"):
            delta = torch.sqrt(p.clamp_min(1e-12)) * delta
        return delta

    def _energy(self, student_top: List[torch.Tensor], ref_top=None) -> float:
        """Fisher-weighted, centered squared-norm energy of the perturbation, computed ON GPU
        (one scalar transfer), avoiding the [P, r] CPU/numpy transfer of _feature. Used in the
        sequential-pruning hot path where only the scalar energy is needed per candidate."""
        tot = 0.0
        P = 0
        for i, t in enumerate(self.teacher):
            ref = t["tl"] if ref_top is None else ref_top[i]
            w = self._weight(ref - student_top[i], t)
            tot += float((w.float() ** 2).sum().item())
            P += w.shape[0] * w.shape[1] if w.dim() == 3 else w.reshape(-1, w.shape[-1]).shape[0]
        return tot / max(1, P)

    def _feature(self, student_top: List[torch.Tensor], ref_top=None,
                 probs_per_bucket=None) -> np.ndarray:
        """Centered, Fisher-weighted gap of student vs reference (default = dense teacher)."""
        feats = []
        for i, t in enumerate(self.teacher):
            ref = t["tl"] if ref_top is None else ref_top[i]
            pr = None if probs_per_bucket is None else probs_per_bucket[i]
            w = self._weight(ref - student_top[i], t, probs=pr)
            feats.append(w.reshape(-1, w.shape[-1]).float().cpu().numpy())
        return np.concatenate(feats, axis=0)  # [P, r]

    # ----------------------------------------------------------- first order
    def single_features(self, head_set: Sequence[int]) -> np.ndarray:
        """dz_u centered Fisher features for each unit, stacked [m, P, r]."""
        return np.stack([self._feature(self._top_logits([self._lh(u)])) for u in head_set], axis=0)

    @staticmethod
    def first_order_affinity(single: np.ndarray) -> np.ndarray:
        m, P = single.shape[0], single.shape[1]
        flat = single.reshape(m, -1)
        H = (flat @ flat.T) / max(1, P)
        d = np.sqrt(np.clip(np.diag(H), 1e-12, None))
        return H / np.outer(d, d)

    # ---------------------------------------------------------- second order
    def pairwise_synergy(self, head_set: Sequence[int], single: Optional[np.ndarray] = None,
                         normalize: bool = True) -> np.ndarray:
        """S_uv = mean_t ||dz_{u,v} - dz_u - dz_v||^2 over the head set (symmetric, O(m^2))."""
        if single is None:
            single = self.single_features(head_set)
        m, P = len(head_set), single.shape[1]
        S = np.zeros((m, m))
        for a, b in combinations(range(m), 2):
            dz_uv = self._feature(self._top_logits([self._lh(head_set[a]), self._lh(head_set[b])]))
            inter = dz_uv - single[a] - single[b]
            S[a, b] = S[b, a] = float((inter ** 2).sum() / max(1, P))
        if normalize:
            e = np.array([float((single[i] ** 2).sum() / max(1, P)) for i in range(m)])
            S = S / (np.sqrt(np.outer(e, e)) + 1e-12)
        return S

    def conditional_compensation(self, seed_heads: Sequence[Tuple[int, int]],
                                 head_set: Optional[Sequence[int]] = None,
                                 geometry: str = "clean") -> Dict[str, np.ndarray]:
        """For each unit u: marginal Fisher energy of ablating u GIVEN seed_heads ablated,
        vs unconditionally. comp_u = cond_u - single_u is large for parallel-substitute
        backups (dormant until the seed primaries are gone). Seed baseline computed once."""
        head_set = list(range(self.nU)) if head_set is None else list(head_set)
        seed = list(seed_heads)
        seed_units = set(C.head_index(l, h, self.nH) for (l, h) in seed)
        base_seed_top = self._top_logits(seed)                 # reference = seed ablated (once)
        dense_top = [t["tl"] for t in self.teacher]            # reference = dense (once)
        if geometry not in ("clean", "conditional"):
            raise ValueError(f"geometry must be 'clean' or 'conditional', got {geometry!r}")
        # "conditional": weight/center the conditional feature by p_S, the seed-ablated
        # distribution on the same top-r support, instead of the clean p_0.
        cond_probs = ([torch.softmax(b, dim=-1) for b in base_seed_top]
                      if geometry == "conditional" else None)
        single = np.full(self.nU, np.nan)
        cond = np.full(self.nU, np.nan)
        # couple_u(S) = E|| dz_{u|S} - dz_u ||^2, the Fisher norm of the EXACT all-order
        # interaction aggregate G_u(S) = sum_{R subseteq S, R nonempty} I_{R u {u}}.
        # This is the right measure of "does u interact with S at all". The score comp_u(S)
        # expands as ||G_u||^2 + 2<dz_u, G_u>, so its magnitude is NOT an interaction strength:
        # the cross term can cancel the coupling norm, leaving comp = 0 with G_u != 0. Computing
        # couple costs no extra forward pass -- both features are already materialized here.
        couple = np.full(self.nU, np.nan)
        for u in head_set:
            lh = self._lh(u)
            su = self._feature(self._top_logits([lh]))         # vs dense
            single[u] = float((su ** 2).sum() / max(1, su.shape[0]))
            if u in seed_units:
                continue
            both_top = self._top_logits(seed + [lh])
            cu = self._feature(both_top, ref_top=base_seed_top,
                               probs_per_bucket=cond_probs)      # marginal vs seed-ablated
            cond[u] = float((cu ** 2).sum() / max(1, cu.shape[0]))
            gu = cu - su                                          # = G_u(S) in feature space
            couple[u] = float((gu ** 2).sum() / max(1, gu.shape[0]))
        comp = cond - single
        return {"single": single, "conditional": cond, "compensation": comp,
                "coupling": couple}

    def greedy_circuit(self, size: int, head_set: Optional[Sequence[int]] = None,
                       seed_heads: Optional[Sequence[Tuple[int, int]]] = None):
        """Discover a circuit by greedy co-ablation: repeatedly add the head whose MARGINAL
        Fisher energy (effect of ablating it GIVEN the already-selected set ablated) is largest.
        Because the score is conditional, the redundant heads that first-order saliency misses
        (compensating name-movers, backups) are picked up once their partners are in the set --
        so this recovers the full functional circuit, not just the high-saliency surface."""
        pool = list(range(self.nU)) if head_set is None else list(head_set)
        sel_units: List[int] = [C.head_index(l, h, self.nH) for (l, h) in (seed_heads or [])]
        picked: List[int] = []
        scores: List[float] = []
        for _ in range(min(size, len(pool))):
            sel_lh = [self._lh(u) for u in sel_units]
            base_top = self._top_logits(sel_lh) if sel_units else [t["tl"] for t in self.teacher]
            best_u, best_e = None, -1.0
            for u in pool:
                if u in sel_units:
                    continue
                cu = self._feature(self._top_logits(sel_lh + [self._lh(u)]), ref_top=base_top)
                e = float((cu ** 2).sum() / max(1, cu.shape[0]))
                if e > best_e:
                    best_e, best_u = e, u
            if best_u is None:
                break
            sel_units.append(best_u); picked.append(best_u); scores.append(best_e)
        return picked, scores

    def single_energy(self, head_set: Optional[Sequence[int]] = None) -> np.ndarray:
        """First-order saliency: unconditional Fisher energy of each unit (the baseline a greedy
        circuit-discovery must beat -- it misses self-repaired heads)."""
        pool = list(range(self.nU)) if head_set is None else list(head_set)
        out = np.full(self.nU, np.nan)
        for u in pool:
            out[u] = self._energy(self._top_logits([self._lh(u)]))
        return out
