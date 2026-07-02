r"""Conditional Co-Ablation (CoAx).

Every effect is read in the output distribution's own metric. Ablating a unit changes the
clean logits :math:`z_0` to :math:`z_M`; on the clean top-:math:`r` support we form the
**centered, Fisher-weighted** feature

.. math::   \widetilde{\delta z} \;=\; \sqrt{p_0}\,\odot\,(\delta z - \mathbb{E}_{p_0}[\delta z]\mathbf 1),

so that :math:`\langle\widetilde{\delta z}_u,\widetilde{\delta z}_v\rangle=\delta z_u^\top F\,\delta z_v`
is a Fisher inner product (:math:`F=\mathrm{diag}(p_0)-p_0p_0^\top`). The **energy**
:math:`\mathcal E(\delta z)=\mathbb E_{x,t}\|\widetilde{\delta z}\|^2` is the mean KL cost of the
ablation. Two quantities follow:

* ``single_energy`` --- first-order saliency, :math:`\mathcal E(\delta z_u)`. Dormant backups
  score *low*, which is exactly why single-ablation scoring misses them.
* ``conditional_compensation`` --- the CoAx score, the **growth** of a unit's ablation energy
  once a primary seed :math:`S` is removed,
  :math:`\mathrm{comp}_u(S)=\mathcal E(\delta z_u\mid S)-\mathcal E(\delta z_u)`.
  Large for a backup (silent alone, load-bearing once its primary is gone), near zero otherwise.

``pairwise_synergy`` exposes the symmetric second-order interaction :math:`I_{uv}` directly; the
conditional score recovers the same backup-revealing signal at :math:`O(|\mathcal U|)` forward
passes instead of :math:`O(|\mathcal U|^2)`.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .model import Model


class CoAx:
    """Fisher-metric co-ablation scores over a frozen model's attention heads."""

    def __init__(self, model: Model, prompts: Sequence[str], top_r: int = 192):
        self.model = model
        seqs = model.encode(prompts)
        # Bucket by length so each bucket is one batched forward (rows are independent under
        # the causal mask, so batched logits are identical to a per-prompt loop).
        buckets: Dict[int, List[torch.Tensor]] = {}
        for ids in seqs:
            buckets.setdefault(int(ids.shape[1]), []).append(ids)
        self.teacher: List[Dict[str, torch.Tensor]] = []
        for T in sorted(buckets):
            ids = torch.cat(buckets[T], dim=0).to(model.device)
            logits = model.logits(ids)[:, :-1, :]                 # predict-next at each position
            tl, ti = torch.topk(logits, k=min(top_r, logits.shape[-1]), dim=-1)
            tp = torch.softmax(tl, dim=-1)                        # clean p0 on the top-r support
            self.teacher.append({"ids": ids, "ti": ti, "tl": tl, "tp": tp})

    # ------------------------------------------------------------------ core
    def _gather(self, ablate: Sequence[Tuple[int, int]]) -> List[torch.Tensor]:
        """Ablated logits gathered onto each bucket's fixed clean top-r index set."""
        out = []
        for t in self.teacher:
            s = self.model.logits(t["ids"], ablate=ablate)[:, :-1, :]
            out.append(torch.gather(s, -1, t["ti"]))
        return out

    def _energy(self, student_top: List[torch.Tensor],
                reference: Optional[List[torch.Tensor]] = None) -> float:
        """Mean centered Fisher energy of (reference - student). ``reference=None`` uses the
        clean teacher logits (unconditional effect); pass seed-ablated logits for a conditional
        effect."""
        total, rows = 0.0, 0
        for i, t in enumerate(self.teacher):
            ref = t["tl"] if reference is None else reference[i]
            d = ref - student_top[i]
            d = d - (t["tp"] * d).sum(-1, keepdim=True)           # center against p0
            d = torch.sqrt(t["tp"].clamp_min(1e-12)) * d          # Fisher weighting
            total += float((d.float() ** 2).sum().item())
            rows += d.shape[0] * d.shape[1]
        return total / max(1, rows)

    # ------------------------------------------------------------------ first order
    def single_energy(self, units: Optional[Sequence[int]] = None) -> np.ndarray:
        """First-order saliency :math:`\\mathcal E(\\delta z_u)` for each unit."""
        units = range(self.model.num_units) if units is None else units
        out = np.full(self.model.num_units, np.nan)
        for u in units:
            out[u] = self._energy(self._gather([self.model.layer_head(u)]))
        return out

    # ------------------------------------------------------------------ second order
    def conditional_compensation(self, seed: Sequence[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """CoAx score for every unit given a primary ``seed``.

        Returns ``single`` (unconditional energy), ``conditional`` (energy given the seed
        ablated), and ``compensation`` = conditional - single (the CoAx score). The
        seed-ablated reference is computed once and shared, so the whole scan is
        :math:`O(|\\mathcal U|)` forward passes.
        """
        seed = list(seed)
        seed_units = {self.model.head_index(l, h) for (l, h) in seed}
        seed_reference = self._gather(seed)                       # reference = seed ablated (once)
        single = np.full(self.model.num_units, np.nan)
        conditional = np.full(self.model.num_units, np.nan)
        for u in range(self.model.num_units):
            lh = self.model.layer_head(u)
            single[u] = self._energy(self._gather([lh]))          # effect of u alone (vs clean)
            if u in seed_units:
                continue
            conditional[u] = self._energy(self._gather(seed + [lh]), reference=seed_reference)
        return {"single": single, "conditional": conditional, "compensation": conditional - single}

    def pairwise_synergy(self, units: Sequence[int], normalize: bool = True) -> np.ndarray:
        r"""Symmetric second-order interaction matrix
        :math:`S_{uv}=\mathbb E_{x,t}\|\widetilde{I}_{uv}\|^2` with
        :math:`I_{uv}=\delta z_{uv}-\delta z_u-\delta z_v` over ``units`` (:math:`O(m^2)`)."""
        units = list(units)
        m = len(units)
        single_feat = [self._features(self._gather([self.model.layer_head(u)])) for u in units]
        energy = np.array([float((f ** 2).mean(0).sum()) for f in single_feat])
        S = np.zeros((m, m))
        for a, b in combinations(range(m), 2):
            pair = self._features(self._gather([self.model.layer_head(units[a]),
                                                self.model.layer_head(units[b])]))
            inter = pair - single_feat[a] - single_feat[b]
            S[a, b] = S[b, a] = float((inter ** 2).mean(0).sum())
        if normalize:
            S = S / (np.sqrt(np.outer(energy, energy)) + 1e-12)
        return S

    # ------------------------------------------------------------------ wake-up signature
    def _head_norms(self, ablate: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Mean attention-output norm of every head (summed over positions/prompts), measured
        under the given ablation. Captured from the output-projection input."""
        nH, hd = self.model.num_heads, self.model.head_dim
        total = np.zeros(self.model.num_units)
        count = 0
        handles = []

        def mk(li):
            def hook(_m, inp):
                x = inp[0]
                if x.dim() != 3:
                    return
                for hi in range(nH):
                    n = x[..., hi * hd:(hi + 1) * hd].norm(dim=-1).sum().item()
                    total[self.model.head_index(li, hi)] += float(n)
            return hook

        for li in range(self.model.num_layers):
            handles.append(self.model._out_proj(li).register_forward_pre_hook(mk(li)))
        try:
            for t in self.teacher:
                self.model.logits(t["ids"], ablate=ablate)
                count += t["ids"].shape[0] * t["ids"].shape[1]
        finally:
            for h in handles:
                h.remove()
        return total / max(1, count)

    def wakeup_ratios(self, seed: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Per-head output-norm ratio ``||output | seed ablated|| / ||output | clean||``. A dormant
        backup *wakes up* (ratio > 1) once its primary is gone; most heads stay near 1."""
        clean = self._head_norms([])
        ablated = self._head_norms(list(seed))
        return ablated / np.maximum(clean, 1e-9)

    def _features(self, student_top: List[torch.Tensor],
                  reference: Optional[List[torch.Tensor]] = None) -> np.ndarray:
        """Stacked centered Fisher feature rows ``[num_positions, top_r]`` (for synergy)."""
        feats = []
        for i, t in enumerate(self.teacher):
            ref = t["tl"] if reference is None else reference[i]
            d = ref - student_top[i]
            d = d - (t["tp"] * d).sum(-1, keepdim=True)
            d = torch.sqrt(t["tp"].clamp_min(1e-12)) * d
            feats.append(d.reshape(-1, d.shape[-1]).float().cpu().numpy())
        return np.concatenate(feats, axis=0)
