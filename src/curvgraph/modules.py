"""Curvature-module discovery from the PSD co-ablation kernel.

A *curvature module* is the unit-support of a dominant eigen-mode of H. Because the
eigenvectors are SIGNED, a single module binds units that cooperate (same-sign loading)
and units that compensate (opposite-sign loading) into one coherent output-effect axis
-- the correct operationalization of a heterogeneous circuit, which a positive-affinity
cluster would split. A Laplacian route over the nonnegative affinity is provided only as
a cooperative-only comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ModuleAssignment:
    labels: np.ndarray            # [num_units] int in [0, K)
    masses: np.ndarray            # [K] module curvature mass (eigenvalue or within-module energy)
    route: str
    num_modules: int
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loadings: Optional[np.ndarray] = None   # [num_units, K] signed mode loadings (eigh route)


def select_num_modules_eigengap(
    eigenvalues: np.ndarray, k_min: int = 4, k_max: int = 64
) -> int:
    """Pick K by the largest relative eigengap in [k_min, k_max].

    eigenvalues are assumed sorted descending and non-negative. The relative gap at k is
    (lambda_k - lambda_{k+1}) / lambda_k; the K with the largest relative gap is the
    natural number of dominant curvature modes.
    """
    ev = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    ev = np.clip(ev, 0.0, None)
    n = ev.size
    hi = min(k_max, n - 1)
    lo = max(2, k_min)
    if hi <= lo:
        return max(lo, min(hi, n))
    best_k, best_gap = lo, -np.inf
    for k in range(lo, hi + 1):
        denom = ev[k - 1] if ev[k - 1] > 1e-30 else 1e-30
        gap = (ev[k - 1] - ev[k]) / denom
        if gap > best_gap:
            best_gap, best_k = gap, k
    return int(best_k)


def discover_modules_eigh(
    h: np.ndarray, num_modules: int
) -> ModuleAssignment:
    """Hard module assignment from the top-K signed eigen-modes of H.

    Unit u is assigned to the mode in which it carries the most curvature energy,
    lambda_m * phi_m[u]^2. Module mass is the mode eigenvalue lambda_m. The signed
    loadings are retained so we can characterize cooperating vs compensating membership.
    """
    h = 0.5 * (h + h.T)
    eigvals, eigvecs = np.linalg.eigh(h)         # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    k = int(max(1, min(num_modules, eigvals.size)))
    top_vals = eigvals[:k]
    top_vecs = eigvecs[:, :k]                     # [num_units, k]
    energy = (top_vecs ** 2) * top_vals[None, :]  # [num_units, k]
    labels = np.argmax(energy, axis=1).astype(np.int64)
    return ModuleAssignment(
        labels=labels,
        masses=top_vals.astype(np.float64),
        route="eigh",
        num_modules=k,
        eigenvalues=eigvals.astype(np.float64),
        loadings=top_vecs.astype(np.float64),
    )


def discover_modules_laplacian(
    affinity: np.ndarray, num_modules: int, seed: int = 0
) -> ModuleAssignment:
    """Spectral clustering of the nonnegative affinity A_+ (cooperative-only comparison)."""
    from sklearn.cluster import SpectralClustering

    a = np.clip(np.asarray(affinity, dtype=np.float64), 0.0, None)
    np.fill_diagonal(a, 1.0)
    k = int(max(2, min(num_modules, a.shape[0])))
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    )
    labels = sc.fit_predict(a).astype(np.int64)
    masses = within_module_mass(a, labels, k)
    return ModuleAssignment(
        labels=labels, masses=masses, route="laplacian", num_modules=k
    )


def within_module_mass(h: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Module mass = total within-module curvature sum_{u,v in M} H_uv (>=0 for PSD blocks)."""
    masses = np.zeros((k,), dtype=np.float64)
    for m in range(k):
        idx = np.where(labels == m)[0]
        if idx.size:
            block = h[np.ix_(idx, idx)]
            masses[m] = float(block.sum())
    return masses


def adjusted_rand(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(labels_a), np.asarray(labels_b)))


def module_summary(
    assignment: ModuleAssignment,
    unit_layers: Sequence[int],
    unit_types: Sequence[str],
) -> List[Dict[str, object]]:
    """Per-module descriptor: size, mass, layer span, attn/ffn mix, sign structure."""
    labels = assignment.labels
    out: List[Dict[str, object]] = []
    for m in range(assignment.num_modules):
        idx = np.where(labels == m)[0]
        if idx.size == 0:
            out.append({"module": m, "size": 0, "mass": float(assignment.masses[m])})
            continue
        layers = [int(unit_layers[u]) for u in idx]
        n_attn = int(sum(1 for u in idx if unit_types[u] == "attn_head"))
        sign_frac_pos = None
        if assignment.loadings is not None:
            load = assignment.loadings[idx, m]
            pos = int((load > 0).sum())
            sign_frac_pos = float(pos / max(1, load.size))
        out.append(
            {
                "module": int(m),
                "size": int(idx.size),
                "mass": float(assignment.masses[m]),
                "layer_min": int(min(layers)),
                "layer_max": int(max(layers)),
                "layer_span": int(max(layers) - min(layers)),
                "num_attn_units": n_attn,
                "num_ffn_units": int(idx.size - n_attn),
                "cross_type": bool(0 < n_attn < idx.size),
                "frac_cooperating": sign_frac_pos,        # vs compensating = 1 - this
            }
        )
    return out
