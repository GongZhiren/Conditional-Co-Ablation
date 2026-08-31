"""Falsifiable circuit validation for the curvature graph (GPT-2-class models).

Tests the central claim --- *curvature modules are circuits* --- against documented
mechanistic circuits, with pre-registered metrics and required-to-beat controls. All
attention internals are accessed with plain HuggingFace hooks / ``output_attentions``;
no external interpretability library is required.

Components:
  * head-level co-ablation curvature kernel H_head (single-head ablations -> Fisher Gram);
  * curvature modules over heads (signed eigen-modes, reused from curvgraph.modules);
  * induction-head detection (empirical induction score on repeated random sequences);
  * the documented GPT-2-small IOI circuit (Wang et al., 2022) as head-level ground truth;
  * circuit overlap (precision / recall / F1 / adjusted Rand) of curvature modules vs a
    circuit, vs two controls: activation-correlation clustering and random partition;
  * causal scrubbing: whole-module vs scattered-same-size ablation effect on the IOI
    logit difference (a module is a functional unit if its ablation is sharper).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from curvgraph._core.model import ModelBundle, bundle_device, layer_modules

from . import modules as mods


# Documented GPT-2-small IOI circuit, head groups as (layer, head). Source: Wang et al.,
# "Interpretability in the Wild: a Circuit for Indirect Object Identification" (2022).
IOI_CIRCUIT: Dict[str, List[Tuple[int, int]]] = {
    "name_mover": [(9, 9), (10, 0), (9, 6)],
    "negative_name_mover": [(10, 7), (11, 10)],
    "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token": [(0, 1), (0, 10), (3, 0)],
    "previous_token": [(2, 2), (4, 11)],
    "backup_name_mover": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
}


def ioi_circuit_heads() -> List[Tuple[int, int]]:
    seen: List[Tuple[int, int]] = []
    for group in IOI_CIRCUIT.values():
        for h in group:
            if h not in seen:
                seen.append(h)
    return seen


# ----------------------------------------------------------------- GPT-2 plumbing
def _attn_module(layer):
    # GPT-2: .attn ; Llama/Qwen/Mistral: .self_attn ; GPT-NeoX (Pythia): .attention.
    for name in ("attn", "self_attn", "attention"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    return None


_OUT_PROJ_NAMES = ("c_proj", "o_proj", "out_proj", "dense")


def _cproj(attn):
    """Attention OUTPUT projection across architectures; its input is the concatenated per-head
    attention output, so zeroing a head's slice of that input ablates exactly that head.
      GPT-2 c_proj | Llama/Qwen o_proj | GPT-NeoX dense | GPT-Neo (nested) attention.out_proj.
    """
    if attn is None:
        return None
    for name in _OUT_PROJ_NAMES:
        p = getattr(attn, name, None)
        if p is not None:
            return p
    inner = getattr(attn, "attention", None)  # GPT-Neo nests the real attention one level down
    if inner is not None:
        for name in _OUT_PROJ_NAMES:
            p = getattr(inner, name, None)
            if p is not None:
                return p
    return None


def head_index(layer: int, head: int, n_heads: int) -> int:
    return layer * n_heads + head


def head_layer_head(idx: int, n_heads: int) -> Tuple[int, int]:
    return idx // n_heads, idx % n_heads


def register_head_ablation(bundle: ModelBundle, layer_idx: int, head_idx: int,
                           replacement: Optional[torch.Tensor] = None):
    """Ablate one head by overwriting its slice of the attention-output-projection input.
    replacement=None -> zero ablation; otherwise set the slice to `replacement` (a [head_dim]
    vector, broadcast over batch/positions) -- e.g. the head's mean activation (mean ablation,
    the expectation of resample ablation). Robustness to this choice defuses Miller et al.
    (2407.08734), which shows faithfulness is sensitive to the ablation value."""
    layer = layer_modules(bundle)[layer_idx]
    attn = _attn_module(layer)
    cproj = _cproj(attn)
    if cproj is None:
        raise RuntimeError("attention output projection not found; head ablation unsupported "
                           "for this architecture (looked for c_proj/o_proj/out_proj/dense).")
    hd = bundle.head_dim
    start, end = head_idx * hd, (head_idx + 1) * hd

    def _hook(_module, inputs):
        if not inputs:
            return inputs
        hidden = inputs[0]
        if hidden.dim() != 3 or hidden.shape[-1] < end:
            return inputs
        masked = hidden.clone()
        if replacement is None:
            masked[..., start:end] = 0.0
        else:
            masked[..., start:end] = replacement.to(hidden.dtype).to(hidden.device)
        return (masked,) + tuple(inputs[1:])

    return cproj.register_forward_pre_hook(_hook)


def head_mean_vectors(bundle: ModelBundle, seqs: Sequence[torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
    """Mean of each head's attention-output-projection-input slice over all positions/sequences,
    for mean ablation. Returns {(layer, head): [head_dim] tensor}."""
    nH, hd = bundle.num_heads, bundle.head_dim
    layers = layer_modules(bundle)
    sums = {(li, hi): None for li in range(bundle.num_layers) for hi in range(nH)}
    count = [0]
    handles = []

    def mk(li):
        def hook(_m, inp):
            x = inp[0].detach()
            if x.dim() != 3:
                return
            flat = x.reshape(-1, x.shape[-1])
            count[0] += flat.shape[0]
            for hi in range(nH):
                s = flat[:, hi * hd:(hi + 1) * hd].sum(0).float().cpu()
                sums[(li, hi)] = s if sums[(li, hi)] is None else sums[(li, hi)] + s
        return hook

    for li in range(bundle.num_layers):
        cp = _cproj(_attn_module(layers[li]))
        if cp is not None:
            handles.append(cp.register_forward_pre_hook(mk(li)))
    try:
        for ids in seqs:
            _forward_logits(bundle, ids)
    finally:
        for h in handles:
            h.remove()
    n = max(1, count[0])
    return {k: (v / n) for k, v in sums.items() if v is not None}


def register_heads_ablation(bundle: ModelBundle, heads: Sequence[Tuple[int, int]],
                            replacements: Optional[Dict[Tuple[int, int], torch.Tensor]] = None):
    return [register_head_ablation(bundle, l, h,
                                   None if replacements is None else replacements.get((l, h)))
            for (l, h) in heads]


def register_heads_ablation_grouped(bundle: ModelBundle, heads: Sequence[Tuple[int, int]]):
    """Zero-ablate many heads with ONE hook per layer (zeroes all that layer's slices at once),
    instead of one hook per head. Equivalent output to register_heads_ablation(replacements=None)
    but O(#layers) hooks instead of O(#heads) -- critical for sequential pruning where the
    conditioning set grows to hundreds of heads. Returns handles to remove."""
    hd = bundle.head_dim
    by_layer: Dict[int, list] = {}
    for (l, h) in heads:
        by_layer.setdefault(int(l), []).append(int(h))
    layers = layer_modules(bundle)
    handles = []

    def mk(hids):
        spans = [(hi * hd, (hi + 1) * hd) for hi in hids]

        def _hook(_module, inputs):
            if not inputs:
                return inputs
            hidden = inputs[0]
            if hidden.dim() != 3:
                return inputs
            masked = hidden.clone()
            for s, e in spans:
                if hidden.shape[-1] >= e:
                    masked[..., s:e] = 0.0
            return (masked,) + tuple(inputs[1:])
        return _hook

    for li, hids in by_layer.items():
        cproj = _cproj(_attn_module(layers[li]))
        if cproj is not None:
            handles.append(cproj.register_forward_pre_hook(mk(hids)))
    return handles


def _forward_logits(bundle: ModelBundle, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = bundle.model(input_ids=input_ids, use_cache=False)
    return out.logits


# ------------------------------------------------------------- calibration sample
def _calibration_token_batches(
    bundle: ModelBundle, num_sequences: int, seq_len: int, seed: int
) -> List[torch.Tensor]:
    """Real text token windows from the shared calibration mix (generic LM text)."""
    from curvgraph._core.calibration import load_calibration_texts

    # Reuse the project's calibration config indirectly: read the mix jsonl directly so
    # this stays independent of the unit-pruning calibration knobs.
    path = Path("data/calibration/calibration_mix.jsonl")
    texts: List[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("text"):
                    texts.append(row["text"])
                if len(texts) >= num_sequences:
                    break
    device = bundle_device(bundle)
    batches: List[torch.Tensor] = []
    for t in texts[:num_sequences]:
        enc = bundle.tokenizer(t, return_tensors="pt", truncation=True, max_length=seq_len)
        ids = enc["input_ids"].to(device)
        if ids.shape[1] >= 8:
            batches.append(ids)
    return batches


def _centered_fisher_feature(
    teacher_top_logits: torch.Tensor,
    teacher_top_probs: torch.Tensor,
    student_top_logits: torch.Tensor,
) -> torch.Tensor:
    """Centered, Fisher-weighted logit-gap feature (same form as the core ablation)."""
    delta = teacher_top_logits - student_top_logits
    centered = delta - (teacher_top_probs * delta).sum(dim=-1, keepdim=True)
    return torch.sqrt(teacher_top_probs.clamp_min(1e-12)) * centered


# ----------------------------------------------------- head-level curvature kernel
def head_curvature_kernel(
    bundle: ModelBundle,
    num_sequences: int = 128,
    seq_len: int = 128,
    top_r: int = 256,
    seed: int = 42,
    sequences: Optional[List[torch.Tensor]] = None,
) -> np.ndarray:
    """H_head[i,j] = Fisher inner product of single-head ablation perturbations.

    One ablation forward per (head, sequence); features stacked over selected token
    positions and reduced to a Gram matrix, exactly as the unit-level kernel. If
    ``sequences`` (pre-tokenized [1,T] tensors) is given, the kernel is built on those
    *task-exercising* inputs instead of generic calibration text — so a dormant circuit
    becomes visible to the (output-effect) curvature.
    """
    n_layers, n_heads = bundle.num_layers, bundle.num_heads
    n_units = n_layers * n_heads
    batches = sequences if sequences is not None else _calibration_token_batches(bundle, num_sequences, seq_len, seed)
    if not batches:
        raise RuntimeError("No calibration text available for head kernel.")

    # Teacher top-r per sequence (predict next token => use positions :-1).
    teacher: List[Dict[str, torch.Tensor]] = []
    for ids in batches:
        logits = _forward_logits(bundle, ids)[:, :-1, :]
        probs = torch.softmax(logits, dim=-1)
        tp, ti = torch.topk(probs, k=min(top_r, probs.shape[-1]), dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        tl = torch.gather(logits, -1, ti)
        teacher.append({"ids": ids, "ti": ti, "tp": tp, "tl": tl})

    # Per-unit stacked features over all tokens.
    feats: List[List[np.ndarray]] = [[] for _ in range(n_units)]
    for li in range(n_layers):
        for hi in range(n_heads):
            uid = head_index(li, hi, n_heads)
            handle = register_head_ablation(bundle, li, hi)
            try:
                for tinfo in teacher:
                    s_logits = _forward_logits(bundle, tinfo["ids"])[:, :-1, :]
                    s_top = torch.gather(s_logits, -1, tinfo["ti"])
                    feat = _centered_fisher_feature(tinfo["tl"], tinfo["tp"], s_top)
                    feats[uid].append(feat.reshape(-1, feat.shape[-1]).float().cpu().numpy())
            finally:
                handle.remove()
    # Stack and Gram.
    stacked = [np.concatenate(f, axis=0) for f in feats]   # each [P, top_r]
    p = stacked[0].shape[0]
    flat = np.stack([s.reshape(-1) for s in stacked], axis=0)  # [n_units, P*top_r]
    h = (flat @ flat.T) / max(1, p)
    return 0.5 * (h + h.T)


# ------------------------------------------------------------- induction detection
def induction_scores(
    bundle: ModelBundle,
    seq_len: int = 50,
    num_sequences: int = 128,
    vocab_low: int = 1000,
    vocab_high: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Per-head induction score on repeated random sequences.

    A sequence is rand(T) concatenated with itself. In the second copy, an induction head
    attends from position p to the token AFTER the previous occurrence of token[p], i.e.
    to position p - T + 1. The score is that attention weight averaged over p and heads.
    """
    n_layers, n_heads = bundle.num_layers, bundle.num_heads
    device = bundle_device(bundle)
    rng = np.random.default_rng(seed)
    vocab_high = min(vocab_high, int(bundle.tokenizer.vocab_size))
    acc = np.zeros((n_layers, n_heads), dtype=np.float64)
    count = 0
    for _ in range(num_sequences):
        half = rng.integers(vocab_low, vocab_high, size=seq_len)
        seq = np.concatenate([half, half])
        ids = torch.tensor(seq[None, :], dtype=torch.long, device=device)
        with torch.no_grad():
            out = bundle.model(input_ids=ids, use_cache=False, output_attentions=True)
        attns = out.attentions  # tuple[n_layers] of [1, n_heads, 2T, 2T]
        # second-copy query positions p in [T+1, 2T-1] attend to p-T+1
        for li in range(n_layers):
            a = attns[li][0]  # [n_heads, 2T, 2T]
            for p in range(seq_len + 1, 2 * seq_len):
                tgt = p - seq_len + 1
                acc[li] += a[:, p, tgt].float().cpu().numpy()
        count += (2 * seq_len - 1) - (seq_len + 1)
    scores = acc / max(1, count)
    return scores.reshape(-1)  # [n_layers*n_heads]


# --------------------------------------------------------------------- IOI task
_IOI_NAMES = ["John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Anna",
              "Paul", "Kate", "Mark", "Lucy", "Peter", "Sarah", "David", "Emma"]
_IOI_PLACES = ["store", "park", "school", "office", "garden", "station", "market", "library"]
_IOI_OBJECTS = ["drink", "book", "ball", "ring", "kiss", "snack", "bone", "note"]


def ioi_prompts(num_prompts: int = 128, seed: int = 42) -> List[Dict[str, str]]:
    rng = np.random.default_rng(seed)
    out: List[Dict[str, str]] = []
    for _ in range(num_prompts):
        io, s = rng.choice(_IOI_NAMES, size=2, replace=False)
        place = rng.choice(_IOI_PLACES)
        obj = rng.choice(_IOI_OBJECTS)
        # ABB structure: IO and S go; S gives object to -> IO
        prompt = f"When {io} and {s} went to the {place}, {s} gave a {obj} to"
        out.append({"prompt": prompt, "io": " " + io, "s": " " + s})
    return out


_GT_NOUNS = ["war", "siege", "reign", "feud", "voyage", "famine", "plague", "drought", "trial",
             "revolt", "boom", "tour", "study", "project", "dynasty", "conflict", "epidemic",
             "expedition", "campaign", "festival", "strike", "blockade", "renovation", "exhibition"]


def greater_than_prompts(num_prompts: int = 96, seed: int = 42):
    """Greater-than circuit prompts (Hanna et al. 2023): 'The {noun} lasted from the year 17YY to
    the year 17'. The model should put more mass on completions > YY. Used here as a NEGATIVE
    CONTROL -- a documented circuit that is MLP-dominated with no documented self-repair."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_prompts):
        noun = rng.choice(_GT_NOUNS)
        yy = int(rng.integers(2, 98))
        prompt = f"The {noun} lasted from the year 17{yy:02d} to the year 17"
        out.append({"prompt": prompt, "yy": yy})
    return out


# Documented greater-than circuit attention heads (Hanna et al. 2023): a9.h1 is the direct-logit
# head; the rest feed the greater-than MLPs (8-11). MLP-dominated, no documented backup structure.
GREATER_THAN_HEADS = [(9, 1), (8, 11), (8, 8), (7, 10), (6, 9), (5, 5), (5, 1)]


def ioi_token_sequences(
    bundle: ModelBundle, num_prompts: int = 128, seed: int = 42
) -> List[torch.Tensor]:
    """Tokenized IOI prompts as [1, T] tensors, to build a *task-exercised* curvature
    kernel (head_curvature_kernel(sequences=...)). Generic calibration text leaves the
    task circuits (S-inhibition, previous-token) dormant; running the kernel on the task
    that exercises them makes them visible to the output-effect curvature."""
    prompts = ioi_prompts(num_prompts, seed)
    device = bundle_device(bundle)
    seqs: List[torch.Tensor] = []
    for ex in prompts:
        ids = bundle.tokenizer(ex["prompt"], return_tensors="pt").to(device)["input_ids"]
        if ids.shape[1] >= 4:
            seqs.append(ids)
    return seqs


def ioi_logit_diff(
    bundle: ModelBundle,
    prompts: Sequence[Dict[str, str]],
    ablate_heads: Optional[Sequence[Tuple[int, int]]] = None,
) -> float:
    """Mean (logit[IO] - logit[S]) at the final position; higher = task intact."""
    device = bundle_device(bundle)
    handles = register_heads_ablation(bundle, ablate_heads) if ablate_heads else []
    diffs: List[float] = []
    try:
        for ex in prompts:
            ids = bundle.tokenizer(ex["prompt"], return_tensors="pt").to(device)["input_ids"]
            io_id = bundle.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
            s_id = bundle.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
            logits = _forward_logits(bundle, ids)[0, -1, :]
            diffs.append(float(logits[io_id].item() - logits[s_id].item()))
    finally:
        for h in handles:
            h.remove()
    return float(np.mean(diffs)) if diffs else 0.0


# --------------------------------------------------------- activation-correlation control
def activation_correlation_kernel(
    bundle: ModelBundle, num_sequences: int = 128, seq_len: int = 128, seed: int = 42
) -> np.ndarray:
    """Per-head co-activation correlation (the control the curvature graph must beat)."""
    n_layers, n_heads, hd = bundle.num_layers, bundle.num_heads, bundle.head_dim
    n_units = n_layers * n_heads
    batches = _calibration_token_batches(bundle, num_sequences, seq_len, seed)
    captured: Dict[int, List[np.ndarray]] = {u: [] for u in range(n_units)}
    handles = []
    layers = layer_modules(bundle)

    def _make_hook(li: int):
        def _hook(_m, inputs):
            hidden = inputs[0]
            if hidden.dim() != 3:
                return
            x = hidden.detach()
            for hi in range(n_heads):
                sl = x[..., hi * hd:(hi + 1) * hd]
                captured[head_index(li, hi, n_heads)].append(
                    sl.norm(dim=-1).reshape(-1).float().cpu().numpy()
                )
        return _hook

    for li in range(n_layers):
        cproj = _cproj(_attn_module(layers[li]))
        if cproj is not None:
            handles.append(cproj.register_forward_pre_hook(_make_hook(li)))
    try:
        for ids in batches:
            _forward_logits(bundle, ids)
    finally:
        for h in handles:
            h.remove()
    mat = np.stack([np.concatenate(captured[u]) for u in range(n_units)], axis=0)  # [units, tokens]
    corr = np.corrcoef(mat)
    return np.nan_to_num(corr)


# ----------------------------------------------------------------- overlap metrics
def circuit_overlap(
    labels: np.ndarray, circuit_units: Sequence[int], num_modules: int
) -> Dict[str, float]:
    """Best-module precision/recall/F1 vs a circuit head-set, plus binary adjusted Rand."""
    circuit = set(int(x) for x in circuit_units)
    n = labels.shape[0]
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "module": -1}
    for m in range(num_modules):
        members = set(int(u) for u in np.where(labels == m)[0])
        if not members:
            continue
        tp = len(members & circuit)
        prec = tp / max(1, len(members))
        rec = tp / max(1, len(circuit))
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if f1 > best["f1"]:
            best = {"precision": prec, "recall": rec, "f1": f1, "module": m}
    # binary partition adjusted Rand: {in best module} vs {in circuit}
    in_circuit = np.array([1 if u in circuit else 0 for u in range(n)])
    in_best = np.array([1 if labels[u] == best["module"] else 0 for u in range(n)])
    ari = mods.adjusted_rand(in_circuit, in_best)
    return {**best, "binary_adjusted_rand": ari}


def random_partition(n_units: int, num_modules: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, num_modules, size=n_units)


def mode_for_circuit(
    h_head: np.ndarray, circuit_units: Sequence[int], n_units: int, k_modes: int = 32
) -> Optional[Dict[str, object]]:
    """Eigen-mode of H whose signed |loading| best ranks circuit membership (ROC-AUC).

    Identifies *the single coherent curvature mode that encodes circuit C*, instead of a
    coarse K-way partition cell. Returns the mode index, its membership-ranking AUC, the
    signed loadings, and the eigenvalue (curvature mass).
    """
    from sklearn.metrics import roc_auc_score

    h = 0.5 * (h_head + h_head.T)
    eigvals, eigvecs = np.linalg.eigh(h)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    mem = np.zeros(n_units, dtype=bool)
    mem[[int(u) for u in circuit_units if 0 <= int(u) < n_units]] = True
    if not mem.any() or mem.all():
        return None
    best_mode, best_auc, best_load, best_val = -1, -1.0, None, 0.0
    for m in range(min(k_modes, n_units)):
        load = np.abs(eigvecs[:, m])
        try:
            auc = float(roc_auc_score(mem.astype(int), load))
        except ValueError:
            continue
        if auc > best_auc:
            best_mode, best_auc, best_load, best_val = m, auc, eigvecs[:, m].copy(), float(eigvals[m])
    if best_load is None:
        return None
    return {"mode": int(best_mode), "mode_auc": best_auc, "loading": best_load, "eigenvalue": best_val}


def movement_head_scores(
    bundle: ModelBundle,
    seq_len: int = 48,
    num_sequences: int = 64,
    vocab_low: int = 1000,
    vocab_high: int = 10000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Per-head attention-offset scores for the three empirically-detectable MOVEMENT circuits,
    computed in a single forward over repeated-random sequences (rand(T)+rand(T)):

      previous_token : attends to p-1                (mean over all query positions)
      duplicate_token: in the 2nd copy, attends to p-T   (the SAME token's earlier occurrence)
      induction      : in the 2nd copy, attends to p-T+1 (the token AFTER that occurrence)

    These give a model-agnostic ground truth for a vs-active functional-specificity test
    (can a signal separate one movement sub-circuit from the others?) without needing IOI
    head-level labels, so the GPT-2-small finding can be replicated cross-scale."""
    n_layers, n_heads = bundle.num_layers, bundle.num_heads
    device = bundle_device(bundle)
    rng = np.random.default_rng(seed)
    vocab_high = min(vocab_high, int(bundle.tokenizer.vocab_size))
    prev = np.zeros((n_layers, n_heads)); dup = np.zeros((n_layers, n_heads)); ind = np.zeros((n_layers, n_heads))
    n_prev = n_sec = 0
    for _ in range(num_sequences):
        half = rng.integers(vocab_low, vocab_high, size=seq_len)
        seq = np.concatenate([half, half])
        ids = torch.tensor(seq[None, :], dtype=torch.long, device=device)
        with torch.no_grad():
            attns = bundle.model(input_ids=ids, use_cache=False, output_attentions=True).attentions
        for li in range(n_layers):
            a = attns[li][0].float().cpu().numpy()  # [n_heads, 2T, 2T]
            for p in range(1, 2 * seq_len):
                prev[li] += a[:, p, p - 1]
            for p in range(seq_len + 1, 2 * seq_len):
                dup[li] += a[:, p, p - seq_len]
                ind[li] += a[:, p, p - seq_len + 1]
        n_prev += (2 * seq_len - 1)
        n_sec += (2 * seq_len - 1) - (seq_len + 1) + 1
    return {"previous_token": (prev / max(1, n_prev)).reshape(-1),
            "duplicate_token": (dup / max(1, n_sec)).reshape(-1),
            "induction": (ind / max(1, n_sec)).reshape(-1)}


def repeated_random_sequences(
    bundle: ModelBundle,
    num_sequences: int = 64,
    seq_len: int = 48,
    vocab_low: int = 1000,
    vocab_high: int = 10000,
    seed: int = 42,
) -> List[torch.Tensor]:
    """rand(T) concatenated with itself, as [1, 2T] id tensors. This is the distribution the
    induction heads actually fire on; co-ablation features for the induction circuit MUST be
    computed here (not on IOI prompts, where the detected induction heads are near-silent and
    every signal collapses to chance)."""
    device = bundle_device(bundle)
    rng = np.random.default_rng(seed)
    vocab_high = min(vocab_high, int(bundle.tokenizer.vocab_size))
    seqs: List[torch.Tensor] = []
    for _ in range(num_sequences):
        half = rng.integers(vocab_low, vocab_high, size=seq_len)
        seq = np.concatenate([half, half])
        seqs.append(torch.tensor(seq[None, :], dtype=torch.long, device=device))
    return seqs


def induction_behavior(
    bundle: ModelBundle,
    ablate_heads: Optional[Sequence[Tuple[int, int]]] = None,
    seq_len: int = 50,
    num_sequences: int = 32,
    vocab_low: int = 1000,
    vocab_high: int = 10000,
    seed: int = 42,
) -> float:
    """Behavioral readout for the induction circuit: mean log-prob of the correct (repeated)
    token at second-copy positions on repeated-random sequences. Higher = in-context copying
    works. This is the readout matched to induction (the IOI logit-diff is insensitive to the
    early induction heads, so it is the wrong probe for that circuit)."""
    device = bundle_device(bundle)
    rng = np.random.default_rng(seed)
    vocab_high = min(vocab_high, int(bundle.tokenizer.vocab_size))
    handles = register_heads_ablation(bundle, ablate_heads) if ablate_heads else []
    lps: List[float] = []
    try:
        for _ in range(num_sequences):
            half = rng.integers(vocab_low, vocab_high, size=seq_len)
            seq = np.concatenate([half, half])
            ids = torch.tensor(seq[None, :], dtype=torch.long, device=device)
            logits = _forward_logits(bundle, ids)[0]               # [2T, V]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            # predict token[p+1] from position p; in the second copy the target repeats.
            for p in range(seq_len, 2 * seq_len - 1):
                tgt = int(seq[p + 1])
                lps.append(float(logprobs[p, tgt].item()))
    finally:
        for h in handles:
            h.remove()
    return float(np.mean(lps)) if lps else 0.0


def _scattered_drops(
    behavior_fn, clean: float, candidate_units: np.ndarray,
    size: int, n_samples: int, n_heads: int, rng,
) -> List[float]:
    drops: List[float] = []
    if candidate_units.size < size:
        return drops
    for _ in range(n_samples):
        pick = rng.choice(candidate_units, size=size, replace=False)
        heads = [head_layer_head(int(u), n_heads) for u in pick]
        drops.append(clean - behavior_fn(heads))
    return drops


def causal_scrubbing_mode(
    bundle: ModelBundle,
    behavior_fn,
    h_head: np.ndarray,
    circuit_units: Sequence[int],
    n_units: int,
    n_heads: int,
    clean: float,
    n_scattered: int = 16,
    seed: int = 42,
) -> Optional[Dict[str, object]]:
    """Falsification condition (b), redesigned to be discriminative.

    The *curvature module* for a (homogeneous) circuit C is the top-|C| heads of the single
    eigen-mode best aligned to C -- small and tied to ONE coherent mode, not the F1-best
    coarse partition cell (which is large, heterogeneous, and was not discriminative). We
    test that ablating this module damages the IOI logit-diff more than two controls:
      * random scattered heads of the same size;
      * saliency-matched scattered heads (highest individual self-energy H_uu, low mutual
        affinity) -- so a positive result is NOT merely 'these are individually important'.
    We also sweep the ablation size j=1..|module| for module (by |loading|) vs random, to
    expose a sharp / near-all-or-nothing threshold (the signature of a functional unit).
    """
    info = mode_for_circuit(h_head, circuit_units, n_units)
    if info is None:
        return None
    rng = np.random.default_rng(seed)
    diag = np.diag(h_head).copy()
    load = np.abs(np.asarray(info["loading"]))
    s = max(1, len(list(circuit_units)))
    module_units = np.argsort(load)[::-1][:s]
    module_heads = [head_layer_head(int(u), n_heads) for u in module_units]
    module_drop = clean - behavior_fn(module_heads)

    # Control 1: random scattered, same size.
    all_units = np.arange(n_units)
    rand_drops = _scattered_drops(behavior_fn, clean, all_units, s, n_scattered, n_heads, rng)

    # Control 2: saliency-matched scattered -- candidates are the high-self-energy heads
    # OUTSIDE the module (so equal-or-higher individual saliency, but not mode-coherent).
    order_sal = np.argsort(diag)[::-1]
    high_sal = np.array([u for u in order_sal if u not in set(int(x) for x in module_units)])
    high_sal = high_sal[: max(s, 3 * s)]
    sal_drops = _scattered_drops(behavior_fn, clean, high_sal, s, n_scattered, n_heads, rng)

    # Sharpness sweep: top-j of module (by loading) vs random j.
    sweep = []
    for j in range(1, s + 1):
        mod_j = [head_layer_head(int(u), n_heads) for u in module_units[:j]]
        d_mod = clean - behavior_fn(mod_j)
        rnd = _scattered_drops(behavior_fn, clean, all_units, j, max(4, n_scattered // 2), n_heads, rng)
        sweep.append({"j": j, "module_drop": float(d_mod),
                      "random_drop_mean": float(np.mean(rnd)) if rnd else None})

    rand_mean = float(np.mean(rand_drops)) if rand_drops else 0.0
    sal_mean = float(np.mean(sal_drops)) if sal_drops else None
    return {
        "mode": info["mode"],
        "mode_auc": info["mode_auc"],
        "module_size": int(s),
        "module_units": [list(head_layer_head(int(u), n_heads)) for u in module_units],
        "module_ablation_drop": float(module_drop),
        "scattered_random_drop_mean": rand_mean,
        "scattered_random_drop_std": float(np.std(rand_drops)) if rand_drops else 0.0,
        "scattered_saliency_matched_drop_mean": sal_mean,
        "sharper_than_random": bool(module_drop > rand_mean),
        "sharper_than_saliency_matched": (None if sal_mean is None else bool(module_drop > sal_mean)),
        "sharpness_sweep": sweep,
    }


def circuit_affinity_auc(affinity: np.ndarray, circuit_units: Sequence[int], n_units: int):
    """Partition-free circuit-recovery metric: pair-level ROC-AUC of 'same-circuit'
    predicted by the affinity. This is the correct test of whether the graph ENCODES the
    circuit (independent of any global K-way partition); best-module F1 conflates encoding
    with an arbitrary partition and penalizes heterogeneous / redundant circuits.
    """
    from sklearn.metrics import roc_auc_score
    mem = np.zeros(n_units, dtype=bool)
    mem[[int(u) for u in circuit_units if 0 <= int(u) < n_units]] = True
    iu = np.triu_indices(n_units, 1)
    same = mem[iu[0]] & mem[iu[1]]
    vals = affinity[iu]
    if not same.any() or not (~same).any():
        return None
    return float(roc_auc_score(same.astype(int), vals))
