#!/usr/bin/env python3
"""Held-out, equal-budget circuit completion across model families.

  DETECT      sequences drawn with seed_detect  -- find the induction primaries, nothing else
  CALIBRATE   sequences drawn with seed_calib   -- every selector receives the same data budget
  EVALUATE    sequences drawn with seed_eval    -- held out; the completion drop is measured here

The three roles never share a sequence. Full score vectors are saved so multiple
top-k completion sizes can be evaluated without rescoring the selectors.

Gradient baselines retain activation gradients while model parameters stay frozen,
avoiding parameter-gradient buffers without changing the attribution scores.

  CUDA_VISIBLE_DEVICES=0 python experiments/paper/cross_model_completion.py \
      --model-key pythia-410m --n-calib 16 --n-eval 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from curvgraph._core.config import load_config
from curvgraph._core.model import load_model_bundle, layer_modules
from curvgraph import circuits as C
from curvgraph.coablation import CoAblation
from curvgraph import baselines as B

def induction_metric(bundle, seqs, seq_len, ablate_heads=None):
    """Mean log-probability of the copied token at second-copy positions."""
    handles = C.register_heads_ablation_grouped(bundle, ablate_heads) if ablate_heads else []
    values = []
    try:
        for ids in seqs:
            logits = C._forward_logits(bundle, ids)[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            tokens = ids[0]
            for pos in range(seq_len, 2 * seq_len - 1):
                values.append(float(log_probs[pos, int(tokens[pos + 1])]))
    finally:
        for handle in handles:
            handle.remove()
    return float(np.mean(values)) if values else 0.0


def induction_logprob_metric(bundle, seq_len):
    """Differentiable version of :func:`induction_metric` for gradient baselines."""
    def metric(bundle_, example):
        ids = example["ids"]
        outputs = bundle_.model(input_ids=ids, use_cache=False)
        log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)
        tokens = ids[0]
        positions = range(seq_len, ids.shape[1] - 1)
        return torch.stack([log_probs[pos, tokens[pos + 1]] for pos in positions]).mean()

    return metric


@torch.no_grad()
def _head_activation_matrix(bundle, seqs):
    """Return a [heads, tokens] output-norm trace for co-activation scoring."""
    n_heads, head_dim = bundle.num_heads, bundle.head_dim
    n_units = bundle.num_layers * n_heads
    rows = {unit: [] for unit in range(n_units)}
    layers = layer_modules(bundle)
    handles = []

    def make_hook(layer_index):
        def hook(_module, inputs):
            x = inputs[0]
            if x.dim() != 3:
                return
            x = x.detach()
            for head_index in range(n_heads):
                unit = C.head_index(layer_index, head_index, n_heads)
                start = head_index * head_dim
                rows[unit].append(
                    x[..., start:start + head_dim].norm(dim=-1).reshape(-1).float().cpu().numpy()
                )
        return hook

    for layer_index in range(bundle.num_layers):
        projection = C._cproj(C._attn_module(layers[layer_index]))
        if projection is not None:
            handles.append(projection.register_forward_pre_hook(make_hook(layer_index)))
    try:
        for ids in seqs:
            bundle.model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return np.stack([np.concatenate(rows[u]) if rows[u] else np.zeros(1)
                     for u in range(n_units)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-key", default="pythia-410m")
    ap.add_argument("--model-path", default=None,
                    help="optional local checkpoint path; overrides configs/model.yaml")
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--n-detect", type=int, default=32)
    ap.add_argument("--n-calib", type=int, default=16, help="identical budget for every selector")
    ap.add_argument("--n-eval", type=int, default=64, help="held-out completion evaluation")
    ap.add_argument("--seed-detect", type=int, default=101)
    ap.add_argument("--seed-calib", type=int, default=202)
    ap.add_argument("--seed-eval", type=int, default=303)
    ap.add_argument("--n-primary", type=int, default=4)
    ap.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--top-r", type=int, default=None,
                    help="fixed clean support; default comes from configs/model.yaml (0 = full)")
    ap.add_argument("--skip-grad", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["models"][args.model_key]["path"] = args.model_path
    model_spec = cfg["model"]["models"][args.model_key]
    bundle = load_model_bundle(model_spec,
                               cfg["model"].get("tokenizer", {}))
    bundle.model.requires_grad_(False)
    bundle.model.enable_input_require_grads()
    nH = bundle.num_heads
    nU = bundle.num_layers * nH
    lh = lambda u: C.head_layer_head(int(u), nH)
    seqs_of = lambda n, sd: C.repeated_random_sequences(bundle, num_sequences=n,
                                                        seq_len=args.seq_len, seed=sd)

    # ---- 1. DETECT: the primary set, on its own sequences ---------------------------------------
    isc = C.induction_scores(bundle, num_sequences=args.n_detect, seed=args.seed_detect)
    if args.model_key == "gpt2-small":
        primaries = [tuple(x) for x in C.IOI_CIRCUIT["induction"]]
        prim_u = [C.head_index(l, h, nH) for (l, h) in primaries]
    else:
        prim_u = [int(u) for u in np.argsort(isc)[::-1][: args.n_primary]]
        primaries = [lh(u) for u in prim_u]
    prim_set = set(prim_u)
    cand = [u for u in range(nU) if u not in prim_set]

    # ---- 2. CALIBRATE: every selector on the same held-in sequences, same count -------------------
    cal = seqs_of(args.n_calib, args.seed_calib)
    configured_top_r = int(model_spec.get("cross_model_top_r", 0))
    requested_top_r = configured_top_r if args.top_r is None else args.top_r
    top_r = requested_top_r or int(bundle.tokenizer.vocab_size)
    co = CoAblation(bundle, cal, top_r=top_r, position_mode="last")
    r = co.conditional_compensation(primaries, head_set=list(range(nU)))
    single, cond = r["single"], r["conditional"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(single > 0, cond / np.maximum(single, 1e-30), np.nan)
    scores = {"coax": r["compensation"], "cond_energy": cond, "ratio": ratio,
              "coupling": r["coupling"], "intact": single}

    act = _head_activation_matrix(bundle, cal)
    Z = (act - act.mean(1, keepdims=True)) / np.maximum(act.std(1, keepdims=True), 1e-9)
    corr = (Z @ Z.T) / max(1, Z.shape[1])
    scores["coact"] = np.array([np.nanmean([abs(corr[u, p]) for p in prim_u])
                                if u in cand else np.nan for u in range(nU)])

    if not args.skip_grad:
        met = induction_logprob_metric(bundle, args.seq_len)
        gp = [{"ids": s} for s in cal]                      # the SAME calibration sequences
        scores["atp"] = B.head_attribution_patching(bundle, gp, metric=met)
        print(f"[fair] ATP done on {len(gp)} calibration sequences", flush=True)
        scores["atpstar"] = B.head_attribution_graddrop(bundle, gp, metric=met)
        print(f"[fair] AtP* done on {len(gp)} calibration sequences", flush=True)

    # ---- 3. EVALUATE: held-out sequences, never seen by any selector ------------------------------
    ev = seqs_of(args.n_eval, args.seed_eval)
    clean = induction_metric(bundle, ev, args.seq_len)
    m_prim = induction_metric(bundle, ev, args.seq_len, ablate_heads=primaries)
    drop_prim = clean - m_prim
    print(f"[fair] {args.model_key}  clean={clean:.4f}  primaries={[list(p) for p in primaries]}  "
          f"primary-only drop={drop_prim:.4f}  (detect {args.n_detect} / calib {args.n_calib} / "
          f"eval {args.n_eval} sequences, disjoint)", flush=True)

    def completion(units):
        return float(clean - induction_metric(bundle, ev, args.seq_len,
                                              ablate_heads=primaries + [lh(u) for u in units]))

    rng = np.random.default_rng(args.seed_eval)
    order = {n: sorted(cand, key=lambda u: -(s[u] if np.isfinite(s[u]) else -np.inf))
             for n, s in scores.items()}
    own_all = [int(u) for u in np.argsort(isc)[::-1] if u not in prim_set]

    out = {"by_k": {}}
    for k in args.topk:
        rnd = [completion(list(rng.choice(cand, size=k, replace=False)))
               for _ in range(args.n_random)]
        row = {"random": {"drop": float(np.mean(rnd)), "drop_std": float(np.std(rnd))},
               "own": {"drop": completion(own_all[:k]),
                       "heads": [list(map(int, lh(u))) for u in own_all[:k]]}}
        for n in order:
            sel = order[n][:k]
            row[n] = {"drop": completion(sel), "heads": [list(map(int, lh(u))) for u in sel]}
        base = max(1e-9, row["random"]["drop"])
        for n, v in row.items():
            v["over_random"] = v["drop"] / base
            v["over_primary_only"] = v["drop"] / max(1e-9, drop_prim)
        out["by_k"][str(k)] = row
        print(f"[fair]  k={k:2d}  " + "  ".join(
            f"{n}={row[n]['over_random']:.2f}" for n in ["coax", "cond_energy", "coupling",
                                                          "ratio", "atpstar"] if n in row),
              flush=True)

    dest = args.out or f"outputs/coablation/complfair_{args.model_key}.json"
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(
        {"model": args.model_key, "protocol": {
            "n_detect": args.n_detect, "n_calib": args.n_calib, "n_eval": args.n_eval,
            "seed_detect": args.seed_detect, "seed_calib": args.seed_calib,
            "seed_eval": args.seed_eval, "seq_len": args.seq_len, "top_r": top_r,
            "top_r_approximation": requested_top_r > 0,
            "uniform_calibration_budget": True, "splits_disjoint": True},
         "primaries": [list(map(int, p)) for p in primaries],
         "clean": clean, "primary_only_drop": drop_prim,
         "scores": {n: [None if not np.isfinite(v) else float(v) for v in s]
                    for n, s in scores.items()},
         **out}, indent=2), encoding="utf-8")
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
