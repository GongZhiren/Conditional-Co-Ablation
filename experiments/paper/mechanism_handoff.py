#!/usr/bin/env python3
"""Mechanistic closure: how do the discovered backups take over in GPT-2-small?

Two complementary, label-free mechanism probes, each as a function of how much of the primary
circuit is removed (k = 0,1,2,3 primary name-movers ablated, ordered by their own single effect):

  (1) Wake-up curve. For each documented backup head we track its output-norm activation ratio
      (||head out|| with k primaries ablated / clean) and its conditional causal drop (IOI
      logit-diff drop from ablating it given the k primaries are gone). A real backup wakes up and
      becomes load-bearing monotonically as more primaries are removed; a matched random head does
      not.

  (2) Direct logit attribution (DLA) handoff. We decompose each head's direct contribution to the
      IO-vs-S logit at the final position via the frozen final-LayerNorm + unembedding (the standard
      logit-lens DLA). We show (i) the primary name-movers carry large positive clean DLA (sanity:
      they write the answer), and (ii) the backups' DLA to the correct answer RISES once the
      primaries are ablated -- a direct measurement of the hand-off, versus a flat random control.

  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/run_mechanism.py --model-key gpt2-small
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import curvgraph  # noqa: F401
from curvgraph._core.config import load_config
from curvgraph._core.model import load_model_bundle, bundle_device, layer_modules
from curvgraph import circuits as C
from curvgraph.coablation import CoAblation


def head_output_norms(bundle, seqs, ablate_heads=None):
    """Mean L2 norm of each head's attention-output slice over positions/sequences (the validated
    activation-ratio probe; identical to run_backup_qualitative)."""
    nH, hd = bundle.num_heads, bundle.head_dim
    cap = {u: 0.0 for u in range(bundle.num_layers * nH)}
    cnt = [0]
    handles = []
    abl = C.register_heads_ablation(bundle, ablate_heads) if ablate_heads else []

    def mk(li):
        def hook(_m, inp):
            x = inp[0].detach()
            if x.dim() != 3:
                return
            flat = x.reshape(-1, x.shape[-1]); cnt[0] += flat.shape[0]
            for hi in range(nH):
                cap[C.head_index(li, hi, nH)] += float(flat[:, hi * hd:(hi + 1) * hd].norm(dim=-1).sum())
        return hook

    for li in range(bundle.num_layers):
        cp = C._cproj(C._attn_module(layer_modules(bundle)[li]))
        if cp is not None:
            handles.append(cp.register_forward_pre_hook(mk(li)))
    try:
        for s in seqs:
            C._forward_logits(bundle, s)
    finally:
        for h in handles + abl:
            h.remove()
    n = max(1, cnt[0])
    return {u: cap[u] / n for u in cap}


def dla_io_minus_s(bundle, prompts, heads, ablate_heads=None):
    """Mean direct-logit-attribution of each head in `heads` to (IO - S) at the final position,
    via the frozen final LayerNorm + unembedding. Returns {(l,h): mean_DLA}."""
    dev = bundle_device(bundle)
    model = bundle.model
    nH, hd = bundle.num_heads, bundle.head_dim
    lnf = model.transformer.ln_f
    gamma = lnf.weight.detach().to(dev).float()
    beta = lnf.bias.detach().to(dev).float() if lnf.bias is not None else None
    eps = getattr(lnf, "eps", 1e-5)
    WU = model.lm_head.weight.detach().to(dev).float()              # [V, d]
    layers = layer_modules(bundle)
    cprojs = [C._cproj(C._attn_module(layers[li])) for li in range(bundle.num_layers)]
    Wc = [cp.weight.detach().to(dev).float() for cp in cprojs]       # Conv1D weight [d_in, d_out]

    cap = {}        # (li) -> c_proj input at final pos [d]
    resid = {}      # input to ln_f at final pos [d]

    def mk(li):
        def hook(_m, inp):
            cap[li] = inp[0][:, -1, :].detach().float()             # [1, d]
        return hook

    def lnf_hook(_m, inp):
        resid["x"] = inp[0][:, -1, :].detach().float()

    sums = {(l, h): 0.0 for (l, h) in heads}
    n = 0
    for ex in prompts:
        ids = bundle.tokenizer(ex["prompt"], return_tensors="pt").to(dev)["input_ids"]
        io = bundle.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
        s = bundle.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
        d_dir = WU[io] - WU[s]                                       # [d] unembed direction
        handles = [cprojs[li].register_forward_pre_hook(mk(li)) for li in range(bundle.num_layers)]
        handles.append(lnf.register_forward_pre_hook(lnf_hook))
        abl = C.register_heads_ablation(bundle, ablate_heads) if ablate_heads else []
        try:
            C._forward_logits(bundle, ids)
        finally:
            for h in handles + abl:
                h.remove()
        x = resid["x"][0]                                            # [d] full final residual
        mu, sd = x.mean(), x.std(unbiased=False)
        for (l, h) in heads:
            a_h = cap[l][0, h * hd:(h + 1) * hd]                     # [hd] head slice into c_proj
            r_h = a_h @ Wc[l][h * hd:(h + 1) * hd, :]               # [d] residual contribution
            scaled = gamma * (r_h - r_h.mean()) / (sd + eps)        # frozen-LN linearization
            sums[(l, h)] += float((scaled @ d_dir).item())
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-key", default="gpt2-small")
    ap.add_argument("--model-path", default=None,
                    help="optional local checkpoint path; overrides configs/model.yaml")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["models"][args.model_key]["path"] = args.model_path
    bundle = load_model_bundle(cfg["model"]["models"][args.model_key], cfg["model"].get("tokenizer", {}))
    nH, nU = bundle.num_heads, bundle.num_layers * bundle.num_heads
    dev = bundle_device(bundle)
    lh = lambda u: C.head_layer_head(int(u), nH)

    prompts = C.ioi_prompts(args.num_prompts, seed=args.seed)
    seqs = [bundle.tokenizer(ex["prompt"], return_tensors="pt").to(dev)["input_ids"] for ex in prompts]
    seqs = [s for s in seqs if s.shape[1] >= 4]

    P_all = list(C.IOI_CIRCUIT["name_mover"])
    backups = list(C.IOI_CIRCUIT["backup_name_mover"])
    backup_u = [C.head_index(l, h, nH) for (l, h) in backups]
    # order primaries by their own single-ablation IOI effect (so k=1 ablates the strongest)
    b0 = C.ioi_logit_diff(bundle, prompts)
    P_order = sorted(P_all, key=lambda lhd: -(b0 - C.ioi_logit_diff(bundle, prompts, ablate_heads=[lhd])))
    # matched random control heads (not primary, not backup), fixed across k
    rng = np.random.default_rng(args.seed)
    primary_units = set(C.head_index(l, h, nH) for (l, h) in P_all)
    candidates = [u for u in range(nU) if u not in primary_units]
    random_pool = [u for u in candidates if u not in set(backup_u)]
    rand_u = [int(x) for x in rng.choice(random_pool, size=len(backups), replace=False)]
    rand_heads = [lh(u) for u in rand_u]

    # Select CoAx once using the full primary set, then hold that set fixed across k.
    # This is the fixed-set protocol behind the paper's graded wake-up curve.
    co = CoAblation(bundle, seqs, top_r=int(bundle.tokenizer.vocab_size), position_mode="last")
    comp = co.conditional_compensation(P_all, head_set=list(range(nU)))["compensation"]
    coax_u = sorted(candidates, key=lambda u: -np.nan_to_num(comp[u], nan=-np.inf))[:len(backups)]
    coax_heads = [lh(u) for u in coax_u]

    norm_clean = head_output_norms(bundle, seqs)

    # ---- (1) wake-up curve over k primaries ablated ----
    curve = []
    for k in range(0, len(P_order) + 1):
        Pk = P_order[:k]
        norm_k = head_output_norms(bundle, seqs, ablate_heads=Pk) if Pk else norm_clean
        base = C.ioi_logit_diff(bundle, prompts, ablate_heads=Pk) if Pk else b0
        def stats(units):
            ratios = [norm_k[u] / max(1e-9, norm_clean[u]) for u in units]
            drops = [base - C.ioi_logit_diff(bundle, prompts, ablate_heads=Pk + [lh(u)]) for u in units]
            return float(np.mean(ratios)), float(np.mean(drops))
        br, bd = stats(backup_u)
        cr, cd = stats(coax_u)
        rr, rd = stats(rand_u)
        curve.append({"k": k, "ioi_logit_diff": base,
                      "backup_act_ratio": br, "backup_cond_drop": bd,
                      "coax_act_ratio": cr, "coax_cond_drop": cd,
                      "random_act_ratio": rr, "random_cond_drop": rd})
        print(f"  k={k}  backup[ratio {br:.3f}, drop {bd:+.3f}]  "
              f"CoAx[ratio {cr:.3f}, drop {cd:+.3f}]  "
              f"random[ratio {rr:.3f}, drop {rd:+.3f}]")

    # ---- (2) DLA handoff: clean vs all-primaries-ablated ----
    nm_u = [C.head_index(l, h, nH) for (l, h) in P_all]
    all_heads = [lh(u) for u in range(nU)]
    other_heads = [head for head in all_heads if head not in set(P_all) | set(backups)]
    probe = all_heads
    dla_clean = dla_io_minus_s(bundle, prompts, probe)
    dla_abl = dla_io_minus_s(bundle, prompts, probe, ablate_heads=P_order)  # all primaries gone
    def mean_dla(heads, table):
        return float(np.mean([table[(l, h)] for (l, h) in heads]))
    dla = {
        "name_mover_clean": mean_dla(P_all, dla_clean),               # sanity: should be large +
        "backup_clean": mean_dla(backups, dla_clean),
        "backup_primaries_ablated": mean_dla(backups, dla_abl),
        "other_clean": mean_dla(other_heads, dla_clean),
        "other_primaries_ablated": mean_dla(other_heads, dla_abl),
        "random_clean": mean_dla(rand_heads, dla_clean),
        "random_primaries_ablated": mean_dla(rand_heads, dla_abl),
    }
    print(f"\n[DLA to IO-S]  name-movers (clean) {dla['name_mover_clean']:+.3f}  (sanity: large +)")
    print(f"  backups  clean {dla['backup_clean']:+.3f} -> primaries ablated {dla['backup_primaries_ablated']:+.3f}"
          f"  (handoff +{dla['backup_primaries_ablated']-dla['backup_clean']:.3f})")
    print(f"  remaining clean {len(other_heads) * dla['other_clean']:+.3f} -> primaries ablated "
          f"{len(other_heads) * dla['other_primaries_ablated']:+.3f} (branch totals)")
    print(f"  random   clean {dla['random_clean']:+.3f} -> primaries ablated {dla['random_primaries_ablated']:+.3f}")

    res = {"model": args.model_key, "clean_logit_diff": b0, "wakeup_curve": curve, "dla": dla,
           "primary_order": [list(map(int, x)) for x in P_order],
           "backup_heads": [list(map(int, x)) for x in backups],
           "coax_heads": [list(map(int, x)) for x in coax_heads],
           "random_heads": [list(map(int, x)) for x in rand_heads]}
    out = args.out or f"outputs/coablation/mechanism_{args.model_key}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[mechanism] wrote {out}")


if __name__ == "__main__":
    main()
