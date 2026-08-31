#!/usr/bin/env python3
"""Faithfulness / completeness / minimality of the CoAx-completed IOI circuit (the community-standard
circuit-reliability triple, Wang et al. 2022).

A first-order method finds the IOI circuit MINUS its backups (backups score low). We ask whether
completing that circuit with CoAx-discovered backups restores the three standard reliability criteria.

Circuit "run" = mean-ablate every head OUTSIDE the circuit; the active set is the circuit. We compare:
  C_base  = the primary circuit: the documented IOI circuit with its backup branch removed
  +<sel>  = C_base + the top-k heads of each label-free selector (CoAx, the amplification ratio,
            the conditional energy alone, co-activation, AtP* GradDrop, CoAx's next-ranked k)
  +random = C_base + k random heads
  C_full  = C_base + the documented backups (the complete reference circuit)

Every selector contributes its raw top-k ranking on the same prompts.  In particular, the CoAx
completion is exactly the headline last-position/full-vocabulary ranking: no backup labels, task
gradient, or task-direction filter enters selection.

Metrics:
  Faithfulness   = circuit's IOI logit-diff (complement mean-ablated) / clean logit-diff.
  Completeness   = |drop_full(K) - drop_circuit(K)| for K = name-movers (the self-repair test): a
                   complete circuit reproduces the full model's response to ablating K. Lower = better.
  Minimality     = each added backup is individually load-bearing once the name-movers are gone.

  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/run_circuit_eval.py --model-key gpt2-small
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import curvgraph  # noqa: F401
from curvgraph._core.config import load_config
from curvgraph._core.model import load_model_bundle, bundle_device
from curvgraph import circuits as C
from curvgraph.coablation import CoAblation, coactivation_affinity
from curvgraph import baselines as B


@torch.no_grad()
def ld_active(bundle, prompts, active, mean_vecs, all_heads):
    """IOI logit-diff with every head NOT in `active` mean-ablated (the circuit run)."""
    dev = bundle_device(bundle)
    repl = {hd: mean_vecs[hd] for hd in all_heads if hd not in active}
    handles = C.register_heads_ablation(bundle, list(repl.keys()), repl)
    diffs = []
    try:
        for ex in prompts:
            ids = bundle.tokenizer(ex["prompt"], return_tensors="pt").to(dev)["input_ids"]
            io = bundle.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
            s = bundle.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
            lg = C._forward_logits(bundle, ids)[0, -1, :]
            diffs.append(float(lg[io].item() - lg[s].item()))
    finally:
        for h in handles:
            h.remove()
    return float(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-key", default="gpt2-small")
    ap.add_argument("--model-path", default=None,
                    help="optional local checkpoint path; overrides configs/model.yaml")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--top-r", type=int, default=0, help="0 = full vocabulary (main protocol)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 8, 15, 22])
    ap.add_argument("--out", default="outputs/coablation/circuit_eval_MAIN.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["models"][args.model_key]["path"] = args.model_path
    bundle = load_model_bundle(cfg["model"]["models"][args.model_key], cfg["model"].get("tokenizer", {}))
    nH, nU = bundle.num_heads, bundle.num_layers * bundle.num_heads
    dev = bundle_device(bundle)
    lh = lambda u: C.head_layer_head(int(u), nH)
    top_r = args.top_r or int(bundle.tokenizer.vocab_size)
    per_seed, order = {}, None
    for sd in args.seeds:
        per_seed[str(sd)] = run_one(args, cfg, bundle, nH, nU, dev, lh, top_r, sd)
        order = list(per_seed[str(sd)]["circuits"])
    summary = {n: {m: {"mean": float(np.mean([per_seed[str(s)]["circuits"][n][m] for s in args.seeds])),
                       "std": float(np.std([per_seed[str(s)]["circuits"][n][m] for s in args.seeds]))}
                   for m in ("incompleteness", "faithfulness_norm")} for n in order}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model_key, "num_prompts": args.num_prompts, "seeds": args.seeds,
         "top_r": top_r, "by_seed": per_seed, "summary": summary}, indent=2))
    print(f"\n== incompleteness, mean over {len(args.seeds)} prompt seeds (lower is better) ==")
    for n in sorted(order, key=lambda n: summary[n]["incompleteness"]["mean"]):
        v = summary[n]["incompleteness"]
        print(f"  {n:26s} {v['mean']:.3f} +- {v['std']:.3f}"
              f"   (faithfulness {summary[n]['faithfulness_norm']['mean']:.2f})")
    print(f"[ceval] wrote {args.out}")


def run_one(args, cfg, bundle, nH, nU, dev, lh, top_r, sd):
    prompts = C.ioi_prompts(args.num_prompts, seed=sd)
    seqs = [bundle.tokenizer(ex["prompt"], return_tensors="pt").to(dev)["input_ids"] for ex in prompts]
    seqs = [s for s in seqs if s.shape[1] >= 4]
    mean_vecs = C.head_mean_vectors(bundle, seqs)
    all_heads = [lh(u) for u in range(nU)]

    NM = [tuple(x) for x in C.IOI_CIRCUIT["name_mover"]]                 # the K we ablate (self-repair test)
    BK = [tuple(x) for x in C.IOI_CIRCUIT["backup_name_mover"]]
    # C_base = documented circuit WITHOUT backups (what a first-order method finds)
    base = []
    for role, hs in C.IOI_CIRCUIT.items():
        if role == "backup_name_mover":
            continue
        base += [tuple(x) for x in hs]
    base = list(dict.fromkeys(base))
    k = len(BK)

    # Every label-free selector contributes its raw ranking.  Do not filter candidates with the
    # IOI task direction here: the paper's completion claim explicitly reuses the headline ranking.
    co = CoAblation(bundle, seqs, top_r=top_r, position_mode="last")
    r = co.conditional_compensation(NM, head_set=list(range(nU)))
    A_act = coactivation_affinity(bundle, seqs, list(range(nU)))
    prim_u = set(C.head_index(l, h, nH) for (l, h) in NM)
    eps = 1e-12
    vecs = {
        "+CoAx": r["compensation"],
        "+amplification ratio": r["conditional"] / (r["single"] + eps),
        "+conditional only": r["conditional"],
        "+co-activation": np.array([np.abs(A_act[u, sorted(prim_u)]).mean() for u in range(nU)]),
        "+AtP* GradDrop": np.asarray(B.head_attribution_graddrop(bundle, prompts), dtype=float),
    }
    def pick(vec, k, skip=0):
        """Raw top-k non-primary heads, optionally continuing after ``skip`` entries."""
        ranked = []
        for u in np.argsort(np.nan_to_num(np.asarray(vec, float), nan=-1e9))[::-1]:
            if u in prim_u:
                continue
            ranked.append(lh(u))
            if len(ranked) >= k + skip:
                break
        return ranked[skip:]

    sel = {n: pick(v, k) for n, v in vecs.items()}
    sel["+next-ranked"] = pick(vecs["+CoAx"], k, skip=k)     # CoAx's own next k, size-matched
    rng = np.random.default_rng(sd)
    pool = [u for u in range(nU) if lh(u) not in set(base) and u not in prim_u]
    sel["+random"] = [lh(int(u)) for u in rng.choice(pool, size=k, replace=False)]

    circuits = {"primary circuit": base, **{n: base + hs for n, hs in sel.items()},
                "documented circuit": base + BK}

    clean = ld_active(bundle, prompts, set(all_heads), mean_vecs, all_heads)           # no ablation
    drop_full_K = clean - ld_active(bundle, prompts, set(all_heads) - set(NM), mean_vecs, all_heads)

    res = {"seed": sd, "k_backups": k, "clean": clean, "drop_full_namemover": drop_full_K,
           "circuits": {}, "selected": {n: [list(map(int, h)) for h in hs] for n, hs in sel.items()}}
    print(f"[ceval] seed={sd} clean logit-diff {clean:.2f}; full-model drop when name-movers ablated {drop_full_K:.2f} (self-repair)")
    for name, Cset in circuits.items():
        Cset = set(Cset)
        faith = ld_active(bundle, prompts, Cset, mean_vecs, all_heads)                  # circuit alone
        perf_K = ld_active(bundle, prompts, Cset - set(NM), mean_vecs, all_heads)       # circuit minus name-movers
        drop_C_K = faith - perf_K
        incompleteness = abs(drop_full_K - drop_C_K)
        # minimality: each added backup load-bearing once name-movers gone (within the circuit)
        added = [h for h in Cset if h not in set(base)]
        nec = []
        for v in added:
            nec.append(perf_K - ld_active(bundle, prompts, Cset - set(NM) - {v}, mean_vecs, all_heads))
        res["circuits"][name] = {
            "faithfulness_norm": faith / clean, "faithfulness_raw": faith,
            "drop_when_namemovers_ablated": drop_C_K, "incompleteness": incompleteness,
            "minimality_mean_necessity": float(np.mean(nec)) if nec else 0.0,
            "minimality_frac_necessary": float(np.mean([x > 0 for x in nec])) if nec else 0.0,
        }
        r = res["circuits"][name]
        print(f"  {name:7s}: faith {r['faithfulness_norm']:.2f} | drop@NM-abl {drop_C_K:.2f} "
              f"| INCOMPLETENESS {incompleteness:.2f} | minimality {r['minimality_frac_necessary']:.2f} "
              f"({r['minimality_mean_necessity']:+.2f})")
    return res


if __name__ == "__main__":
    main()
