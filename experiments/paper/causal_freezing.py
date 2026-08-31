#!/usr/bin/env python3
"""Counterfactual freezing of each SELECTOR'S OWN set, not of the documented backups.

The causal-freezing figure freezes the documented backups and shows that preventing their state
change causes a substantial IOI-margin loss.
That establishes the mechanism, but it does not by itself establish that the set CoAx returns is
that mechanism. This script closes the asymmetry: every selector proposes a size-matched set with
no backup annotations, and we ask the same causal question of each one.

Protocol, per prompt:
  clean                      intact model.
  A = primaries ablated      self-repair happens.
  B = primaries + documented backups ablated      the no-backup floor.
  freeze(X)                  A, with set X's final-position output frozen to its clean value.

The paper's primary readout is the absolute freeze-induced loss

  freeze_loss(X) = A - freeze(X).

For completeness the JSON also retains the dimensionless fraction

  carried(X) = (A - freeze(X)) / repair

which is the share of the available self-repair carried by X's state change. Do not substitute
this fraction for the absolute margin loss in the paper table.

Raw magnitude alone is not the right yardstick here, for the same reason Section 4 gives for
knockout: freezing a set that contains large non-backup heads disrupts the computation and can push
the behaviour far BELOW the no-backup floor, which scores high on "repair removed" while being
nothing like the documented intervention. We therefore also report the overshoot |carried(X) - 1|
against the documented set's own value, so undershooting and overshooting are both penalised.

  PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 python scripts/run_freeze_selected.py \
      --model-key gpt2-small --num-prompts 96 --seeds 1 8 15 22 --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import curvgraph  # noqa: F401
from curvgraph._core.config import load_config
from curvgraph._core.model import load_model_bundle, bundle_device
from curvgraph import circuits as C
from curvgraph import baselines as B
from curvgraph.coablation import CoAblation, coactivation_affinity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curvgraph.patching import capture_final_head_slices, ioi_logit_diff_with_patch


def _io_s_ids(bundle, example):
    io = bundle.tokenizer(example["io"], add_special_tokens=False)["input_ids"][0]
    subject = bundle.tokenizer(example["s"], add_special_tokens=False)["input_ids"][0]
    return io, subject


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-key", default="gpt2-small")
    ap.add_argument("--model-path", default=None,
                    help="optional local checkpoint path; overrides configs/model.yaml")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 8, 15, 22])
    ap.add_argument("--k", type=int, default=8, help="set size; 8 matches the documented backups")
    ap.add_argument("--position-mode", default="last", choices=["all", "last", "full"])
    ap.add_argument("--out", default="outputs/coablation/freeze_selected_MAIN.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["models"][args.model_key]["path"] = args.model_path
    bundle = load_model_bundle(cfg["model"]["models"][args.model_key],
                              cfg["model"].get("tokenizer", {}))
    nH, dev = bundle.num_heads, bundle_device(bundle)
    nU = bundle.num_layers * nH
    top_r = int(bundle.tokenizer.vocab_size)

    prim_h = list(C.IOI_CIRCUIT["name_mover"])
    doc_h = list(C.IOI_CIRCUIT["backup_name_mover"])
    prim_u = set(C.head_index(l, h, nH) for (l, h) in prim_h)
    cand = [u for u in range(nU) if u not in prim_u]
    lh = lambda u: C.head_layer_head(int(u), nH)

    by_seed = {}
    for sd in args.seeds:
        prompts = C.ioi_prompts(args.num_prompts, seed=sd)
        seqs = [bundle.tokenizer(e["prompt"], return_tensors="pt").to(dev)["input_ids"]
                for e in prompts]
        seqs = [s for s in seqs if s.shape[1] >= 4]

        # ---- selectors, none of which sees a backup label ----
        co = CoAblation(bundle, seqs, top_r=top_r, position_mode=args.position_mode)
        r = co.conditional_compensation(prim_h, head_set=list(range(nU)))
        A_act = coactivation_affinity(bundle, seqs, list(range(nU)))
        aps = B.head_attribution_graddrop(bundle, prompts)
        eps = 1e-12
        vec = {
            "CoAx growth": r["compensation"],
            "conditional ratio": r["conditional"] / (r["single"] + eps),
            "conditional only": r["conditional"],
            "co-activation": np.array([np.abs(A_act[u, sorted(prim_u)]).mean()
                                       for u in range(nU)]),
            "AtP* GradDrop": np.asarray(aps, dtype=float),
        }
        rng = np.random.default_rng(sd)
        sets = {n: [lh(u) for u in sorted(cand, key=lambda u: -np.nan_to_num(v[u], nan=-np.inf))[:args.k]]
                for n, v in vec.items()}
        sets["random"] = [lh(u) for u in rng.choice(cand, size=args.k, replace=False)]
        sets["documented (oracle)"] = doc_h[:args.k]

        # ---- the same causal question, asked of every set ----
        acc = {k: [] for k in ["clean", "A", "B", *sets]}
        for ex in prompts:
            ids = bundle.tokenizer(ex["prompt"], return_tensors="pt").to(dev)["input_ids"]
            if ids.shape[1] < 4:
                continue
            io, s = _io_s_ids(bundle, ex)
            acc["clean"].append(ioi_logit_diff_with_patch(bundle, ids, io, s))
            acc["A"].append(ioi_logit_diff_with_patch(bundle, ids, io, s, ablate=prim_h))
            acc["B"].append(ioi_logit_diff_with_patch(bundle, ids, io, s,
                                                        ablate=prim_h + doc_h))
            for n, hs in sets.items():
                acc[n].append(ioi_logit_diff_with_patch(
                    bundle, ids, io, s, ablate=prim_h,
                    patch=capture_final_head_slices(bundle, ids, hs, ablate=None)))

        m = {k: float(np.mean(v)) for k, v in acc.items()}
        repair = m["A"] - m["B"]
        carried = {n: (m["A"] - m[n]) / max(1e-9, repair) for n in sets}
        ref = carried["documented (oracle)"]
        by_seed[str(sd)] = {
            "means": m, "repair": repair,
            "freeze_loss": {n: m["A"] - m[n] for n in sets},
            "carried": carried,
            "overshoot_vs_documented": {n: abs(carried[n] - ref) for n in sets},
            "overlap_with_documented": {n: len(set(map(tuple, hs)) & set(map(tuple, doc_h)))
                                        for n, hs in sets.items()},
            "sets": {n: [list(map(int, x)) for x in hs] for n, hs in sets.items()},
        }
        print(f"[freeze] seed={sd}  repair={repair:.3f}  " + "  ".join(
            f"{n}={carried[n]:.2f}" for n in sets), flush=True)

    names = list(by_seed[str(args.seeds[0])]["carried"])
    summary = {n: {"freeze_loss_mean": float(np.mean([by_seed[str(s)]["freeze_loss"][n]
                                                       for s in args.seeds])),
                   "freeze_loss_std": float(np.std([by_seed[str(s)]["freeze_loss"][n]
                                                     for s in args.seeds])),
                   "carried_mean": float(np.mean([by_seed[str(s)]["carried"][n] for s in args.seeds])),
                   "carried_std": float(np.std([by_seed[str(s)]["carried"][n] for s in args.seeds])),
                   "overshoot_mean": float(np.mean([by_seed[str(s)]["overshoot_vs_documented"][n]
                                                    for s in args.seeds])),
                   "overlap_mean": float(np.mean([by_seed[str(s)]["overlap_with_documented"][n]
                                                  for s in args.seeds]))}
               for n in names}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model_key, "num_prompts": args.num_prompts, "seeds": args.seeds,
         "k": args.k, "by_seed": by_seed, "summary": summary}, indent=2), encoding="utf-8")

    print(f"\n== freeze-induced IOI-margin loss for each size-{args.k} set ==")
    print(f"{'selector':22s} {'margin loss':>16s} {'fraction':>10s} {'doc overlap':>12s}")
    for n in sorted(names, key=lambda n: summary[n]["overshoot_mean"]):
        s = summary[n]
        print(f"{n:22s} {s['freeze_loss_mean']:6.3f}+-{s['freeze_loss_std']:.3f} "
              f"{s['carried_mean']:8.3f}"
              f"   {s['overlap_mean']:5.2f}/{args.k}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
