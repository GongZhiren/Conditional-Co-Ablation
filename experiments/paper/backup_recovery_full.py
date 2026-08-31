#!/usr/bin/env python3
"""Run backup recovery under the main setting (answer position, full vocabulary).

The JSON output stores per-head score vectors and summary metrics for reproducible analysis.

  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/run_backup_dump_main.py \
      --model-key gpt2-small --num-prompts 96 --seeds 1 8 15 22
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import curvgraph  # noqa: F401
from curvgraph._core.config import load_config
from curvgraph._core.model import load_model_bundle, bundle_device
from curvgraph import circuits as C
from curvgraph import baselines as B
from curvgraph.coablation import CoAblation

        # Stable output-schema labels.
K_SINGLE = "single-ablation saliency (1st-order)"
K_ATP = "ATP (1st-order grad)"
K_GIM = "GIM-style cond. attribution (1st-order, fair adapt.)"
K_EAP = "EAP-IG (1st-order)"
K_APS = "AtP* GradDrop (1st-order)"
K_COND = "conditional energy (removed-state control)"
K_COAX = "conditional co-ablation (ours, 2nd-order)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-key", default="gpt2-small")
    ap.add_argument("--model-path", default=None,
                    help="optional local checkpoint path; overrides configs/model.yaml")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 8, 15, 22])
    ap.add_argument("--position-mode", default="last", choices=["all", "last", "full"])
    ap.add_argument("--top-r", type=int, default=0, help="0 = full vocabulary")
    ap.add_argument("--skip-grad", action="store_true",
                    help="skip ATP/EAP-IG/AtP* and conditional-gradient baselines (smoke tests)")
    ap.add_argument("--suffix", default="", help="e.g. '_ctx' to keep an alternative setting aside")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["models"][args.model_key]["path"] = args.model_path
    bundle = load_model_bundle(cfg["model"]["models"][args.model_key],
                              cfg["model"].get("tokenizer", {}))
    nH = bundle.num_heads
    nU = bundle.num_layers * nH
    dev = bundle_device(bundle)
    top_r = args.top_r or int(bundle.tokenizer.vocab_size)
    Path("outputs/coablation").mkdir(parents=True, exist_ok=True)

    prim_h = C.IOI_CIRCUIT["name_mover"]
    prim = set(C.head_index(l, h, nH) for (l, h) in prim_h)
    backup = set(C.head_index(l, h, nH) for (l, h) in C.IOI_CIRCUIT["backup_name_mover"])
    cand = [u for u in range(nU) if u not in prim]
    y = [1 if u in backup else 0 for u in cand]

    def auc(vec):
        v = np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0)
        return float(roc_auc_score(y, v))

    per_seed = {}
    ranking_per_seed = {}
    for sd in args.seeds:
        prompts = C.ioi_prompts(args.num_prompts, seed=sd)
        seqs = [bundle.tokenizer(e["prompt"], return_tensors="pt").to(dev)["input_ids"]
                for e in prompts]
        seqs = [s for s in seqs if s.shape[1] >= 4]

        co = CoAblation(bundle, seqs, top_r=top_r, position_mode=args.position_mode)
        r = co.conditional_compensation(prim_h, head_set=list(range(nU)))
        scores = {
            "cand": [int(u) for u in cand],
            "y": y,
            K_SINGLE: [float(np.nan_to_num(r["single"][u])) for u in cand],
            K_COND: [float(np.nan_to_num(r["conditional"][u])) for u in cand],
            K_COAX: [float(np.nan_to_num(r["compensation"][u])) for u in cand],
        }
        if not args.skip_grad:
            atp = B.head_attribution_patching(bundle, prompts)
            eap = B.integrated_gradient_attribution(bundle, prompts)
            aps = B.head_attribution_graddrop(bundle, prompts)
            gim = B.conditional_attribution_patching(bundle, prompts, prim_h)
            scores.update({
                K_ATP: [float(atp[u]) for u in cand],
                K_EAP: [float(eap[u]) for u in cand],
                K_GIM: [float(gim[u]) for u in cand],
                K_APS: [float(aps[u]) for u in cand],
            })
        aucs = {k: auc(v) for k, v in scores.items() if k not in ("cand", "y")}
        per_seed[str(sd)] = aucs
        ranking_metrics = {}
        y_arr = np.asarray(y, dtype=int)
        for name, values in scores.items():
            if name in ("cand", "y"):
                continue
            values_arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
            ranking_metrics[name] = {
                "average_precision": float(average_precision_score(y_arr, values_arr)),
                "precision_at_8": float(y_arr[np.argsort(-values_arr)[:8]].mean()),
            }
        ranking_per_seed[str(sd)] = ranking_metrics

        # name-mover (primary) recovery, for the right-hand column of the published table
        nm_cand = [u for u in range(nU) if u not in backup]
        nm_y = [1 if u in prim else 0 for u in nm_cand]

        def nm_auc(full):
            v = np.nan_to_num(np.array([full[u] for u in nm_cand], dtype=float), nan=0.0)
            return float(roc_auc_score(nm_y, v))

        nm = {K_SINGLE: nm_auc(r["single"])}
        if not args.skip_grad:
            nm.update({K_ATP: nm_auc(atp), K_EAP: nm_auc(eap)})

        base = {"model": args.model_key, "seed": sd,
                "position_mode": args.position_mode, "top_r": top_r,
                "backup_recovery_auc": aucs,
                "backup_recovery_ranking_metrics": ranking_metrics,
                "name_mover_recovery_auc": nm}
        Path(f"outputs/coablation/bc_seed{sd}{args.suffix}.json").write_text(
            json.dumps(base, indent=2), encoding="utf-8")
        Path(f"outputs/coablation/bc_dump_seed{sd}{args.suffix}.json").write_text(
            json.dumps({**base, "scores": scores}, indent=2), encoding="utf-8")
        display = {K_SINGLE: "single", K_COND: "conditional", K_COAX: "CoAx",
                   K_ATP: "AtP", K_EAP: "EAP-IG", K_GIM: "conditional-grad",
                   K_APS: "AtP*"}
        print(f"[dump] seed={sd}  " + "  ".join(
            f"{display.get(k, k)}={v:.3f}" for k, v in aucs.items()), flush=True)

    methods = list(next(iter(per_seed.values())))
    summary = {
        method: {
            "mean": float(np.mean([per_seed[str(seed)][method] for seed in args.seeds])),
            "std": float(np.std([per_seed[str(seed)][method] for seed in args.seeds])),
        }
        for method in methods
    }
    ranking_summary = {
        method: {
            metric: {
                "mean": float(np.mean([ranking_per_seed[str(seed)][method][metric]
                                        for seed in args.seeds])),
                "std": float(np.std([ranking_per_seed[str(seed)][method][metric]
                                      for seed in args.seeds])),
            }
            for metric in ("average_precision", "precision_at_8")
        }
        for method in methods
    }
    summary_path = Path(f"outputs/coablation/bc_summary{args.suffix}.json")
    summary_path.write_text(json.dumps({
        "protocol": {"model": args.model_key, "num_prompts": args.num_prompts,
                     "seeds": args.seeds, "position_mode": args.position_mode,
                     "top_r": top_r, "full_vocabulary": args.top_r == 0,
                     "gradient_baselines": not args.skip_grad},
        "by_seed": per_seed, "summary": summary,
        "ranking_metrics_by_seed": ranking_per_seed,
        "ranking_metrics_summary": ranking_summary,
    }, indent=2), encoding="utf-8")
    print("\n[dump] mean +/- std")
    for method, values in summary.items():
        print(f"  {display.get(method, method):16s} "
              f"{values['mean']:.3f} +/- {values['std']:.3f}")
    print(f"[dump] wrote per-seed dumps and {summary_path}")


if __name__ == "__main__":
    main()
