#!/usr/bin/env python3
"""Compare intact, removed-state, and two-state gradient attribution scores.

This is the matched control for the paper's conditional-measurement claim.  AtP
and AtP* GradDrop are evaluated in the intact model and in the model with the
supplied primary heads zeroed; their two-state scores are the removed-minus-
intact differences.  Exact-metadata headline artifacts are reused when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from curvgraph._core.config import load_config
from curvgraph._core.model import bundle_device, load_model_bundle
from curvgraph import baselines as B
from curvgraph import circuits as C
from curvgraph.artifacts import load_headline_vectors
from curvgraph.coablation import CoAblation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-key", default="gpt2-small")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-prompts", type=int, default=96)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 15, 22, 8])
    parser.add_argument("--position-mode", default="last", choices=["all", "last", "full"])
    parser.add_argument("--top-r", type=int, default=0, help="0 = full vocabulary")
    parser.add_argument(
        "--headline-dump-template",
        default="outputs/coablation/bc_dump_seed{seed}.json",
        help="exact-metadata headline artifact to reuse when available",
    )
    parser.add_argument(
        "--out", default="outputs/coablation/conditional_gradient_contrast.json"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["models"][args.model_key]["path"] = args.model_path
    bundle = load_model_bundle(
        cfg["model"]["models"][args.model_key], cfg["model"].get("tokenizer", {})
    )
    bundle.model.requires_grad_(False)
    bundle.model.enable_input_require_grads()

    n_heads = bundle.num_heads
    n_units = bundle.num_layers * n_heads
    device = bundle_device(bundle)
    top_r = args.top_r or int(bundle.tokenizer.vocab_size)
    primary = [tuple(head) for head in C.IOI_CIRCUIT["name_mover"]]
    primary_units = {C.head_index(layer, head, n_heads) for layer, head in primary}
    backup_units = {
        C.head_index(layer, head, n_heads)
        for layer, head in C.IOI_CIRCUIT["backup_name_mover"]
    }
    candidates = [unit for unit in range(n_units) if unit not in primary_units]
    labels = np.asarray([unit in backup_units for unit in candidates], dtype=int)

    def auc(vector: np.ndarray) -> float:
        values = np.asarray(vector, dtype=float)[candidates]
        valid = np.isfinite(values)
        return float(roc_auc_score(labels[valid], values[valid]))

    by_seed: dict[str, dict[str, float]] = {}
    score_vectors: dict[str, dict[str, list[float]]] = {}
    for seed in args.seeds:
        prompts = C.ioi_prompts(args.num_prompts, seed=seed)
        token_sequences = [
            bundle.tokenizer(example["prompt"], return_tensors="pt").to(device)["input_ids"]
            for example in prompts
        ]
        cached = load_headline_vectors(
            args.headline_dump_template.format(seed=seed),
            model=args.model_key,
            seed=seed,
            num_prompts=args.num_prompts,
            position_mode=args.position_mode,
            top_r=top_r,
            num_units=n_units,
        )
        if cached is None:
            atp_intact = B.head_attribution_patching(bundle, prompts)
            atpstar_intact = B.head_attribution_graddrop(bundle, prompts)
            result = CoAblation(
                bundle, token_sequences, top_r=top_r, position_mode=args.position_mode
            ).conditional_compensation(primary, head_set=list(range(n_units)))
        else:
            print(f"[conditional-gradient] seed={seed} reusing validated headline scores")
            atp_intact = cached["atp"]
            atpstar_intact = cached["atpstar"]
            result = {
                "single": cached["single"],
                "conditional": cached["conditional"],
                "compensation": cached["compensation"],
            }

        atp_conditional = B.conditional_attribution_patching(bundle, prompts, primary)
        atpstar_conditional = B.head_attribution_graddrop(
            bundle, prompts, ablate_heads=primary
        )
        scores = {
            "AtP intact": atp_intact,
            "AtP conditional": atp_conditional,
            "AtP two-state difference": atp_conditional - atp_intact,
            "AtP* intact": atpstar_intact,
            "AtP* conditional": atpstar_conditional,
            "AtP* two-state difference": atpstar_conditional - atpstar_intact,
            "CoAx intact energy": result["single"],
            "CoAx conditional energy": result["conditional"],
            "CoAx signed growth": result["compensation"],
        }
        by_seed[str(seed)] = {name: auc(values) for name, values in scores.items()}
        score_vectors[str(seed)] = {
            name: [float(values[unit]) for unit in candidates]
            for name, values in scores.items()
        }
        print(
            f"[conditional-gradient] seed={seed} "
            f"AtP-diff={by_seed[str(seed)]['AtP two-state difference']:.3f} "
            f"AtP*-diff={by_seed[str(seed)]['AtP* two-state difference']:.3f} "
            f"CoAx={by_seed[str(seed)]['CoAx signed growth']:.3f}",
            flush=True,
        )

    names = list(next(iter(by_seed.values())))
    summary = {
        name: {
            "mean": float(np.mean([by_seed[str(seed)][name] for seed in args.seeds])),
            "std": float(np.std([by_seed[str(seed)][name] for seed in args.seeds])),
        }
        for name in names
    }
    output = {
        "schema_version": 2,
        "model": args.model_key,
        "num_prompts": args.num_prompts,
        "seeds": args.seeds,
        "position_mode": args.position_mode,
        "top_r": top_r,
        "candidate_units": candidates,
        "labels": labels.tolist(),
        "by_seed": by_seed,
        "summary": summary,
        "scores": score_vectors,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[conditional-gradient] wrote {destination}")


if __name__ == "__main__":
    main()

