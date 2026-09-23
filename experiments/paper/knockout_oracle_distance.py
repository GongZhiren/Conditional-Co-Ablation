#!/usr/bin/env python3
"""Knockout, judged by distance to the documented oracle rather than by how far accuracy falls.

Reporting that a first-order top-up drives IOI accuracy to 0.24 while CoAx reaches 0.70 does not
by itself show the first-order set is worse: if the goal were only to disable a capability, 0.24
would look like the better outcome. What we actually claim is narrower and testable -- that CoAx
removes the components the documented backup intervention removes, while the first-order top-up
cuts into the broader core circuit. The right measurement is therefore distance to the documented
oracle's behaviour, plus the collateral damage each selection inflicts away from the target.

For every selector we ablate primaries + its top-k set and measure

  * IOI accuracy and logit-difference          (target effect, as before)
  * |m_sel - m_oracle| per prompt              (behavioural distance to the documented oracle)
  * KL(p_oracle || p_sel) at the answer position (full-distribution distance to the oracle)
  * KL(p_clean || p_sel) on UNRELATED prompts    (off-target collateral damage)

If CoAx sits closest to the oracle on the first two and the first-order top-up is far on all of
them, the experiment supports 'removes the right components', which 'accuracy fell less' does not.

  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/paper/knockout_oracle_distance.py \
      --model-key gpt2-small --num-prompts 96 --seeds 1 15 22 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import curvgraph  # noqa: F401
from curvgraph._core.config import load_config
from curvgraph._core.model import load_model_bundle, bundle_device
from curvgraph import circuits as C
from curvgraph import baselines as B
from curvgraph.coablation import CoAblation, coactivation_affinity
from curvgraph.artifacts import load_headline_vectors

UNRELATED = [
    "The capital of France is a city that has long been",
    "In 1969 the first humans landed on the surface of the",
    "Water boils at one hundred degrees on the Celsius",
    "The quick brown fox jumps over the lazy",
    "A prime number is a natural number greater than one that has no",
    "The mitochondrion is often described as the powerhouse of the",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-key", default="gpt2-small")
    ap.add_argument("--model-path", default=None,
                    help="optional local checkpoint path; overrides configs/model.yaml")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 15, 22, 8])
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--random-draws", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--headline-dump-template",
                    default="outputs/coablation/bc_dump_seed{seed}.json",
                    help="validated score cache from the headline run; recomputed if absent/mismatched")
    ap.add_argument("--out", default="outputs/coablation/knockout_oracle_distance.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg["model"]["models"][args.model_key]["path"] = args.model_path
    bundle = load_model_bundle(cfg["model"]["models"][args.model_key],
                              cfg["model"].get("tokenizer", {}))
    nH = bundle.num_heads
    nU = bundle.num_layers * nH
    dev = bundle_device(bundle)
    V = int(bundle.tokenizer.vocab_size)

    primary = C.IOI_CIRCUIT["name_mover"]
    prim = set(C.head_index(l, h, nH) for (l, h) in primary)
    doc_backup = C.IOI_CIRCUIT["backup_name_mover"]

    def lh(u):
        return C.head_layer_head(int(u), nH)

    def answer_probs(prompts, ablate):
        """Final-position log distributions, batched by tokenized length."""
        records = []
        for index, example in enumerate(prompts):
            text = example["prompt"] if isinstance(example, dict) else example
            ids = bundle.tokenizer(text, return_tensors="pt")["input_ids"]
            records.append((index, ids))
        handles = C.register_heads_ablation_grouped(bundle, ablate) if ablate else []
        try:
            out = [None] * len(records)
            groups = defaultdict(list)
            for record in records:
                groups[record[1].shape[1]].append(record)
            with torch.no_grad():
                for items in groups.values():
                    ids = torch.cat([item[1] for item in items], dim=0).to(dev)
                    lp = torch.log_softmax(C._forward_logits(bundle, ids)[:, -1, :].float(), dim=-1)
                    for item, value in zip(items, lp.cpu()):
                        out[item[0]] = value
            return out
        finally:
            for h in handles:
                h.remove()

    def acc_and_ld(prompts, log_probs):
        diffs = []
        for example, lp in zip(prompts, log_probs):
            io = bundle.tokenizer(example["io"], add_special_tokens=False)["input_ids"][0]
            subject = bundle.tokenizer(example["s"], add_special_tokens=False)["input_ids"][0]
            diffs.append(float(lp[io] - lp[subject]))
        return float(np.mean(np.asarray(diffs) > 0)), float(np.mean(diffs)), diffs

    report = {"model": args.model_key, "topk": args.topk,
              "random_draws": args.random_draws, "bootstrap": args.bootstrap,
              "bootstrap_seed": args.bootstrap_seed, "by_seed": {}}
    for pseed in args.seeds:
        prompts = C.ioi_prompts(args.num_prompts, seed=pseed)
        seqs = [bundle.tokenizer(e["prompt"], return_tensors="pt").to(dev)["input_ids"]
                for e in prompts]
        seqs = [s for s in seqs if s.shape[1] >= 4]
        cached = load_headline_vectors(
            args.headline_dump_template.format(seed=pseed), model=args.model_key, seed=pseed,
            num_prompts=args.num_prompts, position_mode="last", top_r=V, num_units=nU,
        )
        if cached is None:
            co = CoAblation(bundle, seqs, top_r=V, position_mode="last")
            r = co.conditional_compensation(primary, head_set=list(range(nU)))
            aps = B.head_attribution_graddrop(bundle, prompts)
        else:
            print(f"[ko] seed={pseed} reusing validated headline scores", flush=True)
            r = cached
            aps = cached["atpstar"]
        # input-side co-activation, ranked exactly as in the backup-AUC comparison: mean |corr|
        # of each candidate with the primaries
        A_act = coactivation_affinity(bundle, seqs, list(range(nU)))
        coact = {u: float(np.abs(A_act[u, sorted(prim)]).mean()) for u in range(nU)}
        cand = [u for u in range(nU) if u not in prim]

        def top(vec, k, skip=0):
            a = np.nan_to_num(np.array([vec[u] for u in cand], dtype=float), nan=-1e9)
            return [lh(cand[i]) for i in np.argsort(-a)[skip:skip + k]]

        k = args.topk
        if k != len(doc_backup):
            raise ValueError(f"the documented-backup oracle has {len(doc_backup)} heads; "
                             f"this comparison requires --topk {len(doc_backup)}")
        sel = {
            "documented (oracle)": list(doc_backup),
            "coax":  top(r["compensation"], k),
            "conditional only": top(r["conditional"], k),
            "amplification ratio": top(
                r["conditional"] / np.maximum(r["single"], 1e-12), k
            ),
            "CoAx next-8": top(r["compensation"], k, skip=k),
            "atpstar": top({u: aps[u] for u in cand}, k),
            "co-activation": top(coact, k),
        }

        # references
        lp_clean = answer_probs(prompts, None)
        _, _, ld_clean = acc_and_ld(prompts, lp_clean)
        lp_clean_unrel = answer_probs(UNRELATED, None)
        oracle_ab = list(primary) + sel["documented (oracle)"]
        lp_o = answer_probs(prompts, oracle_ab)
        acc_o, ldm_o, ld_o = acc_and_ld(prompts, lp_o)

        row = {"clean_logit_diff": float(np.mean(ld_clean)), "oracle_accuracy": acc_o,
               "selectors": {}}

        def evaluate_heads(heads):
            ab = list(primary) + list(heads)
            lp = answer_probs(prompts, ab)
            acc, ldm, ld = acc_and_ld(prompts, lp)
            margin_distance = [abs(a - b) for a, b in zip(ld, ld_o)]
            output_kl = [float((q.exp() * (q - p)).sum()) for q, p in zip(lp_o, lp)]
            lp_u = answer_probs(UNRELATED, ab)
            collateral = [float((q.exp() * (q - p)).sum()) for q, p in zip(lp_clean_unrel, lp_u)]
            return {"accuracy": acc, "logit_diff": ldm,
                    "oracle_behavioural_distance": float(np.mean(margin_distance)),
                    "oracle_kl": float(np.mean(output_kl)),
                    "collateral_kl_unrelated": float(np.mean(collateral)),
                    "per_prompt_margin_distance": margin_distance,
                    "per_prompt_oracle_kl": output_kl,
                    "per_prompt_collateral_kl": collateral,
                    "selected_heads": [list(map(int, head)) for head in heads]}

        for name, heads in sel.items():
            row["selectors"][name] = evaluate_heads(heads)
            values = row["selectors"][name]
            print(f"[ko] seed={pseed} {name:26s} acc={values['accuracy']:.3f} "
                  f"ld={values['logit_diff']:+.3f} "
                  f"|d_oracle|={values['oracle_behavioural_distance']:.3f} "
                  f"KL_oracle={values['oracle_kl']:.4f} "
                  f"collateral={values['collateral_kl_unrelated']:.4f}",
                  flush=True)

        random_rows = []
        random_heads = []
        for draw in range(args.random_draws):
            rng = np.random.default_rng(np.random.SeedSequence([pseed, draw]))
            heads = [lh(int(x)) for x in rng.choice(cand, size=k, replace=False)]
            random_heads.append([list(map(int, head)) for head in heads])
            random_rows.append(evaluate_heads(heads))
        scalar_fields = ("accuracy", "logit_diff", "oracle_behavioural_distance",
                         "oracle_kl", "collateral_kl_unrelated")
        prompt_fields = ("per_prompt_margin_distance", "per_prompt_oracle_kl",
                         "per_prompt_collateral_kl")
        random_result = {field: float(np.mean([item[field] for item in random_rows]))
                         for field in scalar_fields}
        random_result.update({
            field: np.mean(np.asarray([item[field] for item in random_rows], dtype=float), axis=0).tolist()
            for field in prompt_fields
        })
        random_result["selected_heads_by_draw"] = random_heads
        row["selectors"]["random"] = random_result
        random_label = f"random ({args.random_draws} draws)"
        print(f"[ko] seed={pseed} {random_label:26s} "
              f"acc={random_result['accuracy']:.3f} ld={random_result['logit_diff']:+.3f} "
              f"|d_oracle|={random_result['oracle_behavioural_distance']:.3f} "
              f"KL_oracle={random_result['oracle_kl']:.4f} "
              f"collateral={random_result['collateral_kl_unrelated']:.4f}", flush=True)
        report["by_seed"][str(pseed)] = row

    names = list(report["by_seed"][str(args.seeds[0])]["selectors"])
    summ = {}
    for n in names:
        def col(f):
            return float(np.mean([report["by_seed"][str(s)]["selectors"][n][f] for s in args.seeds]))
        summ[n] = {f: col(f) for f in ("accuracy", "logit_diff", "oracle_behavioural_distance",
                                       "oracle_kl", "collateral_kl_unrelated")}
    report["summary"] = summ
    pooled = {
        name: np.concatenate([
            np.asarray(report["by_seed"][str(seed)]["selectors"][name]
                       ["per_prompt_margin_distance"], dtype=float)
            for seed in args.seeds
        ])
        for name in names
    }
    paired_bootstrap = {}
    for selector_index, name in enumerate(names):
        if name in ("documented (oracle)", "coax"):
            continue
        rng = np.random.default_rng(np.random.SeedSequence(
            [args.bootstrap_seed, selector_index]
        ))
        n = len(pooled["coax"])
        draws = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, n, size=n)
            draws.append(float((pooled[name][idx] - pooled["coax"][idx]).mean()))
        paired_bootstrap[name] = {
            "contrast": "selector margin distance - CoAx margin distance",
            "mean_difference": float((pooled[name] - pooled["coax"]).mean()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        }
    report["paired_prompt_bootstrap_vs_coax"] = paired_bootstrap
    print("\n=== mean over seeds ===")
    for n, d in summ.items():
        print(f"  {n:26s} acc={d['accuracy']:.3f}  |d_oracle|={d['oracle_behavioural_distance']:.3f}"
              f"  KL_oracle={d['oracle_kl']:.4f}  collateral={d['collateral_kl_unrelated']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ko] wrote {args.out}")


if __name__ == "__main__":
    main()
