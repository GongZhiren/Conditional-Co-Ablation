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

  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/run_knockout_oracle_distance.py \
      --model-key gpt2-small --num-prompts 96 --seeds 1 15 22 8
"""
from __future__ import annotations

import argparse
import json
import sys
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
        """Final-position distribution on each prompt, under an ablation."""
        handles = C.register_heads_ablation_grouped(bundle, ablate) if ablate else []
        try:
            out = []
            for e in prompts:
                ids = bundle.tokenizer(e["prompt"] if isinstance(e, dict) else e,
                                       return_tensors="pt").to(dev)["input_ids"]
                lg = C._forward_logits(bundle, ids)[0, -1, :].float()
                out.append(torch.log_softmax(lg, dim=-1))
            return out
        finally:
            for h in handles:
                h.remove()

    def acc_and_ld(prompts, ablate):
        handles = C.register_heads_ablation_grouped(bundle, ablate) if ablate else []
        try:
            hits, diffs = [], []
            for e in prompts:
                ids = bundle.tokenizer(e["prompt"], return_tensors="pt").to(dev)["input_ids"]
                lg = C._forward_logits(bundle, ids)[0, -1, :]
                io = bundle.tokenizer(e["io"], add_special_tokens=False)["input_ids"][0]
                s = bundle.tokenizer(e["s"], add_special_tokens=False)["input_ids"][0]
                diffs.append(float(lg[io] - lg[s]))
                hits.append(1.0 if float(lg[io]) > float(lg[s]) else 0.0)
            return float(np.mean(hits)), float(np.mean(diffs)), diffs
        finally:
            for h in handles:
                h.remove()

    report = {"model": args.model_key, "topk": args.topk, "by_seed": {}}
    for pseed in args.seeds:
        prompts = C.ioi_prompts(args.num_prompts, seed=pseed)
        seqs = [bundle.tokenizer(e["prompt"], return_tensors="pt").to(dev)["input_ids"]
                for e in prompts]
        seqs = [s for s in seqs if s.shape[1] >= 4]
        co = CoAblation(bundle, seqs, top_r=V, position_mode="last")
        r = co.conditional_compensation(primary, head_set=list(range(nU)))
        aps = B.head_attribution_graddrop(bundle, prompts)
        # input-side co-activation, ranked exactly as in the backup-AUC comparison: mean |corr|
        # of each candidate with the primaries
        A_act = coactivation_affinity(bundle, seqs, list(range(nU)))
        coact = {u: float(np.abs(A_act[u, sorted(prim)]).mean()) for u in range(nU)}
        cand = [u for u in range(nU) if u not in prim]
        rng = np.random.default_rng(pseed)

        def top(vec, k):
            a = np.nan_to_num(np.array([vec[u] for u in cand], dtype=float), nan=-1e9)
            return [lh(cand[i]) for i in np.argsort(-a)[:k]]

        k = len(doc_backup)
        sel = {
            "documented (oracle)": list(doc_backup),
            "coax":  top(r["compensation"], k),
            "own (first-order top-up)": top(r["single"], k),
            "atpstar": top({u: aps[u] for u in cand}, k),
            "co-activation": top(coact, k),
            "random": [lh(int(x)) for x in rng.choice(cand, size=k, replace=False)],
        }

        # references
        _, _, ld_clean = acc_and_ld(prompts, None)
        lp_clean_unrel = answer_probs(UNRELATED, None)
        oracle_ab = list(primary) + sel["documented (oracle)"]
        acc_o, ldm_o, ld_o = acc_and_ld(prompts, oracle_ab)
        lp_o = answer_probs(prompts, oracle_ab)

        row = {"clean_logit_diff": float(np.mean(ld_clean)), "oracle_accuracy": acc_o,
               "selectors": {}}
        for name, heads in sel.items():
            ab = list(primary) + list(heads)
            acc, ldm, ld = acc_and_ld(prompts, ab)
            lp = answer_probs(prompts, ab)
            # distance to the oracle's behaviour, per prompt then averaged
            d_behav = float(np.mean([abs(a - b) for a, b in zip(ld, ld_o)]))
            d_kl = float(np.mean([float((q.exp() * (q - p)).sum()) for q, p in zip(lp_o, lp)]))
            # off-target collateral: how much unrelated text is disturbed
            lp_u = answer_probs(UNRELATED, ab)
            coll = float(np.mean([float((q.exp() * (q - p)).sum())
                                  for q, p in zip(lp_clean_unrel, lp_u)]))
            row["selectors"][name] = {"accuracy": acc, "logit_diff": ldm,
                                      "oracle_behavioural_distance": d_behav,
                                      "oracle_kl": d_kl, "collateral_kl_unrelated": coll}
            print(f"[ko] seed={pseed} {name:26s} acc={acc:.3f} ld={ldm:+.3f} "
                  f"|d_oracle|={d_behav:.3f} KL_oracle={d_kl:.4f} collateral={coll:.4f}",
                  flush=True)
        report["by_seed"][str(pseed)] = row

    names = list(report["by_seed"][str(args.seeds[0])]["selectors"])
    summ = {}
    for n in names:
        def col(f):
            return float(np.mean([report["by_seed"][str(s)]["selectors"][n][f] for s in args.seeds]))
        summ[n] = {f: col(f) for f in ("accuracy", "logit_diff", "oracle_behavioural_distance",
                                       "oracle_kl", "collateral_kl_unrelated")}
    report["summary"] = summ
    print("\n=== mean over seeds ===")
    for n, d in summ.items():
        print(f"  {n:26s} acc={d['accuracy']:.3f}  |d_oracle|={d['oracle_behavioural_distance']:.3f}"
              f"  KL_oracle={d['oracle_kl']:.4f}  collateral={d['collateral_kl_unrelated']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ko] wrote {args.out}")


if __name__ == "__main__":
    main()
