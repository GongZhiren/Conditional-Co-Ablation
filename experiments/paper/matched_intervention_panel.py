#!/usr/bin/env python3
"""Mechanism-matched hierarchical intervention panel from the paper."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from curvgraph._core.config import load_config
from curvgraph._core.model import bundle_device, load_model_bundle
from curvgraph import baselines as B
from curvgraph import circuits as C
from curvgraph.coablation import CoAblation
from curvgraph.patching import FreezeValidator, capture_head_activations, register_freeze


TEMPLATES = {
    "standard":   "When {io} and {s} went to the {place}, {s} gave a {obj} to",
    "afterwards": "After {io} and {s} arrived at the {place}, {s} handed a {obj} to",
    "while":      "While {io} and {s} were at the {place}, {s} passed a {obj} to",
}
PRIMARY_SETS = ["name_mover", "s_inhibition", "induction", "duplicate_token"]
SEED_METRIC = {"name_mover": "ioi", "s_inhibition": "ioi",
               "induction": "induction", "duplicate_token": "induction"}
METHODS = ["coax", "atpstar", "single", "eapig", "atp"]


def make_prompts(template: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        io, s = rng.choice(C._IOI_NAMES, size=2, replace=False)
        place = rng.choice(C._IOI_PLACES)
        obj = rng.choice(C._IOI_OBJECTS)
        out.append({"prompt": template.format(io=io, s=s, place=place, obj=obj),
                    "io": " " + io, "s": " " + s})
    return out


def induction_batch(bundle, num_seqs: int, seq_len: int, seed: int) -> Tuple[torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    dev = bundle_device(bundle)
    hi = min(10000, int(bundle.tokenizer.vocab_size))
    halves = [rng.integers(1000, hi, size=seq_len) for _ in range(num_seqs)]
    ids = torch.tensor(np.stack([np.concatenate([h, h]) for h in halves]),
                       dtype=torch.long, device=dev)
    return ids, seq_len


class InductionFreezeValidator:
    def __init__(self, bundle, ids: torch.Tensor, half_len: int):
        self.bundle = bundle
        self.ids = ids
        self.T = int(half_len)
        self.nH = bundle.num_heads
        self.nU = bundle.num_layers * bundle.num_heads

    def _metric(self, ablate=None, frozen=None) -> float:
        handles = C.register_heads_ablation_grouped(self.bundle, ablate) if ablate else []
        if frozen:
            handles += register_freeze(self.bundle, frozen)
        try:
            with torch.no_grad():
                lp = torch.log_softmax(C._forward_logits(self.bundle, self.ids).float(), dim=-1)
            tgt = self.ids[:, self.T + 1:2 * self.T]
            got = lp[:, self.T:2 * self.T - 1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            return float(got.mean().item())
        finally:
            for h in handles:
                h.remove()

    def repair_removed(self, primary: Sequence[Tuple[int, int]], units: Sequence[int]) -> Dict:
        prim_units = set(C.head_index(l, h, self.nH) for (l, h) in primary)
        pool = [int(u) for u in units if int(u) not in prim_units]
        heads = [C.head_layer_head(u, self.nH) for u in pool]
        m_clean = self._metric()
        m_S = self._metric(ablate=list(primary))
        clean_acts = capture_head_activations(self.bundle, self.ids, heads)
        rr = {}
        for u, lh in zip(pool, heads):
            m_frz = self._metric(ablate=list(primary), frozen={lh: clean_acts[lh]})
            rr[u] = m_S - m_frz
        return {"clean": m_clean, "seed_ablated": m_S, "repair_removed": rr}


def nested_summary(inst: Dict[str, Dict], bootstrap: int) -> Dict:
    by_ps = defaultdict(dict)
    for key, row in inst.items():
        tpl, ps = key.split("/")
        by_ps[ps][tpl] = row
    ps_names = sorted(by_ps)
    rng = np.random.default_rng(0)
    out = {"per_instance": {}, "per_primary_set": {}, "macro": {}, "nested_ci": {},
           "loo": {}, "sign": {}}
    for m in METHODS:
        out["per_instance"][m] = {k: float(v["by_method"][m]["spearman"])
                                  for k, v in sorted(inst.items())}
        per_ps = {}
        for ps in ps_names:
            vals = [by_ps[ps][tpl]["by_method"][m]["spearman"] for tpl in by_ps[ps]]
            per_ps[ps] = float(np.mean(vals))
        out["per_primary_set"][m] = per_ps
        vals = list(per_ps.values())
        out["macro"][m] = float(np.mean(vals))
        loo = [float(np.mean([v for p, v in per_ps.items() if p != drop])) for drop in ps_names]
        out["loo"][m] = {"min": float(np.min(loo)), "max": float(np.max(loo)),
                         "dropped_for_min": ps_names[int(np.argmin(loo))],
                         "dropped_for_max": ps_names[int(np.argmax(loo))]}
        draws = []
        for _ in range(bootstrap):
            acc = []
            for ps in rng.choice(ps_names, size=len(ps_names), replace=True):
                tpls = list(by_ps[ps])
                picks = rng.choice(len(tpls), size=len(tpls), replace=True)
                acc.append(np.mean([by_ps[ps][tpls[i]]["by_method"][m]["spearman"]
                                    for i in picks]))
            draws.append(float(np.mean(acc)))
        out["nested_ci"][m] = [float(np.percentile(draws, 2.5)),
                               float(np.percentile(draws, 97.5))]
        pis = list(out["per_instance"][m].values())
        out["sign"][m] = {"positive_instances": int(sum(v > 0 for v in pis)),
                          "n_instances": len(pis),
                          "positive_primary_sets": int(sum(v > 0 for v in vals)),
                          "n_primary_sets": len(vals)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model-key", default="gpt2-small")
    ap.add_argument("--model-path", default=None,
                    help="optional local checkpoint; overrides configs/model.yaml")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--ind-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--calib-seed", type=int, default=1)
    ap.add_argument("--valid-seed", type=int, default=15)
    ap.add_argument("--position-mode", default="last", choices=["all", "last", "full"])
    ap.add_argument("--top-r", type=int, default=0, help="0 = full vocab")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--out", default="outputs/coablation/panel_metric_match_hierarchical.json")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run with identical protocol settings")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_spec = dict(cfg["model"]["models"][args.model_key])
    if args.model_path:
        model_spec["path"] = args.model_path
        model_spec["local_files_only"] = True
    bundle = load_model_bundle(model_spec,
                               cfg["model"].get("tokenizer", {}))
    dev = bundle_device(bundle)
    nH, nU = bundle.num_heads, bundle.num_layers * bundle.num_heads
    top_r = args.top_r or int(bundle.tokenizer.vocab_size)

    protocol = {"model": args.model_key, "num_prompts": args.num_prompts,
              "ind_seqs": args.ind_seqs, "seq_len": args.seq_len,
              "top_r": top_r, "position_mode": args.position_mode,
              "calib_seed": args.calib_seed, "valid_seed": args.valid_seed,
              "bootstrap": args.bootstrap, "seed_metric": SEED_METRIC}
    out_path = Path(args.out)
    if args.resume and out_path.is_file():
        report = json.loads(out_path.read_text(encoding="utf-8"))
        mismatched = [k for k, value in protocol.items() if report.get(k) != value]
        if mismatched:
            raise ValueError("Cannot resume with changed protocol fields: " +
                             ", ".join(mismatched))
        report.pop("summary", None)
    else:
        report = {"schema_version": 1, **protocol, "instances": {}}

    ind_ids, ind_T = induction_batch(bundle, args.ind_seqs, args.seq_len, args.valid_seed)
    ind_validator = InductionFreezeValidator(bundle, ind_ids, ind_T)

    for tname, template in TEMPLATES.items():
        missing = [p for p in PRIMARY_SETS if f"{tname}/{p}" not in report["instances"]]
        if not missing:
            print(f"[mm-hier] {tname}: all instances already present", flush=True)
            continue
        calib = make_prompts(template, args.num_prompts, args.calib_seed)
        seqs = [bundle.tokenizer(e["prompt"], return_tensors="pt").to(dev)["input_ids"]
                for e in calib]
        seqs = [s for s in seqs if s.shape[1] >= 4]
        atp = B.head_attribution_patching(bundle, calib)
        eap = B.integrated_gradient_attribution(bundle, calib)
        atpstar = B.head_attribution_graddrop(bundle, calib)

        for pname in missing:
            primary = C.IOI_CIRCUIT[pname]
            prim_units = set(C.head_index(l, h, nH) for (l, h) in primary)
            cand = [u for u in range(nU) if u not in prim_units]
            co = CoAblation(bundle, seqs, top_r=top_r, position_mode=args.position_mode)
            comp = co.conditional_compensation(primary, head_set=list(range(nU)))
            scores = {
                "coax": np.array([comp["compensation"][u] for u in cand]),
                "single": np.array([comp["single"][u] for u in cand]),
                "atp": np.array([atp[u] for u in cand]),
                "eapig": np.array([eap[u] for u in cand]),
                "atpstar": np.array([atpstar[u] for u in cand]),
            }
            scores = {k: np.nan_to_num(v, nan=0.0) for k, v in scores.items()}

            if SEED_METRIC[pname] == "ioi":
                valid = make_prompts(template, args.num_prompts, args.valid_seed)
                res = FreezeValidator(bundle, valid).repair_removed(primary)
            else:
                res = ind_validator.repair_removed(primary, cand)
            rr = np.array([res["repair_removed"][u] for u in cand])
            row = {"metric": SEED_METRIC[pname], "clean": res["clean"],
                   "seed_ablated": res["seed_ablated"], "by_method": {}}
            for m in METHODS:
                row["by_method"][m] = {"spearman": float(spearmanr(scores[m], rr).statistic)}
            report["instances"][f"{tname}/{pname}"] = row
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[mm-hier] {tname}/{pname:15s} metric={row['metric']:9s} " +
                  " ".join(f"{m}={row['by_method'][m]['spearman']:+.3f}" for m in METHODS),
                  flush=True)

    report["summary"] = nested_summary(report["instances"], args.bootstrap)
    print("\n=== matched-metric nested summary ===")
    for m in METHODS:
        s = report["summary"]
        ci = s["nested_ci"][m]
        loo = s["loo"][m]
        sign = s["sign"][m]
        ps = s["per_primary_set"][m]
        print(f"{m:8s} dup={ps['duplicate_token']:+.3f} ind={ps['induction']:+.3f} "
              f"name={ps['name_mover']:+.3f} s-inhib={ps['s_inhibition']:+.3f} "
              f"macro={s['macro'][m]:+.3f} CI=[{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"LOO=[{loo['min']:+.3f},{loo['max']:+.3f}] "
              f"inst+={sign['positive_instances']}/{sign['n_instances']} "
              f"clusters+={sign['positive_primary_sets']}/{sign['n_primary_sets']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[mm-hier] wrote {args.out}")


if __name__ == "__main__":
    main()
