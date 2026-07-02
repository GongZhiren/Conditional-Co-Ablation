#!/usr/bin/env python3
"""Table 3 --- attribution recovery: the effect self-repair masks.

Ablating the name-mover primaries alone drops the IOI logit-difference only slightly (their
backups absorb the damage). Re-attributing the primaries together with the label-free CoAx
backups recovers the true, much larger drop --- exceeding the matched-random and documented
top-ups.

    python experiments/attribution.py --num-prompts 96

Approximate expected logit-difference drop (clean ~2.5): prim-only ~0.2 · +random ~1.0 ·
+doc ~1.15 · +CoAx ~1.76.
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import recovered_backups
from coax import CoAx, Model
from coax.ioi import BACKUPS, PRIMARIES, ioi_examples, ioi_logit_diff, ioi_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--top-r", type=int, default=192)
    ap.add_argument("--seeds", default="42,1,8,22", help="comma-separated (paper: 4-seed mean)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    model = Model(args.model)
    k = len(BACKUPS)

    prompts = ioi_prompts(args.num_prompts, seed=seeds[0])
    coax = CoAx(model, prompts, top_r=args.top_r)
    res = coax.conditional_compensation(PRIMARIES)
    coax_set = recovered_backups(model, coax, res["compensation"], PRIMARIES, k)
    rng = np.random.default_rng(seeds[0])
    cand = [model.layer_head(u) for u in range(model.num_units) if model.layer_head(u) not in set(PRIMARIES)]
    rand_set = [cand[i] for i in rng.choice(len(cand), size=k, replace=False)]

    example_sets = [ioi_examples(args.num_prompts, seed=s) for s in seeds]
    clean = np.mean([ioi_logit_diff(model, ex) for ex in example_sets])
    drop = lambda ablate: clean - np.mean([ioi_logit_diff(model, ex, ablate) for ex in example_sets])
    sets = {
        "primaries only":         list(PRIMARIES),
        "+ random":               list(PRIMARIES) + rand_set,
        "+ documented backups":   list(PRIMARIES) + list(BACKUPS),
        "+ CoAx backups":         list(PRIMARIES) + coax_set,
    }
    bar = "=" * 54
    print(f"\n{bar}\n  IOI logit-difference drop (clean {clean:.2f}, mean over {len(seeds)} seeds)\n{bar}")
    for name, ablate in sets.items():
        print(f"  {name:26s} {drop(ablate):+.2f}")
    print(bar)
    print("  The masked effect (primaries only) is small; adding the CoAx backups recovers it.")


if __name__ == "__main__":
    main()
