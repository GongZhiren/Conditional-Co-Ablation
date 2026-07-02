#!/usr/bin/env python3
"""Table 4 / Fig 1c --- circuit knockout: a capability needs its backups removed.

Ablating the documented name-mover primaries barely dents IOI accuracy (self-repair). Adding
the label-free CoAx backups is what removes the behavior, matching the documented-backup oracle
--- while a first-order top-up of the same size overshoots into the core name-movers.

    python experiments/knockout.py --num-prompts 96

Approximate expected accuracy: clean 1.00 · -prim 0.97 · +CoAx 0.70 (oracle +doc 0.72) ·
+own 0.24 (over-ablates) · +rand 0.81.
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import recovered_backups, top_units
from coax import CoAx, Model
from coax.ioi import BACKUPS, PRIMARIES, ioi_accuracy, ioi_examples, ioi_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--top-r", type=int, default=192)
    ap.add_argument("--seeds", default="42,1,8,22", help="comma-separated (paper: 4-seed mean)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    model = Model(args.model)
    k = len(BACKUPS)                                        # matched set size (8)

    # Recover the backup set once (it is seed-stable, std <= 0.04 in the paper).
    prompts = ioi_prompts(args.num_prompts, seed=seeds[0])
    coax = CoAx(model, prompts, top_r=args.top_r)
    res = coax.conditional_compensation(PRIMARIES)
    coax_set = recovered_backups(model, coax, res["compensation"], PRIMARIES, k)
    own_set = top_units(res["single"], model, exclude=PRIMARIES, k=k)
    rng = np.random.default_rng(seeds[0])
    cand = [model.layer_head(u) for u in range(model.num_units) if model.layer_head(u) not in set(PRIMARIES)]
    rand_set = [cand[i] for i in rng.choice(len(cand), size=k, replace=False)]

    sets = {
        "clean (no ablation)":        [],
        "-prim (name-movers only)":   list(PRIMARIES),
        "+CoAx backups":              list(PRIMARIES) + coax_set,
        "+doc backups (oracle)":      list(PRIMARIES) + list(BACKUPS),
        "+own (first-order top-up)":  list(PRIMARIES) + own_set,
        "+random":                    list(PRIMARIES) + rand_set,
    }
    # Accuracy is averaged over the seed prompt sets (the paper's protocol).
    example_sets = [ioi_examples(args.num_prompts, seed=s) for s in seeds]
    bar = "=" * 52
    print(f"\n{bar}\n  IOI accuracy under ablation (mean over {len(seeds)} seeds)\n{bar}")
    for name, ablate in sets.items():
        acc = np.mean([ioi_accuracy(model, ex, ablate) for ex in example_sets])
        print(f"  {name:30s} {acc:.2f}")
    print(bar)
    print("  Ordering is the result: +own overshoots (wrong heads); +CoAx matches the oracle.")


if __name__ == "__main__":
    main()
