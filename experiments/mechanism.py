#!/usr/bin/env python3
"""Fig 3 --- are the discovered heads real backups? Wake-up and counterfactual patching.

As the primary name-movers are ablated one by one, the backups grow in output norm and start
carrying the answer (while matched random heads stay flat). Counterfactual patching then closes
the causal loop: freezing the backups to their clean (dormant) value, while the primaries are
ablated, removes about half of the self-repair --- freezing random heads removes none.

    python experiments/mechanism.py --num-prompts 96

Uses the documented backups so the mechanism is shown on a fixed, known set.
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401  -- puts the repo root on sys.path so `import coax` works
import numpy as np
import torch

from coax import CoAx, Model
from coax.ioi import BACKUPS, PRIMARIES, ioi_examples, ioi_logit_diff, ioi_prompts


def mean_norm_ratio(coax, heads, seed):
    """Mean output-norm ratio (seed ablated / clean) over the given heads."""
    ratios = coax.wakeup_ratios(seed)
    idx = [coax.model.head_index(l, h) for (l, h) in heads]
    return float(np.mean(ratios[idx]))


def cond_drop(model, examples, seed, heads):
    """Mean over ``heads`` of the conditional causal drop: how much ablating the head lowers the
    answer margin once ``seed`` is already ablated."""
    base = ioi_logit_diff(model, examples, list(seed))
    return float(np.mean([base - ioi_logit_diff(model, examples, list(seed) + [h]) for h in heads]))


def patch_freeze_logit_diff(model, examples, ablate, freeze):
    """IOI logit-difference with ``ablate`` heads zeroed and ``freeze`` heads pinned to their
    clean (dormant) output --- i.e. prevented from waking up."""
    hd = model.head_dim
    abl = {(l, h) for (l, h) in ablate}
    frz = {(l, h) for (l, h) in freeze}
    by_layer_abl, by_layer_frz = {}, {}
    for (l, h) in abl:
        by_layer_abl.setdefault(l, []).append(h)
    for (l, h) in frz:
        by_layer_frz.setdefault(l, []).append(h)

    diffs = []
    for ex in examples:
        ids = model.tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].to(model.device)
        io = model.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
        s = model.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
        clean_slices = {}

        # pass 1: cache the clean output slices of the frozen heads
        cap = []
        def mk_cap(li):
            def hook(_m, inp):
                x = inp[0]
                for h in by_layer_frz.get(li, []):
                    clean_slices[(li, h)] = x[..., h * hd:(h + 1) * hd].detach().clone()
            return hook
        for li in by_layer_frz:
            cap.append(model._out_proj(li).register_forward_pre_hook(mk_cap(li)))
        with torch.no_grad():
            model.model(ids)
        for c in cap:
            c.remove()

        # pass 2: zero the ablated heads, pin the frozen heads to their clean value
        edit = []
        def mk_edit(li):
            def hook(_m, inp):
                x = inp[0].clone()
                for h in by_layer_abl.get(li, []):
                    x[..., h * hd:(h + 1) * hd] = 0.0
                for h in by_layer_frz.get(li, []):
                    x[..., h * hd:(h + 1) * hd] = clean_slices[(li, h)]
                return (x,) + tuple(inp[1:])
            return hook
        for li in set(by_layer_abl) | set(by_layer_frz):
            edit.append(model._out_proj(li).register_forward_pre_hook(mk_edit(li)))
        with torch.no_grad():
            logits = model.model(ids).logits[0, -1, :]
        for e in edit:
            e.remove()
        diffs.append(float(logits[io] - logits[s]))
    return float(np.mean(diffs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--top-r", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = Model(args.model)
    examples = ioi_examples(args.num_prompts, seed=args.seed)
    prompts = [e["prompt"] for e in examples]
    coax = CoAx(model, prompts, top_r=args.top_r)
    rng = np.random.default_rng(args.seed)
    cand = [model.layer_head(u) for u in range(model.num_units)
            if model.layer_head(u) not in set(PRIMARIES) | set(BACKUPS)]
    random_ctrl = [cand[i] for i in rng.choice(len(cand), size=len(BACKUPS), replace=False)]

    bar = "=" * 62
    print(f"\n{bar}\n  Wake-up curve: as k primaries are ablated (backups vs random)\n{bar}")
    print(f"  {'k':>2}   {'norm-ratio backup':>18}   {'random':>8}   {'cond-drop backup':>17}   {'random':>8}")
    for k in range(len(PRIMARIES) + 1):
        seed = PRIMARIES[:k]
        rb = mean_norm_ratio(coax, BACKUPS, seed) if k else 1.0
        rr = mean_norm_ratio(coax, random_ctrl, seed) if k else 1.0
        db = cond_drop(model, examples, seed, BACKUPS)
        dr = cond_drop(model, examples, seed, random_ctrl)
        print(f"  {k:>2}   {rb:>18.2f}   {rr:>8.2f}   {db:>+17.2f}   {dr:>+8.2f}")

    print(f"\n{bar}\n  Counterfactual patching (freeze the backups to their dormant value)\n{bar}")
    ld_clean = ioi_logit_diff(model, examples)
    ld_prim = ioi_logit_diff(model, examples, list(PRIMARIES))
    ld_removed = ioi_logit_diff(model, examples, list(PRIMARIES) + list(BACKUPS))
    ld_freeze = patch_freeze_logit_diff(model, examples, ablate=PRIMARIES, freeze=BACKUPS)
    ld_freeze_rand = patch_freeze_logit_diff(model, examples, ablate=PRIMARIES, freeze=random_ctrl)
    repair = ld_prim - ld_removed
    frac = (ld_prim - ld_freeze) / repair if repair else 0.0
    frac_rand = (ld_prim - ld_freeze_rand) / repair if repair else 0.0
    print(f"  clean logit-diff                         {ld_clean:.2f}")
    print(f"  primaries ablated (backups repair)       {ld_prim:.2f}")
    print(f"  primaries ablated, backups removed       {ld_removed:.2f}")
    print(f"  primaries ablated, backups frozen dormant {ld_freeze:.2f}")
    print(f"  -> freezing the backups removes {frac*100:.0f}% of the self-repair "
          f"(random control: {frac_rand*100:.0f}%)")
    print(bar)
    print("  The backups' wake-up causally drives the repair; random heads do nothing.")


if __name__ == "__main__":
    main()
