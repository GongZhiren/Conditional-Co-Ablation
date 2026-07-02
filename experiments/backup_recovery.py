#!/usr/bin/env python3
"""Table 1 --- backup-name-mover recovery on GPT-2-small IOI, all methods.

Ranks the 141 non-primary heads by each score and reports the ROC-AUC of the eight documented
backups. Reproduces the headline table: every additive / gradient / self-repair-aware score
falls short, while CoAx (conditional growth) recovers the backups.

    python experiments/backup_recovery.py                    # gpt2 from the HF hub
    python experiments/backup_recovery.py --model <path> --num-prompts 96

Approximate expected AUCs (GPT-2-small, 96 prompts):
    single 0.33 · AtP 0.59 · GIM(seeded) 0.62 · AtP* GradDrop 0.79 · CoAx 0.91
    (co-activation control 0.93 --- ranks them high but over-ablates downstream, see knockout.py)
"""
from __future__ import annotations

import argparse

from _common import backup_auc
from coax import CoAx, Model
from coax import baselines as B
from coax.ioi import BACKUPS, PRIMARIES, ioi_examples, ioi_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num-prompts", type=int, default=96)
    ap.add_argument("--top-r", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="skip the slow gradient baselines")
    args = ap.parse_args()

    model = Model(args.model)
    examples = ioi_examples(args.num_prompts, seed=args.seed)
    prompts = [e["prompt"] for e in examples]
    print(f"Scoring {model.num_units} heads on {len(examples)} IOI prompts (device {model.device}) ...")

    rows = []
    coax = CoAx(model, prompts, top_r=args.top_r)
    res = coax.conditional_compensation(PRIMARIES)
    rows.append(("single ablation      (1st)", res["single"]))
    if not args.fast:
        rows.append(("AtP                  (1st)", B.atp_scores(model, examples)))
        rows.append(("AtP* GradDrop        (1st)", B.graddrop_scores(model, examples)))
        rows.append(("GIM, seeded          (1st)", B.gim_conditional_scores(model, examples, PRIMARIES)))
    rows.append(("CoAx                 (2nd)", res["compensation"]))
    rows.append(("co-activation control     ", B.coactivation_scores(model, prompts, PRIMARIES)))

    bar = "=" * 52
    print(f"\n{bar}\n  Backup recovery ROC-AUC  (8 backups, 141 candidates)\n{bar}")
    for name, score in rows:
        print(f"  {name}   {backup_auc(score, model, BACKUPS, exclude=PRIMARIES):.3f}")
    print(bar)
    print("  Labels are used only to score the ranking; the methods never see them.")


if __name__ == "__main__":
    main()
