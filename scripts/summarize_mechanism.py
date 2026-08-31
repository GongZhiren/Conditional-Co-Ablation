#!/usr/bin/env python3
"""Aggregate the four per-seed mechanism artifacts used in the paper figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mean_std(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/coablation/mechanism_summary.json"))
    args = ap.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not runs:
        raise SystemExit("at least one mechanism artifact is required")
    if any(len(run["wakeup_curve"]) != len(runs[0]["wakeup_curve"]) for run in runs):
        raise ValueError("mechanism artifacts have incompatible wake-up curves")

    curve = []
    fields = [key for key in runs[0]["wakeup_curve"][0] if key != "k"]
    for i, row0 in enumerate(runs[0]["wakeup_curve"]):
        row = {"k": row0["k"]}
        for field in fields:
            row[field] = mean_std([run["wakeup_curve"][i][field] for run in runs])
        curve.append(row)

    dla_per_head = {
        field: mean_std([run["dla"][field] for run in runs])
        for field in runs[0]["dla"]
    }
    # The paper reports branch totals for the 3 primaries and 8 documented backups.
    dla_branch_totals = {
        "name_mover_clean": mean_std([3 * run["dla"]["name_mover_clean"] for run in runs]),
        "backup_clean": mean_std([8 * run["dla"]["backup_clean"] for run in runs]),
        "backup_primaries_ablated": mean_std(
            [8 * run["dla"]["backup_primaries_ablated"] for run in runs]
        ),
        "other_clean": mean_std([133 * run["dla"]["other_clean"] for run in runs]),
        "other_primaries_ablated": mean_std(
            [133 * run["dla"]["other_primaries_ablated"] for run in runs]
        ),
    }
    report = {
        "schema_version": 1,
        "inputs": [path.name for path in args.inputs],
        "n_seeds": len(runs),
        "wakeup_curve": curve,
        "dla_per_head": dla_per_head,
        "dla_branch_totals": dla_branch_totals,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[mechanism-summary] wrote {args.out}")


if __name__ == "__main__":
    main()
