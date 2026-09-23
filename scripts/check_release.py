#!/usr/bin/env python3
"""Fail on common source-release mistakes without reading ignored artifacts."""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".sh", ".json", ".cff", ".ipynb"
}
FORBIDDEN = {
    "local absolute path": re.compile(
        "(?:" + "/" + "scratch/|" + "/" + "home/|" + "/" + "Users/)"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "submission metadata": re.compile(
        r"\b(?:" + "IC" + r"LR|Open" + r"Review)\b", re.IGNORECASE
    ),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
}


def tracked_or_source_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return [ROOT / line for line in proc.stdout.splitlines() if line]


def main() -> int:
    failures: list[str] = []
    for path in tracked_or_source_files():
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            failures.append(f"{rel}: symbolic links are not allowed in the source release")
            continue
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            failures.append(f"{rel}: source-release file exceeds 5 MiB")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{rel}: contains {label}")

    required = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "LICENSE",
        ROOT / "requirements.txt",
        ROOT / "configs/model.yaml",
        ROOT / "src/curvgraph/coablation.py",
        ROOT / "experiments/paper/backup_recovery_full.py",
        ROOT / "experiments/paper/conditional_gradient_contrast.py",
        ROOT / "experiments/paper/causal_freezing.py",
        ROOT / "experiments/paper/circuit_completion.py",
        ROOT / "experiments/paper/knockout_oracle_distance.py",
        ROOT / "experiments/paper/mechanism_handoff.py",
        ROOT / "experiments/paper/cross_model_completion.py",
        ROOT / "experiments/paper/matched_intervention_panel.py",
        ROOT / "results/panel_metric_match_hierarchical.json",
        ROOT / "assets/fig1.png",
        ROOT / "notebooks/coax_quickstart.ipynb",
        ROOT / ".github/workflows/release-check.yml",
    ]
    failures.extend(f"missing required file: {p.relative_to(ROOT)}" for p in required if not p.is_file())

    reference_path = ROOT / "results/reference_metrics.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        cross = reference.get("cross_model_completion", {})
        protocol = cross.get("protocol", {})
        expected_protocol = {
            "n_detect": 32, "n_calib": 16, "n_eval": 64,
            "sequence_length": 48, "n_primary": 4, "top_k": 10, "n_random": 5,
        }
        if reference.get("schema_version") != 3:
            failures.append("results/reference_metrics.json: expected schema version 3")
        if any(protocol.get(key) != value for key, value in expected_protocol.items()):
            failures.append("results/reference_metrics.json: cross-model protocol is inconsistent")
        order = cross.get("selector_order", [])
        rows = cross.get("selectors", {})
        if order != ["coax", "single_ablation", "coactivation", "atp",
                     "atp_star_graddrop", "role_matched_own"] or len(rows) != 8:
            failures.append("results/reference_metrics.json: incomplete cross-model table")
        elif any(len(row) != len(order) for row in rows.values()):
            failures.append("results/reference_metrics.json: malformed cross-model row")
        else:
            counts = cross.get("counts", {})
            above_random = sum(row[0] > 1.0 for row in rows.values())
            above_own = sum(row[0] > row[-1] for row in rows.values())
            if counts.get("coax_above_random") != [above_random, len(rows)]:
                failures.append("results/reference_metrics.json: above-random count is stale")
            if counts.get("coax_above_role_matched_own") != [above_own, len(rows)]:
                failures.append("results/reference_metrics.json: above-own count is stale")
            families = [["pythia-160m", "pythia-410m", "pythia-1.4b"],
                        ["gpt-neo-1.3b"], ["gemma-2-2b"], ["qwen2.5-7b"],
                        ["olmo-2-7b"], ["llama-3.1-8b"]]
            for column, name in enumerate(order):
                macro = sum(sum(rows[m][column] for m in family) / len(family)
                            for family in families) / len(families)
                if not math.isclose(macro, cross.get("family_macro", {}).get(name, math.nan),
                                    rel_tol=0.0, abs_tol=1e-12):
                    failures.append(f"results/reference_metrics.json: stale family macro for {name}")

    if failures:
        print("Release check failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Release check passed: required files present; no forbidden source patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
