#!/usr/bin/env python3
"""Fail on common source-release mistakes without reading ignored artifacts."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".sh", ".json", ".cff"}
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
        ROOT / "experiments/paper/causal_freezing.py",
        ROOT / "experiments/paper/circuit_completion.py",
        ROOT / "experiments/paper/knockout_oracle_distance.py",
        ROOT / "experiments/paper/mechanism_handoff.py",
        ROOT / "experiments/paper/cross_model_completion.py",
        ROOT / "experiments/paper/matched_intervention_panel.py",
        ROOT / "results/panel_metric_match_hierarchical.json",
        ROOT / "assets/fig1.png",
    ]
    failures.extend(f"missing required file: {p.relative_to(ROOT)}" for p in required if not p.is_file())

    if failures:
        print("Release check failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Release check passed: required files present; no forbidden source patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
