"""Validated reuse of paper-protocol score artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


SCORE_KEYS = {
    "single": "single-ablation saliency (1st-order)",
    "conditional": "conditional energy (removed-state control)",
    "compensation": "conditional co-ablation (ours, signed growth)",
    "atp": "ATP (1st-order grad)",
    "conditional_gradient": "conditional AtP (removed-state gradient control)",
    "atpstar": "AtP* GradDrop (1st-order)",
}


def load_headline_vectors(path: str | Path, *, model: str, seed: int,
                          num_prompts: int, position_mode: str, top_r: int,
                          num_units: int) -> Optional[Dict[str, np.ndarray]]:
    """Load score vectors only when every scientific protocol field matches.

    Returns ``None`` for a missing or older artifact so callers can recompute safely.
    """
    source = Path(path)
    if not source.is_file():
        return None
    data = json.loads(source.read_text(encoding="utf-8"))
    expected = {
        "model": model,
        "seed": seed,
        "num_prompts": num_prompts,
        "position_mode": position_mode,
        "top_r": top_r,
    }
    if data.get("schema_version") != 3 or any(data.get(k) != v for k, v in expected.items()):
        return None
    scores = data.get("scores", {})
    candidates = scores.get("cand")
    if not isinstance(candidates, list) or any(key not in scores for key in SCORE_KEYS.values()):
        return None
    vectors: Dict[str, np.ndarray] = {}
    for short_name, artifact_name in SCORE_KEYS.items():
        values = scores[artifact_name]
        if len(values) != len(candidates):
            return None
        vector = np.full(num_units, np.nan, dtype=np.float64)
        vector[np.asarray(candidates, dtype=int)] = np.asarray(values, dtype=np.float64)
        vectors[short_name] = vector
    return vectors
