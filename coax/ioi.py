"""The GPT-2-small IOI task, its documented circuit, and the task metrics.

We reuse the Indirect-Object-Identification (IOI) task and the hand-verified circuit of
Wang et al., *Interpretability in the Wild* (2022) --- the one circuit with head-level
**backup** ground truth. Prompts follow the canonical ABB template
``When {IO} and {S} went to the {place}, {S} gave a {object} to`` and the model must
predict the indirect object ``{IO}``.

For CoAx we mainly need two groups: the ``name_mover`` heads are the **primary seed** (they
write the answer on the intact model), and the eight ``backup_name_mover`` heads are the
**dormant backups** we aim to recover --- silent on the clean model, load-bearing only once
the primaries are ablated.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------- prompt vocabulary
NAMES = ["John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Anna",
         "Paul", "Kate", "Mark", "Lucy", "Peter", "Sarah", "David", "Emma"]
PLACES = ["store", "park", "school", "office", "garden", "station", "market", "library"]
OBJECTS = ["drink", "book", "ball", "ring", "kiss", "snack", "bone", "note"]

# ---------------------------------------------------------------- documented circuit
# Heads are (layer, head). Source: Wang et al. (2022), Table 2.
IOI_CIRCUIT: Dict[str, List[Tuple[int, int]]] = {
    "name_mover":          [(9, 9), (10, 0), (9, 6)],
    "negative_name_mover": [(10, 7), (11, 10)],
    "s_inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction":           [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token":     [(0, 1), (0, 10), (3, 0)],
    "previous_token":      [(2, 2), (4, 11)],
    "backup_name_mover":   [(9, 0), (9, 7), (10, 1), (10, 2),
                            (10, 6), (10, 10), (11, 2), (11, 9)],
}
#: The primary seed CoAx conditions on (the heads that write the answer intact).
PRIMARIES: List[Tuple[int, int]] = IOI_CIRCUIT["name_mover"]
#: The eight dormant backups CoAx is asked to recover.
BACKUPS: List[Tuple[int, int]] = IOI_CIRCUIT["backup_name_mover"]


def ioi_examples(num_prompts: int = 96, seed: int = 42) -> List[Dict[str, str]]:
    """Sample IOI examples as dicts with the prompt and the IO / S answer tokens
    (leading-space form, as GPT-2 tokenizes them). ``seed`` controls the draw."""
    rng = np.random.default_rng(seed)
    out: List[Dict[str, str]] = []
    for _ in range(num_prompts):
        io, s = rng.choice(NAMES, size=2, replace=False)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        out.append({"prompt": f"When {io} and {s} went to the {place}, {s} gave a {obj} to",
                    "io": " " + io, "s": " " + s})
    return out


def ioi_prompts(num_prompts: int = 96, seed: int = 42) -> List[str]:
    """Just the prompt strings (what :class:`coax.CoAx` calibrates on)."""
    return [e["prompt"] for e in ioi_examples(num_prompts, seed)]


# ---------------------------------------------------------------- task metrics
def _answer_ids(model, ex: Dict[str, str]) -> Tuple[torch.Tensor, int, int]:
    ids = model.tokenizer(ex["prompt"], return_tensors="pt")["input_ids"]
    io = model.tokenizer(ex["io"], add_special_tokens=False)["input_ids"][0]
    s = model.tokenizer(ex["s"], add_special_tokens=False)["input_ids"][0]
    return ids, io, s


def ioi_logit_diff(model, examples: Sequence[Dict[str, str]],
                   ablate: Sequence[Tuple[int, int]] = ()) -> float:
    """Mean ``logit[IO] - logit[S]`` at the final position (higher = task intact)."""
    diffs = []
    for ex in examples:
        ids, io, s = _answer_ids(model, ex)
        logits = model.logits(ids, ablate=ablate)[0, -1, :]
        diffs.append(float(logits[io] - logits[s]))
    return float(np.mean(diffs)) if diffs else 0.0


def ioi_accuracy(model, examples: Sequence[Dict[str, str]],
                 ablate: Sequence[Tuple[int, int]] = ()) -> float:
    """Fraction of examples with ``logit[IO] > logit[S]`` (the behavior is correct)."""
    correct = []
    for ex in examples:
        ids, io, s = _answer_ids(model, ex)
        logits = model.logits(ids, ablate=ablate)[0, -1, :]
        correct.append(float(logits[io] > logits[s]))
    return float(np.mean(correct)) if correct else 0.0
