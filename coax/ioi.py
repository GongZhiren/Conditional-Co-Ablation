"""The GPT-2-small IOI task and its documented circuit.

We reuse the Indirect-Object-Identification (IOI) task and the hand-verified circuit of
Wang et al., *Interpretability in the Wild* (2022) --- the one circuit with head-level
**backup** ground truth. Prompts follow the canonical ABB template
``When {IO} and {S} went to the {place}, {S} gave a {object} to`` and the model must
predict the indirect object ``{IO}``.

The circuit below is quoted verbatim from Wang et al. (2022). For CoAx we only need two
groups: the ``name_mover`` heads are the **primary seed** (they write the answer on the
intact model), and the eight ``backup_name_mover`` heads are the **dormant backups** we aim
to recover --- silent on the clean model, load-bearing only once the primaries are ablated.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

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


def ioi_prompts(num_prompts: int = 48, seed: int = 42) -> List[str]:
    """Sample ``num_prompts`` IOI prompts. ``seed`` controls the name/place/object draw;
    vary it across runs for multi-seed error bars (the paper averages over four seeds)."""
    rng = np.random.default_rng(seed)
    prompts: List[str] = []
    for _ in range(num_prompts):
        io, s = rng.choice(NAMES, size=2, replace=False)
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)
        prompts.append(f"When {io} and {s} went to the {place}, {s} gave a {obj} to")
    return prompts
