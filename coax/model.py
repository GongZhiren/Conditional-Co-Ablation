"""A thin, frozen wrapper around a Hugging Face decoder-only transformer.

A **unit** is an attention head. Ablating a head zeros its slice of the attention
output-projection (``c_proj``) input --- i.e. it removes exactly that head's contribution
to the residual stream, using nothing but a plain ``forward_pre_hook``. No fine-tuning,
gradients, or task labels are ever used.

The wrapper is deliberately minimal: it exposes ``logits(input_ids, ablate=...)`` and the
head-index bookkeeping the scorer needs, and nothing else. It works out of the box on
GPT-2 and, via the small architecture map below, on other decoder-only families.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Attention submodule and its output projection, per architecture family. The output
# projection's input is the concatenated per-head output, so zeroing a head's slice of it
# ablates exactly that head.
_ATTN_NAMES = ("attn", "self_attn", "attention")
_OUT_PROJ_NAMES = ("c_proj", "o_proj", "out_proj", "dense")


def _first_attr(obj, names):
    for name in names:
        mod = getattr(obj, name, None)
        if mod is not None:
            return mod
    return None


class Model:
    """Frozen HF causal LM with per-head ablation hooks."""

    def __init__(self, name_or_path: str = "gpt2", device: str | None = None,
                 dtype: torch.dtype = torch.float32):
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Weights are never updated (no optimizer is ever created); we keep grad *enabled* so the
        # gradient baselines can backprop the task metric to the activations. The CoAx scoring path
        # runs under torch.no_grad (see `logits`), so it pays nothing for this.

        cfg = self.model.config
        self.num_layers = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
        self.num_heads = int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)))
        hidden = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))
        self.head_dim = hidden // self.num_heads
        self.num_units = self.num_layers * self.num_heads
        self._layers = self._locate_layers()

    # ------------------------------------------------------------------ bookkeeping
    def head_index(self, layer: int, head: int) -> int:
        return layer * self.num_heads + head

    def layer_head(self, unit: int) -> Tuple[int, int]:
        return divmod(int(unit), self.num_heads)

    # ------------------------------------------------------------------ internals
    def _locate_layers(self) -> List[torch.nn.Module]:
        m = self.model
        for path in ("transformer.h", "model.layers", "gpt_neox.layers", "model.decoder.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError("Could not locate the transformer block list for this model.")

    def _out_proj(self, layer_idx: int):
        attn = _first_attr(self._layers[layer_idx], _ATTN_NAMES)
        proj = _first_attr(attn, _OUT_PROJ_NAMES)
        if proj is None:  # GPT-Neo nests the real attention one level down
            proj = _first_attr(getattr(attn, "attention", None), _OUT_PROJ_NAMES)
        if proj is None:
            raise RuntimeError(f"No attention output projection found in layer {layer_idx}.")
        return proj

    def _ablation_hooks(self, heads: Sequence[Tuple[int, int]]):
        by_layer: dict[int, List[int]] = {}
        for (layer, head) in heads:
            by_layer.setdefault(int(layer), []).append(int(head))
        hd = self.head_dim
        handles = []

        def make_hook(head_ids: List[int]):
            def pre_hook(_module, inputs):
                x = inputs[0].clone()
                for h in head_ids:
                    x[..., h * hd:(h + 1) * hd] = 0.0
                return (x,) + tuple(inputs[1:])
            return pre_hook

        for layer_idx, head_ids in by_layer.items():
            handles.append(self._out_proj(layer_idx).register_forward_pre_hook(make_hook(head_ids)))
        return handles

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, ablate: Iterable[Tuple[int, int]] = ()) -> torch.Tensor:
        """Logits for ``input_ids`` with the given heads ablated (empty = clean)."""
        heads = list(ablate)
        handles = self._ablation_hooks(heads) if heads else []
        try:
            return self.model(input_ids.to(self.device)).logits
        finally:
            for h in handles:
                h.remove()

    def encode(self, prompts: Sequence[str]) -> List[torch.Tensor]:
        """Tokenize prompts into a list of ``[1, T]`` id tensors (variable length)."""
        return [self.tokenizer(p, return_tensors="pt")["input_ids"] for p in prompts]
