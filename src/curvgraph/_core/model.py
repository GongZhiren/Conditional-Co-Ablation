from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import to_torch_dtype


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str
    num_layers: int
    num_heads: int
    hidden_size: int
    head_dim: int
    intermediate_size: int
    kv_heads: int
    attn_pattern: str
    family: str
    model_path: str


def _layer_stack(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX / Pythia
    raise ValueError("Unsupported model architecture for layer extraction")


def load_model_bundle(model_cfg: Dict[str, object], tokenizer_cfg: Dict[str, object]) -> ModelBundle:
    path = str(model_cfg["path"])
    dtype = to_torch_dtype(str(model_cfg.get("torch_dtype", "bfloat16")))
    device_map = model_cfg.get("device_map", "auto")
    local_only = bool(model_cfg.get("local_files_only", False))
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "device_map": device_map,
        "attn_implementation": "eager",
    }
    if model_cfg.get("load_in_8bit", False):
        model_kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=bool(tokenizer_cfg.get("trust_remote_code", True)),
        use_fast=bool(tokenizer_cfg.get("use_fast", True)),
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=local_only, **model_kwargs)
    model.eval()
    layers = _layer_stack(model)
    config = model.config
    num_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 0)))
    hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    # Some configs (e.g. Mistral-7B-v0.3) define head_dim but set it to null;
    # getattr then returns None, not the fallback, so guard explicitly.
    _head_dim = getattr(config, "head_dim", None)
    head_dim = int(_head_dim) if _head_dim else hidden_size // max(1, num_heads)
    intermediate_size = int(getattr(config, "intermediate_size", None) or hidden_size * 4)
    kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    if kv_heads == num_heads:
        attn_pattern = "mha"
    elif kv_heads == 1:
        attn_pattern = "mqa"
    else:
        attn_pattern = "gqa"

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_layers=len(layers),
        num_heads=num_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        kv_heads=kv_heads,
        attn_pattern=attn_pattern,
        family=str(model_cfg.get("family", "unknown")),
        model_path=path,
    )


def layer_modules(bundle: ModelBundle):
    return _layer_stack(bundle.model)


def bundle_device(bundle: ModelBundle) -> torch.device:
    if hasattr(bundle.model, "hf_device_map") and getattr(bundle.model, "hf_device_map"):
        try:
            embed = bundle.model.get_input_embeddings()
            return next(embed.parameters()).device
        except Exception:
            pass
    return next(bundle.model.parameters()).device
