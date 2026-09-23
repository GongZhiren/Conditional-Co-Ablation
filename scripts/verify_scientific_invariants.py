#!/usr/bin/env python3
"""Fast, model-free invariants for scientifically consequential interventions."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from curvgraph import baselines as B
from curvgraph import circuits as C


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_proj = torch.nn.Identity()


class _FakeLayer(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.attn = _FakeAttention()
        self.offset = offset

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attn.c_proj(hidden + self.offset)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(2))
        self.transformer = SimpleNamespace(
            h=torch.nn.ModuleList([_FakeLayer(1.0), _FakeLayer(2.0)])
        )

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2) * self.scale
        for layer in self.transformer.h:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)


class _ResidualBlock(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 2.0 * hidden


def check_layerwise_head_means() -> None:
    bundle = SimpleNamespace(
        model=_FakeModel(), num_layers=2, num_heads=2, head_dim=1
    )
    means = C.head_mean_vectors(bundle, [torch.tensor([[1, 3]])])
    expected = {(0, 0): 3.0, (0, 1): 3.0, (1, 0): 5.0, (1, 1): 5.0}
    for key, value in expected.items():
        got = float(means[key].item())
        if not abs(got - value) < 1e-7:
            raise AssertionError(f"layerwise mean {key}: expected {value}, got {got}")


def check_residual_contribution_graddrop() -> None:
    block = _ResidualBlock()
    clean_x = torch.tensor([2.0], requires_grad=True)
    clean_y = block(clean_x)
    clean_y.backward()
    if not torch.equal(clean_y.detach(), torch.tensor([6.0])):
        raise AssertionError("unexpected toy clean forward")
    if not torch.equal(clean_x.grad, torch.tensor([3.0])):
        raise AssertionError("unexpected toy clean gradient")

    dropped_x = torch.tensor([2.0], requires_grad=True)
    handles = B._register_residual_contribution_graddrop(block)
    try:
        dropped_y = block(dropped_x)
        dropped_y.backward()
    finally:
        for handle in handles:
            handle.remove()
    if not torch.equal(dropped_y.detach(), clean_y.detach()):
        raise AssertionError("GradDrop changed the forward value")
    if not torch.equal(dropped_x.grad, torch.tensor([1.0])):
        raise AssertionError("GradDrop failed to preserve the identity residual gradient")


def check_one_step_eap_matches_atp() -> None:
    bundle = SimpleNamespace(
        model=_FakeModel(), num_layers=2, num_heads=2, head_dim=1
    )
    prompts = [{"ids": torch.tensor([[1, 3]])}]

    def metric(bundle_, example):
        return bundle_.model(example["ids"], use_cache=False).logits.sum()

    atp = B.head_attribution_patching(bundle, prompts, metric=metric)
    eap = B.integrated_gradient_attribution(bundle, prompts, steps=1, metric=metric)
    if not torch.allclose(torch.from_numpy(eap), torch.from_numpy(atp), atol=1e-7, rtol=0.0):
        raise AssertionError(f"one-step activation EAP-IG did not equal AtP: {eap} vs {atp}")


def check_method_specific_dataset_aggregation() -> None:
    """AtP averages magnitudes; official EAP-IG ranks the signed dataset aggregate."""
    bundle = SimpleNamespace(
        model=_FakeModel(), num_layers=2, num_heads=2, head_dim=1
    )
    ids = torch.tensor([[1, 3]])
    prompts = [{"ids": ids, "sign": 1.0}, {"ids": ids, "sign": -1.0}]

    def metric(bundle_, example):
        return example["sign"] * bundle_.model(
            example["ids"], use_cache=False
        ).logits.sum()

    atp = B.head_attribution_patching(bundle, prompts, metric=metric)
    eap = B.integrated_gradient_attribution(bundle, prompts, steps=1, metric=metric)
    if not (atp > 0).all():
        raise AssertionError(f"AtP cancelled opposite-signed examples: {atp}")
    if not torch.allclose(torch.from_numpy(eap), torch.zeros_like(torch.from_numpy(eap)),
                          atol=1e-7, rtol=0.0):
        raise AssertionError(f"EAP-IG did not preserve signed dataset aggregation: {eap}")


def check_sharded_intervention_is_rejected() -> None:
    """A multi-device dispatch must fail rather than silently ignore an ablation hook."""
    model = _FakeModel()
    model.hf_device_map = {"": "cpu"}
    C.ensure_intervention_safe(SimpleNamespace(model=model))
    model.hf_device_map = {"transformer.h.0": 0, "transformer.h.1": 1}
    bundle = SimpleNamespace(model=model)
    try:
        C.ensure_intervention_safe(bundle)
    except RuntimeError as error:
        if "one device" not in str(error):
            raise AssertionError(f"unexpected sharding error: {error}") from error
    else:
        raise AssertionError("multi-device intervention was not rejected")


def main() -> None:
    check_layerwise_head_means()
    check_residual_contribution_graddrop()
    check_one_step_eap_matches_atp()
    check_method_specific_dataset_aggregation()
    check_sharded_intervention_is_rejected()
    print("Scientific intervention invariants passed.")


if __name__ == "__main__":
    main()
