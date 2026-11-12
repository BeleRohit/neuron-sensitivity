from __future__ import annotations

import torch
from torch import nn


LAYERS = {"fc1": "fc1", "fc2": "fc2"}


def neuron_weight(model: nn.Module, layer: str, neuron: int) -> torch.Tensor:
    return getattr(model, LAYERS[layer]).weight[neuron]


@torch.no_grad()
def perturb_neuron(
    model: nn.Module, layer: str, neuron: int, method: str,
    strength: float, seed: int = 0,
) -> float:
    """Modify one incoming weight vector; return its applied L2 delta norm.

    Noise uses delta norm = strength * original norm. Scaling uses relative
    change strength (factor 1+strength). Ablation zeros the incoming row and
    bias, making the hidden unit output identically zero after ReLU.
    """
    module = getattr(model, LAYERS[layer])
    row = module.weight[neuron]
    original_norm = row.norm().item()
    before = row.clone()
    if method == "noise":
        generator = torch.Generator(device=row.device).manual_seed(seed)
        noise = torch.randn(row.shape, generator=generator, device=row.device, dtype=row.dtype)
        noise_norm = noise.norm().clamp_min(torch.finfo(row.dtype).eps)
        row.add_(noise * (strength * original_norm / noise_norm))
    elif method == "scale":
        row.mul_(1.0 + strength)
    elif method == "ablation":
        row.zero_()
        if module.bias is not None:
            module.bias[neuron].zero_()
    else:
        raise ValueError(f"Unknown perturbation method: {method}")
    return (row - before).norm().item()
