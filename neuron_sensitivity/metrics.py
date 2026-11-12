from __future__ import annotations

import torch
from torch import nn


@torch.inference_mode()
def measure(model: nn.Module, baseline_state: dict[str, torch.Tensor], layer: str,
            neuron: int, method: str, strength: float, seed: int,
            x: torch.Tensor, y: torch.Tensor, device: torch.device,
            baseline_logits: torch.Tensor | None = None) -> dict:
    model.load_state_dict(baseline_state)
    base = model(x.to(device)) if baseline_logits is None else baseline_logits
    base_probs = base.softmax(dim=1)
    base_pred = base.argmax(dim=1)
    from .perturb import neuron_weight, perturb_neuron
    row_norm = neuron_weight(model, layer, neuron).norm().item()
    delta = perturb_neuron(model, layer, neuron, method, strength, seed)
    changed = model(x.to(device))
    pred = changed.argmax(dim=1)
    probs = changed.softmax(dim=1)
    flips = (pred != base_pred)
    result = {
        "layer": layer, "neuron": neuron, "method": method,
        "strength": strength, "seed": seed,
        "baseline_accuracy": (base_pred == y.to(device)).float().mean().item(),
        "perturbed_accuracy": (pred == y.to(device)).float().mean().item(),
        "prediction_flips": int(flips.sum().item()),
        "prediction_flip_pct": flips.float().mean().item() * 100,
        "mean_abs_predicted_class_probability_change":
            (probs.gather(1, base_pred[:, None]) - base_probs.gather(1, base_pred[:, None])).abs().mean().item(),
        "perturbation_magnitude": delta,
        "neuron_weight_norm": row_norm,
    }
    model.load_state_dict(baseline_state)
    return result
