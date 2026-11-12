"""Five-seed, full-test-set neuron property and ablation experiment."""
from __future__ import annotations

import gc
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .model import MNISTMLP, choose_device, seed_everything
from .train import train_model


PHASE2_SEEDS = (42, 43, 44, 45, 46)
CONFIG = {
    "architecture": "784-128-64-10",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "cross_entropy",
    "epochs": 5,
    "batch_size": 128,
    "seeds": list(PHASE2_SEEDS),
    "test_examples": 10000,
    "ablation": "zero one hidden neuron's incoming weight row and bias; restore full baseline state after each intervention",
    "target": "fraction of test examples whose predicted class changes after ablation",
    "local_contribution": "mean absolute activation-times-local-logit-Jacobian over test examples and 10 logits; for fc1 the Jacobian includes the fc2 ReLU gate and is explicitly a first-order local approximation",
}


def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top = logits.topk(2, dim=1).values
    return top[:, 0] - top[:, 1]


def _true_label_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return true_logits - masked.max(dim=1).values


@torch.inference_mode()
def _baseline_features(
    model: MNISTMLP,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, float]]:
    """Extract baseline activations and local logit effects, without ablation."""
    model.eval()
    f1, f2 = model.fc1, model.fc2
    n1, n2 = f1.out_features, f2.out_features
    act_sum = {"fc1": torch.zeros(n1, device=device), "fc2": torch.zeros(n2, device=device)}
    active_count = {"fc1": torch.zeros(n1, device=device), "fc2": torch.zeros(n2, device=device)}
    local_sum = {"fc1": torch.zeros(n1, device=device), "fc2": torch.zeros(n2, device=device)}
    predictions, labels_all, top_margins, true_margins = [], [], [], []
    correct = total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    w2 = model.fc2.weight
    w3 = model.fc3.weight
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        a1 = f1(x.reshape(x.shape[0], 784))
        h1 = torch.relu(a1)
        a2 = f2(h1)
        h2 = torch.relu(a2)
        logits = model.fc3(h2)
        pred = logits.argmax(1)
        predictions.append(pred.cpu())
        labels_all.append(y.cpu())
        top_margins.append(_top2_margin(logits).cpu())
        true_margins.append(_true_label_margin(logits, y).cpu())
        correct += (pred == y).sum().item()
        total += y.numel()

        act_sum["fc1"] += h1.sum(0)
        active_count["fc1"] += (h1 > 0).sum(0)
        act_sum["fc2"] += h2.sum(0)
        active_count["fc2"] += (h2 > 0).sum(0)

        # Local Jacobian: d(logit_c)/d(h1_j) = sum_k W3[c,k] * 1[a2_k>0] * W2[k,j].
        # Multiplying by h1_j gives a first-order local contribution, not an exact
        # finite ablation effect, because changing h1 can change the fc2 ReLU gates.
        gated_w3 = w3.unsqueeze(0) * (a2 > 0).to(w3.dtype).unsqueeze(1)
        jacobian1 = torch.bmm(gated_w3, w2.expand(x.shape[0], -1, -1))
        local1 = (h1.unsqueeze(1) * jacobian1).abs().mean((0, 1))
        local_sum["fc1"] += local1 * y.numel()
        # fc2 feeds the final linear layer directly, so this is its exact additive
        # logit contribution at the unablated activation point.
        local2 = (h2.unsqueeze(1) * w3.unsqueeze(0)).abs().mean((0, 1))
        local_sum["fc2"] += local2 * y.numel()

    n = len(dataset)
    base_pred = torch.cat(predictions).numpy()
    labels_np = torch.cat(labels_all).numpy()
    margins = {
        "top2": torch.cat(top_margins).numpy(),
        "true": torch.cat(true_margins).numpy(),
    }
    props = {}
    layer_modules = {"fc1": model.fc1, "fc2": model.fc2}
    for layer, module in layer_modules.items():
        incoming = module.weight.norm(dim=1)
        if layer == "fc1":
            outgoing = model.fc2.weight.norm(dim=0)
        else:
            outgoing = model.fc3.weight.norm(dim=0)
        props[layer] = {
            "incoming_weight_norm": incoming.cpu().numpy(),
            "outgoing_weight_norm": outgoing.cpu().numpy(),
            "activation_frequency": (active_count[layer] / n).cpu().numpy(),
            "mean_activation": (act_sum[layer] / n).cpu().numpy(),
            "mean_abs_local_logit_contribution": (local_sum[layer] / n).cpu().numpy(),
        }
    baseline = {
        "predictions": base_pred,
        "labels": labels_np,
        "top2_margin": margins["top2"],
        "true_label_margin": margins["true"],
        "accuracy": correct / total,
    }
    return props, baseline, margins


@torch.inference_mode()
def _measure_ablation(
    model: MNISTMLP,
    baseline_state: dict[str, torch.Tensor],
    dataset: Dataset,
    baseline: dict[str, np.ndarray | float],
    layer: str,
    neuron: int,
    seed: int,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    model.load_state_dict(baseline_state)
    module = getattr(model, layer)
    module.weight[neuron].zero_()
    module.bias[neuron].zero_()
    base_pred = torch.as_tensor(baseline["predictions"], device=device)
    base_top = torch.as_tensor(baseline["top2_margin"], device=device)
    base_true = torch.as_tensor(baseline["true_label_margin"], device=device)
    flips = correct = seen = 0
    top_margin_sum = true_margin_sum = 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    offset = 0
    try:
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            stop = offset + y.numel()
            logits = model(x)
            pred = logits.argmax(1)
            ref_pred = base_pred[offset:stop]
            flips += (pred != ref_pred).sum().item()
            correct += (pred == y).sum().item()
            top_margin_sum += _top2_margin(logits).sum().item()
            true_margin_sum += _true_label_margin(logits, y).sum().item()
            seen += y.numel()
            offset = stop
        base_acc = float(baseline["accuracy"])
        ablated_acc = correct / seen
        base_top_mean = float(base_top.mean().item())
        base_true_mean = float(base_true.mean().item())
        ablated_top_mean = top_margin_sum / seen
        ablated_true_mean = true_margin_sum / seen
        return {
            "seed": seed, "layer": layer, "neuron": neuron,
            "prediction_flips": flips,
            "prediction_flip_rate": flips / seen,
            "prediction_flip_pct": 100 * flips / seen,
            "baseline_accuracy": base_acc,
            "ablated_accuracy": ablated_acc,
            "accuracy_change_pp": 100 * (ablated_acc - base_acc),
            "mean_baseline_top2_margin": base_top_mean,
            "mean_ablated_top2_margin": ablated_top_mean,
            "mean_top2_margin_change": ablated_top_mean - base_top_mean,
            "mean_baseline_true_label_margin": base_true_mean,
            "mean_ablated_true_label_margin": ablated_true_mean,
            "mean_true_label_margin_change": ablated_true_mean - base_true_mean,
        }
    finally:
        # Restoration is unconditional, even if an evaluation batch raises.
        model.load_state_dict(baseline_state)


def analyze_trained_model(
    model: MNISTMLP,
    dataset: Dataset,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    baseline_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    props, baseline, _ = _baseline_features(model, dataset, device, batch_size)
    baseline_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rows = []
    for layer in ("fc1", "fc2"):
        for neuron in range(getattr(model, layer).out_features):
            row = {key: values[neuron].item() if isinstance(values[neuron], np.generic) else float(values[neuron])
                   for key, values in props[layer].items()}
            row.update(_measure_ablation(model, baseline_state, dataset, baseline, layer, neuron, seed, device))
            rows.append(row)
        print(f"seed {seed}: completed ablations in {layer}", flush=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in baseline_state.items())
    return pd.DataFrame(rows), baseline, baseline_state


def run_phase2(
    train_data: Dataset,
    test_data: Dataset,
    output_dir: str | Path = "results/phase2",
    device: str | torch.device = "auto",
    seeds: tuple[int, ...] = PHASE2_SEEDS,
    epochs: int = 5,
) -> pd.DataFrame:
    """Train and analyze models sequentially, freeing each before the next."""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device) if isinstance(device, str) else device
    config = {**CONFIG, "seeds": list(seeds), "epochs": epochs, "device": str(device),
              "train_examples": len(train_data), "test_examples": len(test_data),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "python_version": platform.python_version()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_rows, all_history, baseline_rows = [], [], []
    for seed in seeds:
        seed_everything(seed)
        tmp_dir = output_dir / f".training_seed_{seed}"
        model, accuracy, history = train_model(train_data, test_data, tmp_dir, device,
                                               seed=seed, epochs=epochs)
        print(f"seed {seed}: training complete; baseline test accuracy={accuracy:.4%}", flush=True)
        history.insert(0, "seed", seed)
        all_history.append(history)
        rows, baseline, state = analyze_trained_model(model, test_data, seed, device)
        checkpoint_path = checkpoint_dir / f"model_seed_{seed}.pt"
        saved_state = {k: v.detach().cpu() for k, v in state.items()}
        torch.save({"model_state": saved_state, "seed": seed, "epochs": epochs,
                    "baseline_accuracy": accuracy, "config": config}, checkpoint_path)
        np.save(output_dir / f"baseline_predictions_seed_{seed}.npy", baseline["predictions"])
        all_rows.append(rows)
        baseline_rows.append({"seed": seed, "baseline_accuracy": accuracy,
                              "baseline_correct": int(round(accuracy * len(test_data))),
                              "train_loss_final": float(history.iloc[-1].train_loss),
                              "checkpoint": str(checkpoint_path)})
        shutil.rmtree(tmp_dir, ignore_errors=True)
        del rows, baseline, state, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    measurements = pd.concat(all_rows, ignore_index=True)
    measurements.to_csv(output_dir / "neuron_properties_and_ablation.csv", index=False)
    pd.concat(all_history, ignore_index=True).to_csv(output_dir / "training_metrics.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_metrics.csv", index=False)
    return measurements
