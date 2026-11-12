from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(results: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = (results.groupby(["layer", "method", "strength"], as_index=False)
           .agg(accuracy=("perturbed_accuracy", "mean"), flips=("prediction_flip_pct", "mean")))
    fig, ax = plt.subplots(figsize=(8, 5))
    for (layer, method), group in agg[agg.method != "ablation"].groupby(["layer", "method"]):
        ax.plot(group.strength, group.accuracy, marker="o", label=f"{layer} {method}")
    ax.set(xlabel="Relative perturbation strength", ylabel="Mean test accuracy", title="Accuracy vs perturbation strength")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_strength.png", dpi=220); plt.close(fig)

    sample = results[results.method == "ablation"].pivot(index="layer", columns="neuron", values="prediction_flip_pct")
    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(sample, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(sample.index)), sample.index); ax.set(xlabel="Neuron index", title="Ablation prediction flips (%)")
    fig.colorbar(im, ax=ax, label="Test predictions changed (%)")
    fig.tight_layout(); fig.savefig(out_dir / "sensitivity_heatmap.png", dpi=220); plt.close(fig)

    agg_neuron = (results[results.method == "ablation"].groupby(["layer", "neuron"], as_index=False)
                  .agg(weight_norm=("neuron_weight_norm", "first"), sensitivity=("prediction_flip_pct", "first")))
    fig, ax = plt.subplots(figsize=(7, 5))
    for layer, group in agg_neuron.groupby("layer"):
        ax.scatter(group.weight_norm, group.sensitivity, s=18, alpha=.7, label=layer)
    ax.set(xlabel="Incoming weight L2 norm", ylabel="Ablation prediction flips (%)", title="Weight norm vs functional change")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out_dir / "weight_norm_vs_sensitivity.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for layer, group in results[results.method == "ablation"].groupby("layer"):
        ax.hist(group.prediction_flip_pct, bins=20, alpha=.55, label=layer)
    ax.set(xlabel="Prediction flips (%) after ablation", ylabel="Neuron count", title="Layer-wise ablation sensitivity")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "layerwise_sensitivity.png", dpi=220); plt.close(fig)
