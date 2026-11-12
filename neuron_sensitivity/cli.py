from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Subset

from .data import load_mnist
from .experiment import run_experiments
from .model import choose_device
from .plots import make_plots
from .train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and analyze an MNIST MLP's neuron sensitivity.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional small test subset for a quick run.")
    args = parser.parse_args()
    out = Path(args.output_dir)
    device = choose_device(args.device)
    print(f"Device: {device}")
    train_data, test_data = load_mnist(args.data_dir)
    model, accuracy, training = train_model(train_data, test_data, out, device, args.seed, args.epochs)
    print(f"Baseline test accuracy: {accuracy:.4%}")
    results = run_experiments(model, test_data, out, device, args.strengths,
                              args.noise_repeats, args.sample_limit)
    make_plots(results, out / "plots")
    print(f"Saved {len(results)} measurements to {out / 'neuron_sensitivity.csv'}")
    print("Ablation mean prediction flips by layer:")
    print(results[results.method == "ablation"].groupby("layer").prediction_flip_pct.mean().to_string())
    summary = results.groupby(["layer", "method", "strength"], as_index=False).agg(
        mean_accuracy=("perturbed_accuracy", "mean"), mean_flip_pct=("prediction_flip_pct", "mean"))
    summary.to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
