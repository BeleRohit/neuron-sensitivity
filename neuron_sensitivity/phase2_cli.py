from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import load_mnist
from .phase2_analysis import analyze_phase2
from .phase2_experiment import PHASE2_SEEDS, run_phase2
from .phase2_plots import make_phase2_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five-seed neuron sensitivity prediction study.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/phase2")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PHASE2_SEEDS))
    args = parser.parse_args()
    train_data, test_data = load_mnist(args.data_dir)
    measurements = run_phase2(train_data, test_data, args.output_dir, args.device,
                              tuple(args.seeds), args.epochs)
    analyses = analyze_phase2(measurements, args.output_dir)
    figures = make_phase2_plots(measurements, analyses, Path(args.output_dir) / "figures")
    print("Baseline metrics:")
    print(__import__("pandas").read_csv(Path(args.output_dir) / "baseline_metrics.csv").to_string(index=False))
    print("Held-out-model performance:")
    print(analyses["predictive_summary"].to_string(index=False))
    print("Wrote", len(measurements), "neuron records and", len(figures), "figures.")


if __name__ == "__main__":
    main()
