# Neuron Sensitivity

A small, local PyTorch experiment on how changes to individual hidden neuron weight vectors affect an MNIST MLP. It is an educational investigation, not a claim of novel research.

## Setup

Python 3.10+ is recommended. On an Apple Silicon Mac, install a PyTorch build that supports MPS when available; this program also runs on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The first run downloads the four standard MNIST IDX files into `data/`. It also reads the raw uncompressed IDX files under `data/MNIST/raw/`, which is useful when you obtain the archive separately. Automatic download requires network access.

## Run

```bash
python -m neuron_sensitivity.cli
```

This trains a `784 → 128 → 64 → 10` ReLU MLP for five epochs, saves `results/baseline.pt` and training metrics, evaluates each hidden neuron under normalized Gaussian noise, relative scaling, and ablation, then writes CSV summaries and plots under `results/`. Use `--device cpu` or `--device mps` to choose a device, `--epochs N` to change training length, and `--sample-limit N` for a quick smoke run. Remove `--sample-limit` for the full 10,000-image test set.

Noise strength is the L2 norm of the added vector divided by the original row norm. Scaling strength `s` multiplies the row by `1+s`. Biases remain unchanged for noise and scaling. Ablation zeros both the incoming row and its bias, which makes that ReLU unit output zero. All experiments reload the same in-memory baseline state before measurement; the saved checkpoint is never edited.

Results include `training_metrics.csv`, `neuron_sensitivity.csv`, `summary.csv`, and plots (`sensitivity_heatmap.png`, `accuracy_vs_strength.png`, `weight_norm_vs_sensitivity.png`, `layerwise_sensitivity.png`). `EXPERIMENT_REPORT.md` records the measured findings and limitations. The notebook in `notebooks/01_experiment_walkthrough.ipynb` walks through the same methodology.

## Tests

```bash
pytest
```

Tests use small synthetic models/data and do not download MNIST.
