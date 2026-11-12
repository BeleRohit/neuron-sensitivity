from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import measure
from .perturb import LAYERS


def run_experiments(model: torch.nn.Module, test_data, out_dir: str | Path,
                    device: torch.device, strengths=(0.1, 0.5, 1.0),
                    noise_repeats: int = 3, sample_limit: int | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    x, y = next(iter(loader))
    if sample_limit is not None:
        x, y = x[:sample_limit], y[:sample_limit]
    baseline_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    rows = []
    for layer in LAYERS:
        count = getattr(model, layer).weight.shape[0]
        for neuron in range(count):
            for strength in strengths:
                for seed in range(noise_repeats):
                    rows.append(measure(model, baseline_state, layer, neuron,
                                        "noise", strength, seed, x, y, device))
                rows.append(measure(model, baseline_state, layer, neuron,
                                    "scale", strength, 0, x, y, device))
            rows.append(measure(model, baseline_state, layer, neuron,
                                "ablation", 1.0, 0, x, y, device))
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "neuron_sensitivity.csv", index=False)
    return frame
