from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import MNISTMLP, seed_everything


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_model(
    train_data: Dataset,
    test_data: Dataset,
    out_dir: str | Path,
    device: torch.device,
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> tuple[MNISTMLP, float, pd.DataFrame]:
    seed_everything(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)
    model = MNISTMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.numel()
            seen += y.numel()
        accuracy = evaluate(model, test_loader, device)
        rows.append({"epoch": epoch, "train_loss": sum_loss / seen, "test_accuracy": accuracy})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "training_metrics.csv", index=False)
    baseline_accuracy = float(metrics.iloc[-1]["test_accuracy"])
    torch.save({"model_state": model.cpu().state_dict(), "seed": seed, "epochs": epochs,
                "baseline_test_accuracy": baseline_accuracy}, out_dir / "baseline.pt")
    model.to(device)
    return model, baseline_accuracy, metrics
