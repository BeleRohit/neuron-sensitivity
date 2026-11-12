import numpy as np
import torch
from torch.utils.data import TensorDataset

from neuron_sensitivity.model import MNISTMLP, seed_everything
from neuron_sensitivity.phase2_analysis import analyze_phase2, iter_model_group_folds
from neuron_sensitivity.phase2_experiment import (
    _baseline_features,
    _measure_ablation,
    analyze_trained_model,
)


def _small_model_data(seed=9):
    seed_everything(seed)
    model = MNISTMLP().eval()
    x = torch.rand(8, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    return model, TensorDataset(x, y), x, y


def test_phase2_feature_norms_and_activations_match_manual_calculation():
    model, dataset, x, _ = _small_model_data()
    props, _, _ = _baseline_features(model, dataset, torch.device("cpu"), batch_size=3)
    with torch.inference_mode():
        a1 = model.fc1(x.reshape(len(x), 784))
        h1 = torch.relu(a1)
        a2 = model.fc2(h1)
        h2 = torch.relu(a2)
        local_jacobian = torch.einsum("ck,bk,kj->bcj", model.fc3.weight,
                                      (a2 > 0).float(), model.fc2.weight)
        local1 = (h1.unsqueeze(1) * local_jacobian).abs().mean((0, 1))
        local2 = (h2.unsqueeze(1) * model.fc3.weight.unsqueeze(0)).abs().mean((0, 1))
    assert np.allclose(props["fc1"]["incoming_weight_norm"], model.fc1.weight.norm(dim=1).detach().numpy())
    assert np.allclose(props["fc1"]["outgoing_weight_norm"], model.fc2.weight.norm(dim=0).detach().numpy())
    assert np.allclose(props["fc2"]["outgoing_weight_norm"], model.fc3.weight.norm(dim=0).detach().numpy())
    assert np.allclose(props["fc1"]["activation_frequency"], (h1 > 0).float().mean(0).numpy())
    assert np.allclose(props["fc2"]["mean_activation"], h2.mean(0).numpy(), atol=1e-7)
    assert np.allclose(props["fc1"]["mean_abs_local_logit_contribution"], local1.numpy(), atol=1e-7)
    assert np.allclose(props["fc2"]["mean_abs_local_logit_contribution"], local2.numpy(), atol=1e-7)


def test_ablation_metrics_restore_exact_state_in_both_layers():
    model, dataset, _, _ = _small_model_data()
    props, baseline, _ = _baseline_features(model, dataset, torch.device("cpu"))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    x, y = dataset.tensors
    with torch.inference_mode():
        baseline_logits = model(x)
        baseline_pred = baseline_logits.argmax(1)
        baseline_top2 = baseline_logits.topk(2, dim=1).values
        baseline_true = baseline_logits.gather(1, y[:, None]).squeeze(1)
        mask = torch.nn.functional.one_hot(y, 10).bool()
        baseline_true_margin = baseline_true - baseline_logits.masked_fill(mask, -torch.inf).max(1).values
    for layer, neuron in (("fc1", 3), ("fc2", 4)):
        model.load_state_dict(state)
        module = getattr(model, layer)
        with torch.no_grad():
            module.weight[neuron].zero_()
            module.bias[neuron].zero_()
        with torch.inference_mode():
            manual_logits = model(x)
            manual_pred = manual_logits.argmax(1)
            manual_flips = int((manual_pred != baseline_pred).sum())
            manual_top2 = manual_logits.topk(2, dim=1).values
            manual_true = manual_logits.gather(1, y[:, None]).squeeze(1)
            manual_true_margin = manual_true - manual_logits.masked_fill(mask, -torch.inf).max(1).values
            manual_top_change = (manual_top2[:, 0] - manual_top2[:, 1]).mean().item() - (baseline_top2[:, 0] - baseline_top2[:, 1]).mean().item()
            manual_true_change = manual_true_margin.mean().item() - baseline_true_margin.mean().item()
        result = _measure_ablation(model, state, dataset, baseline, layer, neuron, 9, torch.device("cpu"))
        assert result["layer"] == layer and result["neuron"] == neuron
        assert result["prediction_flips"] == manual_flips
        assert np.isclose(result["prediction_flip_rate"], manual_flips / len(dataset))
        assert np.isclose(result["mean_top2_margin_change"], manual_top_change, atol=1e-7)
        assert np.isclose(result["mean_true_label_margin_change"], manual_true_change, atol=1e-7)
        assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())


def test_full_small_analysis_is_deterministic_and_restores_baseline():
    model, dataset, _, _ = _small_model_data(17)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    first, baseline1, saved = analyze_trained_model(model, dataset, 17, torch.device("cpu"), batch_size=4)
    second, baseline2, _ = analyze_trained_model(model, dataset, 17, torch.device("cpu"), batch_size=4)
    assert len(first) == 192
    assert first.equals(second)
    assert np.array_equal(baseline1["predictions"], baseline2["predictions"])
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
    assert all(torch.equal(saved[key], value) for key, value in state.items())


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    model, _, x, _ = _small_model_data(23)
    with torch.inference_mode():
        before = model(x)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "seed": 23}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = MNISTMLP()
    restored.load_state_dict(payload["model_state"])
    with torch.inference_mode():
        after = restored(x)
    assert torch.equal(before, after)


def test_predictive_folds_hold_out_whole_models():
    frame = __import__("pandas").DataFrame({"seed": np.repeat([42, 43, 44], 4),
                                             "neuron": list(range(4)) * 3})
    folds = list(iter_model_group_folds(frame))
    assert len(folds) == 3
    for held_out, train, test in folds:
        assert set(test.seed) == {held_out}
        assert held_out not in set(train.seed)
        assert set(train.seed).isdisjoint(set(test.seed))


def test_phase2_analysis_writes_leave_one_model_out_predictions(tmp_path):
    import pandas as pd
    rows = []
    for seed in (42, 43, 44, 45, 46):
        for layer in ("fc1", "fc2"):
            for neuron in range(6):
                v = neuron + seed / 100
                rows.append({"seed": seed, "layer": layer, "neuron": neuron,
                             "incoming_weight_norm": v, "outgoing_weight_norm": v / 3,
                             "activation_frequency": (neuron + 1) / 8,
                             "mean_activation": v / 10,
                             "mean_abs_local_logit_contribution": v / 5,
                             "prediction_flip_rate": (neuron + 1) / 1000,
                             "prediction_flip_pct": (neuron + 1) / 10,
                             "accuracy_change_pp": -v / 100})
    outputs = analyze_phase2(pd.DataFrame(rows), tmp_path)
    assert len(outputs["predictive"]) == 20
    assert len(outputs["predictions"]) == 5 * 2 * 6 * 2
    assert outputs["predictive"].n_train_models.eq(4).all()
    assert (tmp_path / "feature_correlation_summary.csv").exists()
    assert (tmp_path / "heldout_model_predictions.csv").exists()
