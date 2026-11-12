import torch

from neuron_sensitivity.metrics import measure
from neuron_sensitivity.model import MNISTMLP


def test_measure_reports_expected_fields_and_restores_baseline():
    model = MNISTMLP()
    baseline = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.rand(12, 1, 28, 28)
    y = torch.randint(0, 10, (12,))
    result = measure(model, baseline, "fc1", 0, "scale", .1, 0,
                     x, y, torch.device("cpu"))
    expected = {"layer", "neuron", "method", "strength", "seed", "baseline_accuracy",
                "perturbed_accuracy", "prediction_flips", "prediction_flip_pct",
                "mean_abs_predicted_class_probability_change", "perturbation_magnitude",
                "neuron_weight_norm"}
    assert expected == result.keys()
    assert result["layer"] == "fc1" and result["neuron"] == 0
    assert 0 <= result["baseline_accuracy"] <= 1
    assert 0 <= result["perturbed_accuracy"] <= 1
    assert 0 <= result["prediction_flip_pct"] <= 100
    assert result["perturbation_magnitude"] > 0
    assert all(torch.equal(model.state_dict()[key], value) for key, value in baseline.items())


def test_flip_count_and_accuracy_are_calculated_from_predictions():
    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 1)
            self.fc2 = torch.nn.Linear(1, 1)
            self.fc3 = torch.nn.Linear(1, 10)

        def forward(self, x):
            x = x.reshape(x.shape[0], 784)
            return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    model = TinyMLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc1.weight[0, 0] = 1
        model.fc2.weight[0, 0] = 1
        model.fc3.weight[0, 0] = 1
        model.fc3.bias[1] = .4
    baseline = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.zeros(4, 1, 28, 28)
    x[:, 0, 0, 0] = 1
    y = torch.zeros(4, dtype=torch.long)
    result = measure(model, baseline, "fc1", 0, "scale", -1.0, 0,
                     x, y, torch.device("cpu"))
    assert result["baseline_accuracy"] == 1.0
    assert result["perturbed_accuracy"] == 0.0
    assert result["prediction_flips"] == 4
    assert result["prediction_flip_pct"] == 100.0
