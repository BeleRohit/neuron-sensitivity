import torch

from neuron_sensitivity.model import MNISTMLP, seed_everything
from neuron_sensitivity.perturb import perturb_neuron


def test_noise_has_requested_relative_norm_and_is_repeatable():
    seed_everything(3)
    model = MNISTMLP()
    row = model.fc1.weight[0].clone()
    norm = row.norm()
    delta = perturb_neuron(model, "fc1", 0, "noise", .25, seed=19)
    assert abs(delta - .25 * norm.item()) < 1e-5
    first = model.fc1.weight[0].clone()
    with torch.no_grad():
        model.fc1.weight[0].copy_(row)
    perturb_neuron(model, "fc1", 0, "noise", .25, seed=19)
    assert torch.equal(first, model.fc1.weight[0])


def test_scale_and_restore_from_checkpoint_state():
    model = MNISTMLP()
    baseline = {k: v.clone() for k, v in model.state_dict().items()}
    original = model.fc2.weight[2].clone()
    perturb_neuron(model, "fc2", 2, "scale", .5)
    assert torch.allclose(model.fc2.weight[2], original * 1.5)
    model.load_state_dict(baseline)
    assert torch.equal(model.fc2.weight[2], original)


def test_ablation_zeros_row_and_bias():
    model = MNISTMLP()
    perturb_neuron(model, "fc2", 4, "ablation", 1.0)
    assert torch.count_nonzero(model.fc2.weight[4]) == 0
    assert model.fc2.bias[4] == 0


def test_model_initialization_is_reproducible():
    seed_everything(123)
    first = MNISTMLP().state_dict()
    seed_everything(123)
    second = MNISTMLP().state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)
