# Experiment report

## Run status

Completed on the manually downloaded MNIST archive, extracted into `data/MNIST/raw/`. The run used CPU PyTorch 2.9.0, seed 42, five epochs, all 60,000 training images and all 10,000 test images. The baseline test accuracy was 97.16%. The sweep wrote 2,496 full-test measurements. Matplotlib and pytest are unavailable in this environment, so plots were not generated here; six synthetic checks and a synthetic end-to-end smoke run passed earlier.

## Method

The project trains a ReLU MLP with dimensions 784–128–64–10 using Adam and cross-entropy. The standard MNIST train and test partitions are used. Each perturbation starts from the same saved baseline state. Gaussian noise is scaled so the perturbation L2 norm is the requested fraction of the original incoming weight row's L2 norm. Weight scaling multiplies that row by `1 + strength`. Ablation zeros the row and its bias, removing the unit's effective contribution. Biases are preserved for noise and scaling. Each measurement records accuracy, prediction flips, change in the baseline-predicted class probability, perturbation magnitude, and original row norm. Gaussian noise is repeated with fixed per-run seeds.

## Results

Training loss declined from 0.4170 in epoch 1 to 0.0725 in epoch 5. Test accuracy increased from 93.88% to 97.16% over the same epochs.

Mean prediction-flip rates averaged over neurons (and, for noise, three fixed perturbation seeds) were:

| Intervention | Strength | First hidden layer (`fc1`) | Second hidden layer (`fc2`) |
|---|---:|---:|---:|
| Gaussian noise | 0.1 | 0.0160% | 0.0123% |
| Gaussian noise | 0.5 | 0.0823% | 0.0793% |
| Gaussian noise | 1.0 | 0.1828% | 0.1593% |
| Weight scaling | 0.1 | 0.0235% | 0.0378% |
| Weight scaling | 0.5 | 0.1164% | 0.2063% |
| Weight scaling | 1.0 | 0.2321% | 0.4145% |
| Ablation | — | 0.2929% | 0.4661% |

Average flip rates rose with strength at all three tested levels for both noise and scaling. Mean accuracy moved only a little because flips include predictions that changed both from correct to incorrect and from incorrect to correct. For example, full-strength scaling of `fc2` changed 0.4145% of predictions on average while reducing mean accuracy by only 0.0358 percentage points.

Second-layer ablations changed more predictions on average (0.466% vs. 0.293%), while first-layer ablations caused the larger mean accuracy loss (0.058 vs. 0.047 percentage points). The most sensitive individual was in the first layer: `fc1` neuron 104 changed 342 of 10,000 predictions (3.42%) and reduced accuracy from 97.16% to 94.04% (312 fewer correct predictions). The most sensitive `fc2` ablation changed 0.90% of predictions. Seven of 192 neuron ablations changed no test predictions.

Weight norm had a moderate positive rank association with ablation flip rate (`fc1` Spearman ρ = 0.56; `fc2` ρ = 0.32), so norm carried some signal in this one trained model but was not a sufficient measure of functional effect. Two `fc1` neurons with nearly identical norms illustrate the spread: neuron 104 had norm 2.987 and 3.42% flips; neuron 122 had norm 3.006 and 0.61% flips. This single-seed run does not establish a general predictive relationship.

Raw measurements are in `results/neuron_sensitivity.csv`, training history in `results/training_metrics.csv`, and aggregates in `results/summary.csv`.

## Limitations

Results come from one initialization/training seed and one CPU run. They depend on initialization, optimizer settings, and finite test data. The experiment measures individual incoming-vector interventions, not interactions among multiple neurons or retrained behavior. Weight norm is only one parameter-space statistic and does not account for activation frequency or downstream structure. The probability-change metric follows the class predicted by the unmodified model, which makes it comparable even when the perturbed prediction changes. Matplotlib was missing here, so the requested plots remain to be generated after installing project dependencies.

## Reproduce

```bash
python -m neuron_sensitivity.cli --seed 42 --epochs 5
```

To regenerate the plots after installing dependencies, run the CLI command above. These findings describe this recorded run; retrain with additional seeds before treating them as stable conclusions.
