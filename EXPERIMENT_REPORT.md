# Experiment report

## Run status

The implementation and synthetic perturbation checks are available. A full MNIST run has **not** been completed in the current environment: MNIST is not cached, outbound DNS/network access is unavailable, and the environment lacks Matplotlib and pytest. Consequently, this report contains no fabricated training or sensitivity measurements. Run the command below in a network-enabled Python environment to generate the measurements; replace this status with the resulting values before drawing empirical conclusions.

## Method

The project trains a ReLU MLP with dimensions 784–128–64–10 using Adam and cross-entropy. The standard MNIST train and test partitions are used. Each perturbation starts from the same saved baseline state. Gaussian noise is scaled so the perturbation L2 norm is the requested fraction of the original incoming weight row's L2 norm. Weight scaling multiplies that row by `1 + strength`. Ablation zeros the row and its bias, removing the unit's effective contribution. Biases are preserved for noise and scaling. Each measurement records accuracy, prediction flips, change in the baseline-predicted class probability, perturbation magnitude, and original row norm. Gaussian noise is repeated with fixed per-run seeds.

## Results

Not measured yet. The program writes results to `results/neuron_sensitivity.csv` and aggregate values to `results/summary.csv`. No claims about which layer is more sensitive, whether sensitivity rises monotonically, or whether weight norm predicts importance should be made until these outputs have been examined.

## Limitations

Results depend on initialization, training seed, optimizer settings, and finite test data. The experiment measures individual incoming-vector interventions, not interactions among multiple neurons or retrained behavior. Weight norm is only one parameter-space statistic and does not account for activation frequency or downstream structure. The probability-change metric follows the class predicted by the unmodified model, which makes it comparable even when the perturbed prediction changes.

## Reproduce

```bash
python -m neuron_sensitivity.cli --seed 42 --epochs 5
```

Record the command, software versions, device, baseline accuracy, and output summary alongside any added findings.
