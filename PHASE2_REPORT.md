# Phase 2 report: predicting neuron sensitivity across models

## Executive findings

Five independently initialized MNIST MLPs reached a mean baseline test accuracy of **97.244% ± 0.105 percentage points** (mean ± sample SD across the five model seeds). Every one of the 192 hidden neurons in every model was ablated once on the same 10,000-image test set, producing 960 neuron records.

The strongest individual pre-ablation feature was mean absolute local logit contribution: its per-model Spearman correlation with ablation flip rate averaged **0.837 ± 0.029** for `fc1` and **0.820 ± 0.064** for `fc2`. A five-feature ridge model evaluated by leaving out an entire trained network outperformed an incoming-weight-norm-only model in both layers:

| Layer | Predictor | Held-out Spearman ρ, mean ± SD | MAE, percentage points | R², mean ± SD |
|---|---|---:|---:|---:|
| `fc1` | Incoming norm only | 0.588 ± 0.050 | 0.128 | 0.348 ± 0.059 |
| `fc1` | All five pre-ablation features | **0.835 ± 0.057** | **0.094** | **0.561 ± 0.111** |
| `fc2` | Incoming norm only | 0.442 ± 0.106 | 0.134 | 0.373 ± 0.128 |
| `fc2` | All five pre-ablation features | **0.820 ± 0.061** | **0.089** | **0.688 ± 0.064** |

The combined features improved rank ordering and error in all five held-out models in both layers. They predict sensitivity usefully in this setting, but error remains nonzero and five training seeds are a small replication base. The results support combining activation and downstream information with weight statistics; they do not establish a universal neuron-importance score.

## Experimental setup

- **Data:** Standard MNIST: all 60,000 training examples and all 10,000 test examples. The archive was downloaded separately and extracted as raw IDX files under `data/MNIST/raw/`.
- **Architecture:** ReLU MLP `784 → 128 → 64 → 10`.
- **Training:** Adam, cross-entropy, batch size 128, learning rate 0.001, five epochs, seeds 42–46. Models were trained and analyzed one at a time on CPU; checkpoints are saved separately.
- **Intervention:** For each neuron, the baseline state was restored, its incoming weight row and bias were set to zero, and all 10,000 test examples were evaluated. The complete baseline state was restored in a `finally` block before the next neuron.
- **Target:** `prediction_flip_rate`, the fraction of test examples whose predicted class changed after ablation. Accuracy and top-1/top-2 and true-label margin changes are also recorded.
- **Predictor validation:** Separate models were fitted for each hidden layer. Leave-one-model-out folds held out one complete seed and trained on the other four. Standardization is fit only on the four training models in each fold. Ridge regression uses fixed `alpha=1.0`; predictions are clipped to the valid flip-rate range `[0,1]`. No neuron-level random split or hyperparameter tuning was used.

The model seed, not each neuron, is the independent replication unit. Correlation and sensitivity-distribution uncertainty is summarized across five model seeds. Within-model neuron rows are used to fit and evaluate the held-out-model predictor, but are not counted as five hundred independent model replications.

## Neuron properties

The CSV distinguishes parameter-only properties from data-dependent properties:

| Property | Definition | Requires test data? | Uses ablation? |
|---|---|---:|---:|
| Incoming weight norm | L2 norm of the neuron’s incoming row | No | No |
| Outgoing weight norm | L2 norm of the corresponding column in the next linear layer | No | No |
| Activation frequency | Fraction of test examples with positive ReLU activation | Yes | No |
| Mean activation | Mean post-ReLU activation over all test examples, including zeros | Yes | No |
| Mean absolute local logit contribution | Mean absolute activation-times-local-logit-Jacobian, averaged over examples and 10 output logits | Yes | No |
| Prediction flips and margins | Baseline versus single-neuron-ablated outputs | Yes | **Yes** |

For `fc2`, the local logit contribution for neuron `j` is `h2[j] * W3[:, j]`; because the output layer is linear, this is its exact additive logit contribution at the baseline activations. For `fc1`, it is `h1[j] * Σk W3[:, k] · 1[a2[k] > 0] · W2[k, j]`. This includes the `fc2` ReLU gate and is a **first-order local approximation**, not the exact finite effect of ablating `fc1`; ablation can change downstream ReLU gates. The target itself always comes from an exact forward pass after ablation.

Margins are averaged over test examples. The top-two margin is the largest logit minus the second-largest logit. The true-label margin is the true-class logit minus the largest non-true-class logit. The per-neuron CSV stores each model’s baseline mean, ablated mean, and their change.

## Results

### Baseline models and sensitivity distributions

| Seed | Baseline test accuracy |
|---:|---:|
| 42 | 97.16% |
| 43 | 97.27% |
| 44 | 97.11% |
| 45 | 97.33% |
| 46 | 97.35% |

Mean ablation flip rates were **0.293% ± 0.011 percentage points** for `fc1` and **0.465% ± 0.023 percentage points** for `fc2` (mean ± SD across models). The second layer therefore changed more predictions on average, while each seed’s largest single-neuron outlier was in `fc1` (1.83–3.42% of test predictions). Mean accuracy changes were more variable than flip rates because a flip can repair a baseline error as well as create one.

### Feature correlations

Mean per-model Spearman correlations across the five networks:

| Feature | `fc1` ρ, mean ± SD | `fc2` ρ, mean ± SD |
|---|---:|---:|
| Incoming weight norm | 0.587 ± 0.050 | 0.442 ± 0.106 |
| Outgoing weight norm | 0.529 ± 0.078 | 0.258 ± 0.139 |
| Activation frequency | 0.075 ± 0.051 | 0.593 ± 0.089 |
| Mean activation | 0.368 ± 0.059 | 0.737 ± 0.077 |
| Mean absolute local logit contribution | **0.837 ± 0.029** | **0.820 ± 0.064** |

The relationship depended on layer. Activation frequency alone was weak for `fc1`, while it tracked sensitivity more strongly for `fc2`. Mean activation and the local logit contribution were more consistently associated in both layers. The local contribution was the strongest single feature by average rank correlation, but the combined predictor performed best in held-out-model evaluation.

### Held-out-model prediction

The combined predictor reduced mean absolute error from 0.128 to 0.094 percentage points for `fc1` (about 27%) and from 0.134 to 0.089 percentage points for `fc2` (about 34%). Its held-out Spearman correlation exceeded the norm-only result in every fold. Held-out R² was positive in all five folds for both predictors and layers. Full fold-by-fold values are in `heldout_model_predictive_performance.csv`; out-of-fold neuron-level predictions are in `heldout_model_predictions.csv`.

### Phase 1 neurons 104 and 122

Seed 42 reproduced the Phase 1 result: `fc1` neuron 104 changed 3.42% of predictions, versus 0.61% for neuron 122. Across the other four seeds, their corresponding numeric indices were not consistently ranked the same: neuron 104 was more sensitive in three of five seeds. Neuron numbers do not identify the same learned feature across independently initialized models, so this is only an ordinal-index check, not tracking of the same neuron identity.

The broader “similar norms, different effects” pattern did recur. Within each model and layer, there were on average 314 `fc1` neuron pairs and 109 `fc2` pairs whose incoming norms differed by no more than 1%. Across those matched-norm pairs, 35 `fc1` and 24 `fc2` pairs (summed over seeds) differed in flip rate by more than 0.5 percentage points. Incoming norm alone therefore misses consequential differences, even though it is positively correlated with sensitivity.

## Figures and artifacts

Nine publication-oriented vector figures and PNG previews are in `results/phase2/figures/`. Key files include the five-model heatmap, feature-correlation comparison, held-out predicted-versus-observed plot, layer-by-seed distributions, and selected-neuron margin changes. The SVGs are generated without Matplotlib; PNG previews are rasterized from those SVGs.

Checkpoints are in `results/phase2/checkpoints/`; baseline predictions are saved per seed as `.npy`. `config.json`, `training_metrics.csv`, `baseline_metrics.csv`, and `neuron_properties_and_ablation.csv` record the run inputs and outputs.

## Limitations

1. Five model seeds provide only a small sample for uncertainty and held-out generalization. The SDs describe these seeds, not a population-level guarantee.
2. All properties and target measurements use the same fixed MNIST test distribution. The model-level holdout tests transfer across trained networks, not across datasets or image distributions.
3. The network is a small MLP trained for five epochs; conclusions may differ for convolutional models, other datasets, or training regimes.
4. The local logit contribution for `fc1` is a first-order approximation around baseline activations. It should not be confused with an exact ablation attribution.
5. Linear ridge models do not capture nonlinear interactions. Feature correlations are descriptive, not causal claims.
6. Numeric neuron indices are permutation-dependent and are not aligned identities across seeds.

## Reproduction

```bash
python -m neuron_sensitivity.phase2_cli --data-dir data --output-dir results/phase2 --device cpu
```

This retrains all five seeds and regenerates checkpoints, measurements, grouped-model analyses, and figures. Use `--device auto`, `--device mps`, or `--device cuda` where supported. The saved `config.json` records the executed configuration.
