# Phase 2 technical summary

We trained five MNIST MLPs (784–128–64–10; Adam; cross-entropy; five epochs; seeds 42–46) and ablated each of 192 hidden neurons in turn on the full 10,000-example test set. Baseline accuracy averaged 97.244% (SD 0.105 percentage points across models). The primary target was the fraction of predictions changed by ablation.

Incoming weight norm alone predicted held-out-model sensitivity moderately (`fc1` Spearman ρ = 0.588; `fc2` = 0.442). A standardized five-feature ridge model using incoming/outgoing norms, activation frequency, mean activation, and mean absolute local logit contribution improved held-out Spearman to 0.835 and 0.820, respectively. The local contribution was the strongest individual correlate in both layers (mean ρ = 0.837 for `fc1`, 0.820 for `fc2`). These averages and SDs are across five leave-one-model-out folds.

Second-layer ablations changed more predictions on average (0.465%) than first-layer ablations (0.293%), but each model’s most sensitive single neuron was in the first layer. Similar-norm neurons with substantially different flip rates occurred in all five models. Phase 1’s index-104 outlier recurred in seed 42 but was not stable across independently initialized networks; neuron indices do not identify aligned features between models.

This is evidence that activation and downstream-path properties add predictive value over incoming norm for this architecture and dataset. It is not a universal neuron-importance rule: the experiment covers five seeds, one MLP, one training setup, and one fixed test distribution. The `fc1` local-logit feature is a first-order approximation through the downstream ReLU; the sensitivity target itself is measured with an exact ablation forward pass.
