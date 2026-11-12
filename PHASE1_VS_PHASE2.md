# Phase 1 vs Phase 2

## What changed

**Phase 1** established the single-model intervention: train one MNIST MLP, perturb/ablate one hidden neuron at a time, and measure test accuracy, prediction flips, probability changes, and weight norms.

**Phase 2** keeps the same model, data, training recipe, and exact single-neuron ablation, then repeats the study across five independent model seeds. It also measures pre-ablation neuron properties and evaluates predictors on a held-out trained model, with all neurons from that model excluded from training.

| Aspect | Phase 1 | Phase 2 |
|---|---|---|
| Independent trained models | 1, seed 42 | 5, seeds 42–46 |
| Baseline test accuracy | 97.16% | 97.244% ± 0.105 percentage points across seeds |
| Ablation measurements | 192 neurons | 960 neuron/model pairs |
| Sensitivity target | Test prediction flips | Test prediction-flip rate |
| Neuron properties | Incoming norm | Incoming/outgoing norms, activation frequency, mean activation, local logit contribution |
| Predictive evaluation | None | Leave-one-model-out, 5 folds |
| Main result | Norms did not fully explain individual effects | Multi-property predictors generalized better than incoming norm alone |

## Findings carried forward

Phase 1’s second-layer ablations changed more predictions on average than first-layer ablations. That layer-level pattern held across all five Phase 2 seeds: mean flip rate was 0.465% for `fc2` and 0.293% for `fc1`. However, each model’s most sensitive individual neuron was in `fc1`, with the top unit changing 1.83–3.42% of test predictions. The average layer effect and the strongest single-unit effect answer different questions.

Phase 1’s neuron 104 was a large outlier in seed 42: 3.42% of predictions changed after ablation, compared with 0.61% for neuron 122. The same seed-42 result recurred. The numeric index-104 result did not recur as a stable identity across seeds: hidden-unit permutations mean “neuron 104” in another initialization is not the same learned feature. In Phase 2, index 104 ranked above index 122 in three of five models, with one unusually large index-104 effect in seed 42.

The broader Phase 1 observation—that similar incoming norms can hide very different functional effects—was repeated across networks. Within a model/layer, many neuron pairs had incoming norms within 1% yet had materially different flip rates. Across the five models there were 35 first-layer and 24 second-layer matched-norm pairs with flip-rate gaps above 0.5 percentage points.

Phase 2 adds a predictive conclusion: incoming norm alone was associated with sensitivity, but a five-feature ridge model improved held-out-model rank correlation from 0.588 to 0.835 in `fc1` and from 0.442 to 0.820 in `fc2`. Mean absolute local logit contribution was the strongest individual feature by average Spearman correlation. Activation features were much more informative for `fc2` than activation frequency was for `fc1`.

## Interpretation

Phase 1 showed that removing individual neurons can affect predictions unevenly and that incoming weight size is incomplete. Phase 2 shows that activation and downstream pathway measurements add useful predictive information across five independently trained networks. The evidence is encouraging but bounded: five seeds, one MLP architecture, and one fixed dataset do not justify a universal neuron-importance rule. Neuron indices are not identities across models, and the first-layer local logit contribution is an explicitly approximate linearization around the baseline network.

See [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) for Phase 1 results and [`PHASE2_REPORT.md`](PHASE2_REPORT.md) for Phase 2 methods, fold-level metrics, figures, and limitations.
