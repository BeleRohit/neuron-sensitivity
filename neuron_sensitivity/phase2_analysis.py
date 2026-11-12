"""Grouped-by-model correlations, predictive validation, and summaries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "incoming_weight_norm",
    "outgoing_weight_norm",
    "activation_frequency",
    "mean_activation",
    "mean_abs_local_logit_contribution",
]
TARGET = "prediction_flip_rate"


def iter_model_group_folds(frame: pd.DataFrame, group_column: str = "seed"):
    """Yield leave-one-trained-model-out folds; never split neurons randomly."""
    groups = sorted(frame[group_column].unique())
    for held_out in groups:
        train = frame[frame[group_column] != held_out]
        test = frame[frame[group_column] == held_out]
        assert set(train[group_column]).isdisjoint(set(test[group_column]))
        yield held_out, train.copy(), test.copy()


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray,
                   test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Standardized ridge regression with scaling fit on training models only."""
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xtr = (train_x - mean) / scale
    xte = (test_x - mean) / scale
    ymean = train_y.mean()
    yc = train_y - ymean
    design = np.column_stack([np.ones(len(xtr)), xtr])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ yc)
    prediction = ymean + np.column_stack([np.ones(len(xte)), xte]) @ beta
    # A flip rate is bounded; constrain the linear model to the valid range.
    return np.clip(prediction, 0.0, 1.0)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def analyze_phase2(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    correlations = []
    for (seed, layer), group in frame.groupby(["seed", "layer"]):
        for feature in FEATURES:
            correlations.append({"seed": int(seed), "layer": layer, "feature": feature,
                                 "spearman_rho": _spearman(group[feature], group[TARGET]),
                                 "n_neurons": len(group)})
    correlation_frame = pd.DataFrame(correlations)
    correlation_summary = (correlation_frame.groupby(["layer", "feature"], as_index=False)
                           .agg(mean_rho=("spearman_rho", "mean"), sd_across_models=("spearman_rho", "std"),
                                min_rho=("spearman_rho", "min"), max_rho=("spearman_rho", "max"),
                                models=("seed", "nunique")))
    correlation_frame.to_csv(out / "feature_correlations_by_model.csv", index=False)
    correlation_summary.to_csv(out / "feature_correlation_summary.csv", index=False)

    # Sensitivity distribution uncertainty is across the five trained networks.
    by_model = (frame.groupby(["seed", "layer"], as_index=False)
                .agg(mean_flip_rate=(TARGET, "mean"), median_flip_rate=(TARGET, "median"),
                     p90_flip_rate=(TARGET, lambda s: s.quantile(.9)),
                     max_flip_rate=(TARGET, "max"), zero_flip_neurons=(TARGET, lambda s: int((s == 0).sum())),
                     mean_accuracy_change_pp=("accuracy_change_pp", "mean")))
    distribution_summary = (by_model.groupby("layer", as_index=False)
                            .agg(mean_flip_rate=("mean_flip_rate", "mean"),
                                 sd_mean_flip_rate=("mean_flip_rate", "std"),
                                 mean_median_flip_rate=("median_flip_rate", "mean"),
                                 sd_median_flip_rate=("median_flip_rate", "std"),
                                 mean_p90_flip_rate=("p90_flip_rate", "mean"),
                                 mean_max_flip_rate=("max_flip_rate", "mean"),
                                 mean_zero_flip_neurons=("zero_flip_neurons", "mean"),
                                 models=("seed", "nunique")))
    by_model.to_csv(out / "sensitivity_distribution_by_model.csv", index=False)
    distribution_summary.to_csv(out / "sensitivity_distribution_summary.csv", index=False)

    predictive_rows, predictions = [], []
    feature_sets = {"incoming_norm_only": ["incoming_weight_norm"],
                    "all_pre_ablation_properties": FEATURES}
    for layer in sorted(frame.layer.unique()):
        layer_data = frame[frame.layer == layer].copy()
        for held_out, train, test in iter_model_group_folds(layer_data):
            for model_name, features in feature_sets.items():
                predicted = _ridge_predict(train[features].to_numpy(float), train[TARGET].to_numpy(float),
                                          test[features].to_numpy(float), alpha=1.0)
                observed = test[TARGET].to_numpy(float)
                rho = _spearman(pd.Series(predicted), pd.Series(observed))
                mae = float(np.mean(np.abs(predicted - observed)))
                denom = float(np.sum((observed - observed.mean()) ** 2))
                r2 = float(1 - np.sum((observed - predicted) ** 2) / denom) if denom > 0 else float("nan")
                predictive_rows.append({"layer": layer, "held_out_seed": int(held_out),
                                        "model": model_name, "n_train_models": train.seed.nunique(),
                                        "n_test_neurons": len(test), "spearman_rho": rho,
                                        "mae_flip_rate": mae, "r2": r2})
                predictions.append(pd.DataFrame({"seed": test.seed.values, "layer": layer,
                                                 "neuron": test.neuron.values, "model": model_name,
                                                 "observed_flip_rate": observed,
                                                 "predicted_flip_rate": predicted}))
    predictive = pd.DataFrame(predictive_rows)
    predictive_summary = (predictive.groupby(["layer", "model"], as_index=False)
                          .agg(mean_heldout_spearman=("spearman_rho", "mean"),
                               sd_heldout_spearman=("spearman_rho", "std"),
                               mean_mae=("mae_flip_rate", "mean"), sd_mae=("mae_flip_rate", "std"),
                               mean_r2=("r2", "mean"), sd_r2=("r2", "std"), heldout_models=("held_out_seed", "nunique")))
    prediction_frame = pd.concat(predictions, ignore_index=True)
    predictive.to_csv(out / "heldout_model_predictive_performance.csv", index=False)
    predictive_summary.to_csv(out / "heldout_model_predictive_summary.csv", index=False)
    prediction_frame.to_csv(out / "heldout_model_predictions.csv", index=False)

    # Ordinal indices are not neuron identities across independent initializations.
    index_columns = ["seed", "layer", "neuron", "incoming_weight_norm", "prediction_flip_rate"]
    if "prediction_flips" in frame.columns:
        index_columns.append("prediction_flips")
    indexed = frame[frame.neuron.isin([104, 122])][index_columns].copy()
    indexed.to_csv(out / "phase1_index_comparison.csv", index=False)

    # Matched-norm diagnostic: within each trained model/layer, pair neurons
    # whose incoming norms differ by <=1%, then compare their observed effects.
    matched = []
    for (seed, layer), group in frame.groupby(["seed", "layer"]):
        values = group[["neuron", "incoming_weight_norm", TARGET]].to_numpy(float)
        gaps, pairs = [], 0
        for i in range(len(values)):
            denom = max(values[i, 1], 1e-12)
            rel = np.abs(values[i+1:, 1] - values[i, 1]) / denom
            ok = rel <= .01
            if ok.any():
                pairs += int(ok.sum())
                gaps.extend(np.abs(values[i+1:, 2][ok] - values[i, 2]).tolist())
        matched.append({"seed": int(seed), "layer": layer, "pairs_within_1pct_norm": pairs,
                        "median_abs_flip_rate_gap": float(np.median(gaps)) if gaps else np.nan,
                        "p90_abs_flip_rate_gap": float(np.quantile(gaps, .9)) if gaps else np.nan,
                        "pairs_gap_over_0_5pp": sum(g >= .005 for g in gaps)})
    matched_frame = pd.DataFrame(matched)
    matched_frame.to_csv(out / "matched_norm_sensitivity_spread.csv", index=False)

    return {"correlations": correlation_frame, "correlation_summary": correlation_summary,
            "by_model": by_model, "distribution_summary": distribution_summary,
            "predictive": predictive, "predictive_summary": predictive_summary,
            "predictions": prediction_frame, "phase1_indices": indexed, "matched_norm": matched_frame}
