from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

from .features import FEATURES, add_targets
from .validation import calibration_table, purged_date_folds


@dataclass
class ModelDiagnostics:
    trained: bool = False
    brier: float | None = None
    baseline_brier: float | None = None
    mae: float | None = None
    baseline_mae: float | None = None
    rank_ic: float | None = None
    samples: int = 0
    folds: int = 0
    return_error: float | None = None
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def _models() -> tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor]:
    return (
        HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=42),
        HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, learning_rate=0.04, l2_regularization=1.0, loss="absolute_error", random_state=42),
    )


def fit_predict(panel: pd.DataFrame, horizon: int, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, ModelDiagnostics]:
    diagnostics = ModelDiagnostics()
    if as_of is not None:
        # Truncate first. Creating targets before truncation would leak post-cutoff prices.
        panel = panel[pd.to_datetime(panel["date"]) <= pd.Timestamp(as_of)].copy()
    frame = add_targets(panel, horizon)
    trainable = frame.dropna(subset=["future_return", "target_positive"]).copy()
    feature_cols = [c for c in FEATURES if c in trainable]
    trainable = trainable[trainable[feature_cols].notna().sum(axis=1) >= len(feature_cols) // 2]
    latest_source = frame.sort_values(["ticker", "date"]).copy()
    # A provider can publish a latest price row before every field is complete.
    # Carry the most recent known feature forward within that ticker instead of
    # replacing the live signal with cross-sectional training medians.
    latest_source[feature_cols] = latest_source.groupby("ticker")[feature_cols].ffill()
    latest = latest_source.groupby("ticker", as_index=False).tail(1)
    if len(trainable) < 1000:
        diagnostics.warnings.append("Insufficient history for a defensible panel model.")
        return pd.DataFrame(), diagnostics

    raw_x = trainable[feature_cols]
    y_cls = trainable["target_positive"].astype(int).to_numpy()
    y_reg = trainable["future_return"].to_numpy()
    if np.unique(y_cls).size < 2:
        diagnostics.warnings.append("The available training history contains only one return class.")
        return pd.DataFrame(), diagnostics
    folds = purged_date_folds(trainable["date"], n_splits=4, embargo_days=horizon)
    if len(folds) < 3:
        diagnostics.warnings.append("Not enough unique dates for purged walk-forward validation.")
        return pd.DataFrame(), diagnostics

    oof_prob = np.full(len(trainable), np.nan)
    oof_return = np.full(len(trainable), np.nan)
    for train_idx, valid_idx in folds:
        # Fit imputation values on each training fold only. Using medians from
        # the full panel here would leak validation-period information.
        fold_medians = raw_x.iloc[train_idx].median().fillna(0)
        x_train = raw_x.iloc[train_idx].fillna(fold_medians)
        x_valid = raw_x.iloc[valid_idx].fillna(fold_medians)
        classifier, regressor = _models()
        classifier.fit(x_train, y_cls[train_idx])
        regressor.fit(x_train, y_reg[train_idx])
        oof_prob[valid_idx] = classifier.predict_proba(x_valid)[:, 1]
        oof_return[valid_idx] = regressor.predict(x_valid)

    mask = np.isfinite(oof_prob) & np.isfinite(oof_return)
    if mask.sum() < 200:
        diagnostics.warnings.append("Too few out-of-fold predictions.")
        return pd.DataFrame(), diagnostics
    if np.unique(y_cls[mask]).size < 2:
        diagnostics.warnings.append("Out-of-fold outcomes contain only one return class, so probabilities cannot be calibrated.")
        return pd.DataFrame(), diagnostics
    base_probability = float(np.mean(y_cls[~mask])) if (~mask).sum() else float(np.mean(y_cls))
    base_return = float(np.median(y_reg[~mask])) if (~mask).sum() else float(np.median(y_reg))
    diagnostics.brier = brier_score_loss(y_cls[mask], oof_prob[mask])
    diagnostics.baseline_brier = brier_score_loss(y_cls[mask], np.full(mask.sum(), base_probability))
    diagnostics.mae = mean_absolute_error(y_reg[mask], oof_return[mask])
    diagnostics.baseline_mae = mean_absolute_error(y_reg[mask], np.full(mask.sum(), base_return))
    diagnostics.rank_ic = float(pd.Series(oof_return[mask]).corr(pd.Series(y_reg[mask]), method="spearman"))
    diagnostics.samples = int(mask.sum())
    diagnostics.folds = len(folds)
    diagnostics.return_error = float(np.std(y_reg[mask] - oof_return[mask], ddof=1))

    # Platt-scale genuinely out-of-fold probabilities, then refit base models on all past data.
    calibrator = LogisticRegression(C=1.0, random_state=42)
    logits = np.log(np.clip(oof_prob[mask], 1e-6, 1 - 1e-6) / np.clip(1 - oof_prob[mask], 1e-6, 1))
    calibrator.fit(logits.reshape(-1, 1), y_cls[mask])
    diagnostics.calibration = calibration_table(calibrator.predict_proba(logits.reshape(-1, 1))[:, 1], y_cls[mask])
    medians = raw_x.median().fillna(0)
    x = raw_x.fillna(medians)
    classifier, regressor = _models()
    classifier.fit(x, y_cls)
    regressor.fit(x, y_reg)
    x_latest = latest[feature_cols].fillna(medians)
    feature_coverage = latest[feature_cols].notna().mean(axis=1).to_numpy(float)
    raw = classifier.predict_proba(x_latest)[:, 1]
    latest_logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    probability = calibrator.predict_proba(latest_logits.reshape(-1, 1))[:, 1]
    expected = regressor.predict(x_latest)

    # Shrink noisy forecasts toward zero based on genuine out-of-fold evidence.
    regression_skill = max(0.0, 1 - diagnostics.mae / max(diagnostics.baseline_mae, 1e-9))
    classification_skill = max(0.0, 1 - diagnostics.brier / max(diagnostics.baseline_brier, 1e-9))
    reliability = float(np.clip((regression_skill + classification_skill) / 2, 0.05, 0.75))
    diagnostics.trained = True
    if diagnostics.brier >= diagnostics.baseline_brier or diagnostics.mae >= diagnostics.baseline_mae:
        diagnostics.warnings.append("Model failed at least one naive baseline; treat rankings cautiously.")
    # Reliability is evidence about interpretation, not a multiplier on the
    # point forecast. Multiplying by a 5% floor erased useful cross-sectional
    # differences and made every return display as 0.0%.
    return pd.DataFrame(
        {
            "Ticker": latest["ticker"].to_numpy(),
            "Probability Positive": probability,
            "Expected Return": np.clip(expected, -0.25, 0.25),
            "Reliability": reliability * feature_coverage,
            "Feature Coverage": feature_coverage,
            "As Of": pd.to_datetime(latest["date"]).to_numpy(),
        }
    ), diagnostics
