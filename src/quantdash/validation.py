from __future__ import annotations

import numpy as np
import pandas as pd


def purged_date_folds(dates: pd.Series, n_splits: int = 4, embargo_days: int = 5, min_train_dates: int = 252) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate panel-safe folds using calendar dates, never ticker-major row positions."""
    normalized = pd.to_datetime(dates).dt.normalize()
    unique = np.array(sorted(normalized.dropna().unique()))
    if len(unique) < min_train_dates + n_splits * 20:
        return []
    validation_blocks = np.array_split(unique[min_train_dates:], n_splits)
    folds = []
    for block in validation_blocks:
        if len(block) == 0:
            continue
        valid_start = pd.Timestamp(block[0])
        train_cutoff = valid_start - pd.offsets.BDay(max(embargo_days, 1))
        # Strictly exclude the cutoff date so a training target that spans the
        # selected horizon cannot land on the first validation date.
        train_idx = np.flatnonzero(normalized.lt(train_cutoff).to_numpy())
        valid_idx = np.flatnonzero(normalized.isin(block).to_numpy())
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


def calibration_table(probability: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"probability": probability, "actual": actual}).dropna()
    frame["bucket"] = pd.cut(frame["probability"], np.linspace(0, 1, 11), include_lowest=True)
    return frame.groupby("bucket", observed=True).agg(predicted=("probability", "mean"), observed=("actual", "mean"), n=("actual", "size")).reset_index()
