import pandas as pd

from quantdash.validation import purged_date_folds


def test_folds_are_calendar_based_and_embargoed():
    dates = pd.Series(list(pd.date_range("2020-01-01", periods=500, freq="B")) * 3)
    folds = purged_date_folds(dates, n_splits=3, embargo_days=5, min_train_dates=252)
    assert len(folds) == 3
    normalized = pd.to_datetime(dates)
    for train, valid in folds:
        assert normalized.iloc[train].max() < normalized.iloc[valid].min()
        assert normalized.iloc[train].max() + pd.offsets.BDay(5) < normalized.iloc[valid].min()
        assert len(set(normalized.iloc[valid])) < len(valid)  # all tickers share the same dates
