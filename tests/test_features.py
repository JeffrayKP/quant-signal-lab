import numpy as np
import pandas as pd

from quantdash.features import add_targets, build_features


def test_forward_target_is_per_ticker_and_horizon():
    dates = pd.date_range("2025-01-01", periods=4, freq="B")
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["A"] * 4 + ["B"] * 4,
            "close": [100, 101, 102, 104, 50, 49, 48, 47],
        }
    )
    result = add_targets(panel, 2)
    a = result[result["ticker"].eq("A")].reset_index(drop=True)
    assert np.isclose(a.loc[0, "future_return"], 0.02)
    assert a["future_return"].tail(2).isna().all()


def test_feature_build_handles_missing_volume_without_crashing():
    dates = pd.date_range("2024-01-01", periods=320, freq="B")
    close = pd.Series(np.linspace(100, 130, len(dates)), index=dates)
    bars = pd.DataFrame({"Close": close, "Adj Close": close})
    result = build_features({"SPY": bars})
    assert len(result) == len(dates)
    assert result["volume_z"].isna().all()
