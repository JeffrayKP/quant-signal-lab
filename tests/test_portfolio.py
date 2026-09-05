import numpy as np
import pandas as pd

from quantdash.config import Mandate
from quantdash.portfolio import optimize_portfolio, rank_signals


def test_optimizer_weights_sum_and_respect_maximum():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=400, freq="B")
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "GLD", "TLT"]
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (400, 6)), index=dates, columns=tickers)
    signals = pd.DataFrame(
        {
            "Ticker": tickers,
            "Expected Return": [0.02, 0.018, 0.014, 0.012, 0.006, 0.004],
            "Probability Positive": [0.6] * 6,
            "Risk Score": [0.4] * 6,
            "Signal Score": np.linspace(1, 0.5, 6),
            "Eligible Long": [True] * 4 + [False, False],
            "Defensive": [False] * 4 + [True, True],
            "Sector": ["Technology", "Technology", "Financials", "Energy", "Defensive", "Defensive"],
            "beta60": [1.1, 1.0, 1.0, 1.1, 0.1, -0.1],
        }
    )
    result = optimize_portfolio(signals, returns, Mandate(max_holdings=6, max_weight=0.30, sector_cap=0.50))
    assert not result.holdings.empty
    assert np.isclose(result.holdings["Weight"].sum(), 1)
    assert result.holdings["Weight"].max() <= 0.3001


def test_signal_score_equals_visible_contributions():
    predictions = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT"],
            "Probability Positive": [0.60, 0.55],
            "Expected Return": [0.02, 0.01],
            "Reliability": [0.30, 0.30],
            "Feature Coverage": [1.0, 1.0],
            "As Of": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        }
    )
    features = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "vol20": [0.20, 0.25],
            "cvar95": [-0.03, -0.04],
            "beta60": [1.0, 1.1],
        }
    )
    ranked = rank_signals(predictions, features, Mandate())
    components = ranked[["Probability Contribution", "Return Contribution", "Risk Contribution"]].sum(axis=1)
    assert np.allclose(ranked["Signal Score"], components)


def test_missing_risk_input_receives_conservative_score_instead_of_nan():
    predictions = pd.DataFrame(
        {
            "Ticker": ["CAT", "AAPL"],
            "Probability Positive": [0.57, 0.55],
            "Expected Return": [0.004, 0.003],
            "Reliability": [0.30, 0.30],
            "Feature Coverage": [1.0, 1.0],
            "As Of": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        }
    )
    features = pd.DataFrame(
        {
            "ticker": ["CAT", "AAPL"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "vol20": [np.nan, 0.20],
            "cvar95": [np.nan, -0.03],
            "beta60": [np.nan, 1.0],
        }
    )
    ranked = rank_signals(predictions, features, Mandate())
    assert ranked.loc[ranked["Ticker"].eq("CAT"), "Risk Score"].notna().all()


def test_infeasible_constraints_never_publish_violating_fallback_weights():
    rng = np.random.default_rng(7)
    tickers = ["A", "B", "C", "D", "E"]
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (400, 5)), columns=tickers)
    signals = pd.DataFrame(
        {
            "Ticker": tickers,
            "Expected Return": [0.02] * 5,
            "Probability Positive": [0.60] * 5,
            "Risk Score": [0.40] * 5,
            "Signal Score": np.linspace(1.0, 0.6, 5),
            "Eligible Long": [True] * 5,
            "Defensive": [False] * 5,
            "Sector": ["Technology"] * 5,
            "beta60": [1.50] * 5,
        }
    )
    result = optimize_portfolio(signals, returns, Mandate(max_holdings=5, max_weight=0.20, sector_cap=0.40, beta_cap=1.0))
    assert result.holdings.empty
    assert any("No feasible portfolio" in warning for warning in result.warnings)
