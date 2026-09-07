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


def test_latest_incomplete_provider_row_keeps_each_tickers_last_valid_risk_values():
    tickers = ["AAPL", "JPM", "XOM"]
    predictions = pd.DataFrame(
        {
            "Ticker": tickers,
            "Probability Positive": [0.57, 0.56, 0.55],
            "Expected Return": [0.006, 0.005, 0.004],
            "Reliability": [0.30] * 3,
            "Feature Coverage": [1.0] * 3,
            "As Of": pd.to_datetime(["2026-01-02"] * 3),
        }
    )
    rows = []
    for ticker, volatility, cvar, beta in zip(tickers, [0.15, 0.25, 0.40], [-0.02, -0.04, -0.08], [0.8, 1.0, 1.4], strict=True):
        rows.extend(
            [
                {"ticker": ticker, "date": pd.Timestamp("2026-01-01"), "vol20": volatility, "cvar95": cvar, "beta60": beta},
                {"ticker": ticker, "date": pd.Timestamp("2026-01-02"), "vol20": np.nan, "cvar95": np.nan, "beta60": np.nan},
            ]
        )
    ranked = rank_signals(predictions, pd.DataFrame(rows), Mandate())
    assert ranked["Risk Score"].nunique() == 3
    assert ranked.set_index("Ticker").loc["AAPL", "Risk Score"] < ranked.set_index("Ticker").loc["XOM", "Risk Score"]


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


def test_diversified_research_candidates_produce_a_constraint_safe_portfolio():
    rng = np.random.default_rng(10)
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "COST", "CAT", "GLD", "TLT", "SHY"]
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (500, len(tickers))), columns=tickers)
    signals = pd.DataFrame(
        {
            "Ticker": tickers,
            "Expected Return": np.linspace(0.006, 0.001, len(tickers)),
            "Probability Positive": [0.57] * len(tickers),
            "Risk Score": [0.52] * len(tickers),
            "Signal Score": np.linspace(0.9, 0.4, len(tickers)),
            "Eligible Long": [True] * 6 + [False] * 3,
            "Defensive": [False] * 6 + [True] * 3,
            "Sector": ["Technology", "Technology", "Financials", "Energy", "Consumer", "Industrials", "Defensive", "Defensive", "Defensive"],
            "beta60": [1.2, 1.15, 1.05, 1.1, 0.9, 1.0, 0.1, 0.0, 0.05],
        }
    )
    mandate = Mandate()
    result = optimize_portfolio(signals, returns, mandate)
    assert not result.holdings.empty
    assert np.isclose(result.holdings["Weight"].sum(), 1.0)
    assert result.holdings["Weight"].max() <= mandate.max_weight + 1e-5
    assert result.holdings.groupby("Sector")["Weight"].sum().max() <= mandate.sector_cap + 1e-5


def test_large_candidate_pool_keeps_low_beta_diversifiers_available():
    rng = np.random.default_rng(11)
    stocks = [f"S{i:02d}" for i in range(30)]
    tickers = stocks + ["GLD", "TLT"]
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (500, len(tickers))), columns=tickers)
    sectors = ["Technology", "Financials", "Energy", "Consumer", "Industrials", "Healthcare"] * 5
    signals = pd.DataFrame(
        {
            "Ticker": tickers,
            "Expected Return": [0.006] * 30 + [0.001, 0.001],
            "Probability Positive": [0.57] * len(tickers),
            "Risk Score": [0.60] * 30 + [0.10, 0.10],
            "Signal Score": [*np.linspace(1.0, 0.4, 30), 0.05, 0.04],
            "Eligible Long": [True] * 30 + [False, False],
            "Defensive": [False] * 30 + [True, True],
            "Sector": sectors + ["Defensive", "Defensive"],
            "beta60": [1.30] * 30 + [0.05, 0.05],
        }
    )
    mandate = Mandate(beta_cap=0.90)
    result = optimize_portfolio(signals, returns, mandate)
    assert not result.holdings.empty
    assert {"GLD", "TLT"}.issubset(set(result.holdings["Ticker"]))
    assert result.metrics["Portfolio Beta Constraint"] <= mandate.beta_cap + 1e-5
