import numpy as np
import pandas as pd

from quantdash.analytics import evidence_gate, market_regime, strategy_comparison
from quantdash.model import ModelDiagnostics
from quantdash.portfolio import PortfolioResult


def test_evidence_gate_requires_both_baselines_for_validated_edge():
    diagnostics = ModelDiagnostics(brier=0.20, baseline_brier=0.25, mae=0.04, baseline_mae=0.05)
    assert evidence_gate(diagnostics).label == "Validated edge"
    diagnostics.mae = 0.06
    assert evidence_gate(diagnostics).label == "Mixed evidence"


def test_market_regime_detects_orderly_uptrend():
    prices = pd.Series(np.linspace(100, 150, 260))
    result = market_regime(prices)
    assert result.label == "Risk-on"
    assert result.ma50 > result.ma200


def test_strategy_comparison_contains_requested_baskets():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (300, 4)), index=dates, columns=["AAPL", "MSFT", "NVDA", "SPY"])
    signals = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "NVDA", "SPY"],
            "Eligible Long": [True, True, True, False],
            "Defensive": [False, False, False, False],
            "Signal Score": [0.9, 0.8, 0.7, 0.4],
        }
    )
    portfolio = PortfolioResult(returns=returns[["AAPL", "MSFT"]].mean(axis=1))
    summary, cumulative = strategy_comparison(signals, portfolio, returns)
    names = set(summary["Strategy"])
    assert "Best stock (AAPL)" in names
    assert "Optimized portfolio" in names
    assert "SPY" in names
    assert not cumulative.empty


def test_strategy_fallback_never_labels_benchmark_as_best_stock():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    returns = pd.DataFrame({"AAPL": 0.001, "SPY": 0.0005}, index=dates)
    signals = pd.DataFrame(
        {
            "Ticker": ["SPY", "AAPL"],
            "Eligible Long": [False, False],
            "Defensive": [False, False],
            "Signal Score": [0.9, 0.8],
        }
    )
    summary, _ = strategy_comparison(signals, PortfolioResult(), returns)
    assert "Best stock (AAPL)" in set(summary["Strategy"])
