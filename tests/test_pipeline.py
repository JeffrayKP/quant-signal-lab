import numpy as np
import pandas as pd

from quantdash.analytics import evidence_gate, market_regime, strategy_comparison
from quantdash.config import Mandate
from quantdash.data import close_matrix, return_matrix
from quantdash.features import build_features
from quantdash.model import fit_predict
from quantdash.portfolio import optimize_portfolio, rank_signals
from quantdash.reporting import build_research_brief
from quantdash.risk import stress_tests


def _synthetic_market() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20260904)
    dates = pd.date_range("2023-01-02", periods=700, freq="B")
    market = rng.normal(0.0003, 0.009, len(dates))
    data = {}
    for number, ticker in enumerate(["SPY", "AAPL", "MSFT", "JPM", "GLD"]):
        daily_return = market * (0.45 + 0.1 * number) + rng.normal(0.0001, 0.007, len(dates))
        close = 100 * np.exp(np.cumsum(daily_return))
        data[ticker] = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.004,
                "Low": close * 0.996,
                "Close": close,
                "Adj Close": close,
                "Volume": rng.integers(500_000, 5_000_000, len(dates)),
            },
            index=dates,
        )
    return data


def test_full_research_pipeline_produces_finite_outputs_and_pdf():
    data = _synthetic_market()
    mandate = Mandate(max_holdings=5, max_weight=0.35, sector_cap=0.70, beta_cap=1.40)
    panel = build_features(data, mandate.benchmark)
    predictions, diagnostics = fit_predict(panel, mandate.horizon)
    assert diagnostics.trained
    assert not predictions.empty

    signals = rank_signals(predictions, panel, mandate, diagnostics.return_error)
    assert len(signals) == len(data)
    assert np.isfinite(signals["Risk Score"]).all()
    assert signals["Risk Score"].between(0, 1).all()
    assert signals["Feature Coverage"].between(0, 1).all()

    returns = return_matrix(data)
    portfolio = optimize_portfolio(signals, returns, mandate)
    assert not portfolio.holdings.empty
    assert np.isclose(portfolio.holdings["Weight"].sum(), 1.0)

    comparison, _ = strategy_comparison(signals, portfolio, returns, mandate.benchmark)
    leader = signals[(~signals["Defensive"]) & signals["Ticker"].ne(mandate.benchmark)].iloc[0]
    gate = evidence_gate(diagnostics)
    regime = market_regime(close_matrix(data)[mandate.benchmark])
    pdf = build_research_brief(
        leader,
        {"label": gate.label, "detail": gate.detail},
        {"label": regime.label, "detail": regime.detail},
        portfolio.holdings,
        comparison,
        signals.head(5),
        mandate.horizon,
        mandate.target_return,
    )
    assert pdf.startswith(b"%PDF")

    scenarios = stress_tests(
        portfolio.holdings.set_index("Ticker")["Weight"],
        signals.set_index("Ticker")["beta60"],
        returns,
    )
    assert len(scenarios) == 5
    assert np.isfinite(scenarios["Estimated Return"]).all()
