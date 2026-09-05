import numpy as np
import pandas as pd

from quantdash import backtest
from quantdash.config import Mandate
from quantdash.model import ModelDiagnostics
from quantdash.portfolio import PortfolioResult


def _market_data(rows: int = 820) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2022-01-03", periods=rows, freq="B")
    data = {}
    for number, ticker in enumerate(["SPY", "AAPL", "GLD"]):
        close = pd.Series(100 + number * 5 + np.linspace(0, 30 + number, rows), index=dates)
        data[ticker] = pd.DataFrame({"Close": close, "Adj Close": close, "Volume": 1_000_000})
    return data


def test_backtest_rejects_insufficient_history():
    result = backtest.run_walk_forward(_market_data(200), Mandate())
    assert result.periods.empty
    assert "five years" in result.warnings[0]


def test_backtest_completes_periods_and_applies_costs(monkeypatch):
    data = _market_data()
    panel = pd.DataFrame({"date": data["SPY"].index, "ticker": "SPY", "close": data["SPY"]["Close"].to_numpy()})
    predictions = pd.DataFrame({"Ticker": ["AAPL"]})
    ranked = pd.DataFrame({"Ticker": ["AAPL", "GLD"]})
    holdings = pd.DataFrame({"Ticker": ["AAPL", "GLD"], "Weight": [0.60, 0.40]})

    monkeypatch.setattr(backtest, "build_features", lambda *args, **kwargs: panel)
    monkeypatch.setattr(backtest, "fit_predict", lambda *args, **kwargs: (predictions, ModelDiagnostics(trained=True)))
    monkeypatch.setattr(backtest, "rank_signals", lambda *args, **kwargs: ranked)
    monkeypatch.setattr(
        backtest,
        "optimize_portfolio",
        lambda *args, **kwargs: PortfolioResult(holdings=holdings, returns=pd.Series(dtype=float)),
    )

    result = backtest.run_walk_forward(data, Mandate(transaction_cost_bps=10), max_rebalances=3)
    assert len(result.periods) == 3
    assert (result.periods["Net Return"] <= result.periods["Gross Return"]).all()
    assert result.metrics["Rebalances"] == 3
    assert not result.equity.empty
