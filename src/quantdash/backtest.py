from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Mandate
from .features import build_features
from .model import fit_predict
from .portfolio import optimize_portfolio, rank_signals
from .risk import max_drawdown


@dataclass
class BacktestResult:
    periods: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def run_walk_forward(
    data: dict[str, pd.DataFrame],
    mandate: Mandate,
    min_train_days: int = 756,
    rebalance_every: int = 20,
    max_rebalances: int = 36,
) -> BacktestResult:
    """Backtest the same signal, eligibility, and optimizer pipeline shown live."""
    panel = build_features(data, mandate.benchmark)
    closes = pd.DataFrame({t: f["Adj Close"] for t, f in data.items()}).sort_index()
    returns = closes.pct_change(fill_method=None)
    common_dates = closes.dropna(subset=[mandate.benchmark]).index
    if len(common_dates) < min_train_days + mandate.horizon + 20:
        return BacktestResult(warnings=["Use at least five years of data for the walk-forward test."])
    dates = common_dates[min_train_days : -mandate.horizon : rebalance_every][-max_rebalances:]
    previous_weights = pd.Series(dtype=float)
    rows: list[dict] = []
    net_returns: list[float] = []
    benchmark_returns: list[float] = []
    period_dates: list[pd.Timestamp] = []
    warnings: list[str] = []
    for date in dates:
        historical_panel = panel[pd.to_datetime(panel["date"]) <= date]
        predictions, _ = fit_predict(historical_panel, mandate.horizon, as_of=date)
        if predictions.empty:
            continue
        ranked = rank_signals(predictions, historical_panel, mandate)
        historical_returns = returns.loc[:date]
        portfolio = optimize_portfolio(ranked, historical_returns, mandate)
        if portfolio.holdings.empty:
            continue
        weights = portfolio.holdings.set_index("Ticker")["Weight"]
        exit_position = min(closes.index.searchsorted(date) + mandate.horizon, len(closes) - 1)
        exit_date = closes.index[exit_position]
        start_prices = closes.loc[date, weights.index]
        end_prices = closes.loc[exit_date, weights.index]
        realized = (end_prices / start_prices - 1).dropna()
        aligned_weights = weights.reindex(realized.index)
        aligned_weights /= aligned_weights.sum()
        gross = float(aligned_weights.dot(realized))
        all_names = previous_weights.index.union(aligned_weights.index)
        turnover = float(aligned_weights.reindex(all_names, fill_value=0).sub(previous_weights.reindex(all_names, fill_value=0)).abs().sum() / 2)
        cost = turnover * mandate.transaction_cost_bps / 10_000
        net = gross - cost
        benchmark = float(closes.loc[exit_date, mandate.benchmark] / closes.loc[date, mandate.benchmark] - 1)
        rows.append(
            {
                "Rebalance": date,
                "Exit": exit_date,
                "Holdings": ", ".join(weights.index),
                "Gross Return": gross,
                "Net Return": net,
                "Benchmark Return": benchmark,
                "Turnover": turnover,
                "Cost": cost,
            }
        )
        net_returns.append(net)
        benchmark_returns.append(benchmark)
        period_dates.append(exit_date)
        previous_weights = aligned_weights
    if len(rows) < 12:
        warnings.append("Fewer than 12 successful rebalances; do not rely on annualized metrics.")
    if not rows:
        return BacktestResult(warnings=warnings or ["No backtest periods completed."])
    strategy = pd.Series(net_returns, index=period_dates)
    benchmark = pd.Series(benchmark_returns, index=period_dates)
    periods_per_year = 252 / rebalance_every
    strategy_wealth = (1 + strategy).cumprod()
    benchmark_wealth = (1 + benchmark).cumprod()
    annual_return = float(strategy_wealth.iloc[-1] ** (periods_per_year / len(strategy)) - 1)
    annual_vol = float(strategy.std(ddof=1) * np.sqrt(periods_per_year))
    excess = strategy - benchmark
    metrics = {
        "Rebalances": float(len(strategy)),
        "Cumulative Net Return": float(strategy_wealth.iloc[-1] - 1),
        "Benchmark Return": float(benchmark_wealth.iloc[-1] - 1),
        "Annualized Net Return": annual_return,
        "Annualized Volatility": annual_vol,
        "Sharpe (0% RF)": annual_return / annual_vol if annual_vol else np.nan,
        "Information Ratio": float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)) if excess.std(ddof=1) else np.nan,
        "Maximum Drawdown": max_drawdown(strategy),
        "Win Rate": float((strategy > 0).mean()),
        "Average Turnover": float(pd.DataFrame(rows)["Turnover"].mean()),
    }
    equity = pd.DataFrame({"Strategy": strategy_wealth - 1, mandate.benchmark: benchmark_wealth - 1})
    return BacktestResult(pd.DataFrame(rows), equity, metrics, warnings)
