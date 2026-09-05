from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import ModelDiagnostics
from .portfolio import PortfolioResult
from .risk import historical_var, max_drawdown


@dataclass(frozen=True)
class EvidenceGate:
    label: str
    level: str
    detail: str
    probability_pass: bool
    return_pass: bool


@dataclass(frozen=True)
class MarketRegime:
    label: str
    level: str
    detail: str
    price: float
    ma50: float
    ma200: float
    volatility20: float
    drawdown60: float


def evidence_gate(diagnostics: ModelDiagnostics) -> EvidenceGate:
    probability_pass = bool(diagnostics.brier is not None and diagnostics.baseline_brier is not None and diagnostics.brier < diagnostics.baseline_brier)
    return_pass = bool(diagnostics.mae is not None and diagnostics.baseline_mae is not None and diagnostics.mae < diagnostics.baseline_mae)
    if probability_pass and return_pass:
        return EvidenceGate("Validated edge", "positive", "Both probability and return forecasts beat their naive out-of-fold baselines.", True, True)
    if probability_pass or return_pass:
        passed = "probability calibration" if probability_pass else "return error"
        failed = "return error" if probability_pass else "probability calibration"
        return EvidenceGate(
            "Mixed evidence",
            "warning",
            f"The model beats the {passed} baseline but not the {failed} baseline. Treat rankings as exploratory.",
            probability_pass,
            return_pass,
        )
    return EvidenceGate(
        "No demonstrated edge",
        "negative",
        "Neither forecast component beats its naive out-of-fold baseline. Do not treat the ranking as an actionable signal.",
        False,
        False,
    )


def market_regime(benchmark_prices: pd.Series) -> MarketRegime:
    prices = pd.Series(benchmark_prices).dropna().astype(float)
    if len(prices) < 200:
        return MarketRegime(
            "Insufficient history",
            "warning",
            "At least 200 benchmark observations are required for regime classification.",
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    price = float(prices.iloc[-1])
    ma50 = float(prices.tail(50).mean())
    ma200 = float(prices.tail(200).mean())
    volatility20 = float(prices.pct_change(fill_method=None).tail(20).std() * np.sqrt(252))
    drawdown60 = float(price / prices.tail(60).max() - 1)
    if price > ma50 > ma200 and volatility20 < 0.25:
        label, level = "Risk-on", "positive"
        detail = "SPY is above its 50-day and 200-day averages with contained recent volatility."
    elif price < ma200 or volatility20 > 0.35:
        label, level = "Risk-off", "negative"
        detail = "SPY is below its 200-day average or recent volatility is elevated."
    else:
        label, level = "Neutral / transitional", "warning"
        detail = "Trend and volatility signals are not aligned, so market context is mixed."
    return MarketRegime(label, level, detail, price, ma50, ma200, volatility20, drawdown60)


def _annualized_return(values: pd.Series) -> float:
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    wealth = float((1 + values).prod())
    return wealth ** (252 / len(values)) - 1 if wealth > 0 else np.nan


def strategy_comparison(
    signals: pd.DataFrame,
    portfolio: PortfolioResult,
    returns: pd.DataFrame,
    benchmark: str = "SPY",
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    investable = signals[(signals["Eligible Long"]) & signals["Ticker"].isin(returns.columns) & signals["Ticker"].ne(benchmark)]
    if investable.empty:
        investable = signals[(~signals["Defensive"]) & signals["Ticker"].isin(returns.columns) & signals["Ticker"].ne(benchmark)]
    leaders = investable.sort_values("Signal Score", ascending=False)["Ticker"].head(top_n).tolist()
    series: dict[str, pd.Series] = {}
    if leaders:
        series[f"Best stock ({leaders[0]})"] = returns[leaders[0]]
        series[f"Equal-weight top {len(leaders)}"] = returns[leaders].mean(axis=1, skipna=False)
    if not portfolio.returns.empty:
        series["Optimized portfolio"] = portfolio.returns
    if benchmark in returns:
        series[benchmark] = returns[benchmark]
    if not series:
        return pd.DataFrame(), pd.DataFrame()
    history = pd.concat(series, axis=1).sort_index().dropna(how="all")
    rows = []
    for name in history:
        values = history[name].dropna()
        rows.append(
            {
                "Strategy": name,
                "Historical Annual Return": _annualized_return(values),
                "Annual Volatility": float(values.std() * np.sqrt(252)) if len(values) else np.nan,
                "Maximum Drawdown": max_drawdown(values),
                "Daily VaR 95%": historical_var(values),
                "Observations": len(values),
            }
        )
    common = history.dropna()
    cumulative = (1 + common).cumprod() - 1 if not common.empty else pd.DataFrame()
    return pd.DataFrame(rows), cumulative
