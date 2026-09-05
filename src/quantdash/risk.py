from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    values = pd.Series(returns).dropna()
    return float(values.quantile(1 - confidence)) if len(values) else np.nan


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    values = pd.Series(returns).dropna()
    var = historical_var(values, confidence)
    tail = values[values <= var]
    return float(tail.mean()) if len(tail) else var


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + pd.Series(returns).dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min()) if len(wealth) else np.nan


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 60 or aligned.iloc[:, 1].var() == 0:
        return np.nan
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())


def risk_summary(returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    values = pd.Series(returns).dropna()
    annual_vol = float(values.std() * np.sqrt(252)) if len(values) else np.nan
    result = {
        "Annualized Volatility": annual_vol,
        "Daily VaR 95%": historical_var(values),
        "Daily CVaR 95%": historical_cvar(values),
        "Maximum Drawdown": max_drawdown(values),
    }
    if benchmark is not None:
        result["Beta"] = beta(values, benchmark)
    return result


def stress_tests(weights: pd.Series, asset_beta: pd.Series, daily_returns: pd.DataFrame) -> pd.DataFrame:
    portfolio_beta = float(weights.dot(asset_beta.reindex(weights.index).fillna(1.0)))
    portfolio = daily_returns.reindex(columns=weights.index).dropna().dot(weights)
    largest = float(weights.max())
    scenarios = {
        "Market decline 5%": -0.05 * portfolio_beta,
        "Market decline 10%": -0.10 * portfolio_beta,
        "Volatility shock": 2 * historical_cvar(portfolio),
        "Largest holding declines 15%": -0.15 * largest,
        "Top three holdings decline 10%": -0.10 * float(weights.nlargest(3).sum()),
    }
    return pd.DataFrame({"Scenario": scenarios.keys(), "Estimated Return": scenarios.values()})
