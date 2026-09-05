from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from .config import DEFENSIVE, SECTORS, Mandate
from .risk import risk_summary


@dataclass
class PortfolioResult:
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _conservative_risk_input(values: pd.Series, fallback: float) -> pd.Series:
    """Fill missing risk data with the 75th-percentile observed risk, not a favorable value."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed = numeric.dropna()
    fill_value = float(observed.quantile(0.75)) if not observed.empty else fallback
    return numeric.fillna(fill_value)


def rank_signals(
    predictions: pd.DataFrame,
    latest_features: pd.DataFrame,
    mandate: Mandate,
    return_error: float | None = None,
) -> pd.DataFrame:
    latest = latest_features.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    frame = predictions.merge(latest, left_on="Ticker", right_on="ticker", how="left")
    volatility = _conservative_risk_input(frame["vol20"].abs(), 0.30)
    tail_loss = _conservative_risk_input(frame["cvar95"].abs(), 0.04)
    beta_distance = _conservative_risk_input((frame["beta60"] - 1).abs(), 0.25)
    frame["Risk Score"] = (0.45 * volatility.rank(pct=True) + 0.30 * tail_loss.rank(pct=True) + 0.25 * beta_distance.rank(pct=True)).clip(0, 1)
    error = float(np.clip(return_error if return_error is not None else 0.08, 0.03, 0.30))
    frame["Return Low"] = (frame["Expected Return"] - error).clip(-0.50, 0.50)
    frame["Return High"] = (frame["Expected Return"] + error).clip(-0.50, 0.50)
    frame["Chance Low"] = (frame["Probability Positive"] - np.sqrt(frame["Probability Positive"] * (1 - frame["Probability Positive"]) / 100)).clip(0, 1)
    frame["Chance High"] = (frame["Probability Positive"] + np.sqrt(frame["Probability Positive"] * (1 - frame["Probability Positive"]) / 100)).clip(0, 1)
    frame["Probability > 2%"] = norm.sf((0.02 - frame["Expected Return"]) / error)
    frame["Probability > 5%"] = norm.sf((0.05 - frame["Expected Return"]) / error)
    frame["Probability Target"] = norm.sf((mandate.target_return - frame["Expected Return"]) / error)
    frame["Probability Contribution"] = 0.45 * frame["Probability Positive"].rank(pct=True)
    frame["Return Contribution"] = 0.40 * frame["Expected Return"].rank(pct=True)
    frame["Risk Contribution"] = 0.15 * (1 - frame["Risk Score"])
    frame["Signal Score"] = frame[["Probability Contribution", "Return Contribution", "Risk Contribution"]].sum(axis=1)
    frame["High Conviction"] = (
        (frame["Probability Positive"] >= 0.55) & (frame["Expected Return"] > 0.005) & (frame["Risk Score"] <= 0.65) & (frame["Reliability"] >= 0.15)
    )
    frame["Sector"] = frame["Ticker"].map(SECTORS).fillna("Other")
    frame["Defensive"] = frame["Ticker"].isin(DEFENSIVE)
    frame["Eligible Long"] = (
        (frame["Probability Positive"] >= mandate.min_probability)
        & (frame["Expected Return"] > mandate.min_expected_return)
        & (frame["Feature Coverage"] >= 0.65)
        & ~frame["Defensive"]
        & frame["Ticker"].ne(mandate.benchmark)
    )
    return frame.sort_values("Signal Score", ascending=False).reset_index(drop=True)


def optimize_portfolio(signals: pd.DataFrame, returns: pd.DataFrame, mandate: Mandate) -> PortfolioResult:
    eligible = signals[(signals["Eligible Long"]) | (signals["Defensive"])].copy()
    eligible = eligible[eligible["Ticker"].isin(returns.columns) & eligible["Ticker"].ne(mandate.benchmark)]
    eligible = eligible.sort_values("Signal Score", ascending=False).head(max(mandate.max_holdings * 3, 12))
    if eligible.empty:
        return PortfolioResult(warnings=["No securities passed the portfolio eligibility rules."])
    # Correlation-aware greedy selection, distinct from the standalone stock ranking.
    selected: list[str] = []
    corr = returns[eligible["Ticker"]].corr()
    for ticker in eligible["Ticker"]:
        if not selected or float(corr.loc[ticker, selected].abs().mean()) < 0.80:
            selected.append(ticker)
        if len(selected) >= mandate.max_holdings:
            break
    minimum_positions = int(np.ceil(1 / mandate.max_weight - 1e-12))
    if len(selected) < min(max(4, minimum_positions), len(eligible)):
        selected = eligible["Ticker"].head(mandate.max_holdings).tolist()
    if len(selected) < minimum_positions:
        return PortfolioResult(
            warnings=[
                f"No feasible portfolio: a {mandate.max_weight:.0%} position limit requires at least "
                f"{minimum_positions} usable securities, but only {len(selected)} are available."
            ]
        )
    candidates = eligible.set_index("Ticker").loc[selected]
    sample = returns[selected].dropna()
    if len(sample) < 252:
        return PortfolioResult(warnings=["Portfolio requires at least 252 common return observations."])
    cov = LedoitWolf().fit(sample.to_numpy()).covariance_ * mandate.horizon
    expected = candidates["Expected Return"].to_numpy(float)
    betas = candidates["beta60"].fillna(1.0).to_numpy(float)
    n = len(selected)
    bounds = [(0.0, mandate.max_weight)] * n
    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: mandate.beta_cap - float(w @ betas)},
    ]
    sectors = candidates["Sector"].to_numpy()
    for sector in np.unique(sectors):
        mask = (sectors == sector).astype(float)
        constraints.append({"type": "ineq", "fun": lambda w, m=mask: mandate.sector_cap - float(w @ m)})

    def objective(w: np.ndarray) -> float:
        diversification_penalty = 0.10 * float(w @ w)
        return -(float(w @ expected) - mandate.risk_aversion * float(w @ cov @ w) - diversification_penalty)

    result = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        return PortfolioResult(warnings=[f"No feasible portfolio under the selected position, sector, and beta limits: {result.message}"])
    weights = np.clip(result.x, 0, mandate.max_weight)
    if weights.sum() <= 0:
        return PortfolioResult(warnings=["No feasible portfolio: the optimizer returned zero investable weight."])
    weights /= weights.sum()

    tolerance = 1e-5
    sector_weights = {sector: float(weights[sectors == sector].sum()) for sector in np.unique(sectors)}
    violations = []
    if float(weights.max()) > mandate.max_weight + tolerance:
        violations.append("position limit")
    if float(weights @ betas) > mandate.beta_cap + tolerance:
        violations.append("beta limit")
    if any(weight > mandate.sector_cap + tolerance for weight in sector_weights.values()):
        violations.append("sector limit")
    if violations:
        return PortfolioResult(warnings=[f"No feasible portfolio: the proposed weights violated the {', '.join(violations)}."])

    weight_series = pd.Series(weights, index=selected, name="Weight")
    portfolio_returns = sample.dot(weight_series)
    holdings = candidates.reset_index()[["Ticker", "Sector", "Expected Return", "Probability Positive", "Risk Score"]]
    holdings["Weight"] = holdings["Ticker"].map(weight_series)
    holdings["Role"] = np.where(holdings["Ticker"].isin(DEFENSIVE), "Diversifier", "Return candidate")
    benchmark = returns[mandate.benchmark] if mandate.benchmark in returns else None
    metrics = risk_summary(portfolio_returns, benchmark)
    metrics.update(
        {
            "Expected Horizon Return": float(weight_series.dot(candidates["Expected Return"])),
            "Weighted Probability": float(weight_series.dot(candidates["Probability Positive"])),
            "Largest Position": float(weight_series.max()),
            "Portfolio Beta Constraint": float(weight_series.to_numpy() @ betas),
        }
    )
    return PortfolioResult(holdings.sort_values("Weight", ascending=False), portfolio_returns, metrics, [])
