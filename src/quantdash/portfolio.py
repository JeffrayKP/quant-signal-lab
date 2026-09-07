from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
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
    latest_source = latest_features.sort_values(["ticker", "date"]).copy()
    risk_columns = [column for column in ["vol20", "cvar95", "beta60"] if column in latest_source]
    latest_source[risk_columns] = latest_source.groupby("ticker")[risk_columns].ffill()
    latest = latest_source.groupby("ticker", as_index=False).tail(1)
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
    eligible = eligible.sort_values("Signal Score", ascending=False)
    pool_size = max(mandate.max_holdings * 3, 12)
    defensive_pool = eligible[eligible["Defensive"]].head(min(2, pool_size))
    return_pool = eligible[~eligible["Defensive"]].head(pool_size - len(defensive_pool))
    eligible = pd.concat([return_pool, defensive_pool]).sort_values("Signal Score", ascending=False)
    if eligible.empty:
        return PortfolioResult(warnings=["No securities passed the portfolio eligibility rules."])
    # Preserve sector diversity and low-beta diversifiers before using
    # correlation to fill the remaining slots.
    selected: list[str] = []
    corr = returns[eligible["Ticker"]].corr()
    defensive = eligible[eligible["Defensive"]]["Ticker"].head(2).tolist()
    selected.extend(defensive)
    seen_sectors = set(eligible.set_index("Ticker").loc[selected, "Sector"]) if selected else set()
    for row in eligible.itertuples(index=False):
        ticker = str(row.Ticker)
        sector = str(row.Sector)
        if ticker not in selected and sector not in seen_sectors:
            selected.append(ticker)
            seen_sectors.add(sector)
        if len(selected) >= mandate.max_holdings:
            break
    for ticker in eligible["Ticker"]:
        if ticker in selected:
            continue
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
    sector_masks = []
    for sector in np.unique(sectors):
        mask = (sectors == sector).astype(float)
        sector_masks.append(mask)
        constraints.append({"type": "ineq", "fun": lambda w, m=mask: mandate.sector_cap - float(w @ m)})

    linear_inequalities = np.vstack([betas, *sector_masks])
    linear_limits = np.array([mandate.beta_cap, *([mandate.sector_cap] * len(sector_masks))])
    feasible = linprog(
        -expected,
        A_ub=linear_inequalities,
        b_ub=linear_limits,
        A_eq=np.ones((1, n)),
        b_eq=np.ones(1),
        bounds=bounds,
        method="highs",
    )
    if not feasible.success:
        return PortfolioResult(warnings=["No feasible portfolio exists under the selected position, sector, and beta limits."])

    def objective(w: np.ndarray) -> float:
        diversification_penalty = 0.10 * float(w @ w)
        return -(float(w @ expected) - mandate.risk_aversion * float(w @ cov @ w) - diversification_penalty)

    result = minimize(objective, feasible.x, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    warnings: list[str] = []
    if not result.success:
        weights = feasible.x
        warnings.append("The risk optimizer did not converge, so the displayed allocation uses a constraint-verified linear solution.")
    else:
        weights = result.x
    weights = np.clip(weights, 0, mandate.max_weight)
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
    holdings = holdings[holdings["Weight"] > tolerance]
    return PortfolioResult(holdings.sort_values("Weight", ascending=False), portfolio_returns, metrics, warnings)
