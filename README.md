# Quant Signal Lab

Quant Signal Lab is a leakage-aware equity research dashboard that separates standalone stock
ranking from diversified portfolio construction. The Decision Brief connects model evidence,
market regime, the highest-ranked current signal, portfolio context, historical risk, and recent
company events. It is an educational research system, not an investment adviser, brokerage, or
trade-execution platform.

The default model universe includes liquid U.S. listings plus TSM and SKHY. Users can add any
Yahoo Finance-supported stock symbol to a model run, including international exchange suffixes
such as `7203.T`, `VOW3.DE`, `000660.KS`, or `0700.HK`. A single public deployment cannot preload
every global listing without institutional symbol-master data and a licensed market-data feed.
New listings with limited price history remain visible for research but cannot qualify as
high-conviction signals until enough features are available.

## What this version adds

- A model-evidence gate that labels results as `Validated edge`, `Mixed evidence`, or
  `No demonstrated edge` by comparing both forecast components with naive out-of-fold baselines
- A current SPY regime card using the 50-day trend, 200-day trend, recent volatility, and drawdown
- A portfolio overlay on the Opportunity Map, with gold rings for holdings and a gold diamond for
  the highest-ranked signal
- A transparent score chart showing the exact probability, expected-return, and risk contributions
- A historical comparison of the best stock, equal-weight top stocks, optimized portfolio, and SPY
- A two-page downloadable investment-research PDF brief generated from the live dashboard results
- Stock-specific current events with source, publication time, relevance, safe link, and a
  plain-language explanation of why each event may matter
- Clearer language when the model fails one or both validation baselines

Prediction markets were intentionally removed. They were unreliable for stock-level context and
did not strengthen the dashboard's investment-research story.

## Research and engineering controls

- Calendar-date, purged walk-forward validation across the full stock panel
- Forward targets created only after each historical cutoff
- Out-of-fold probability calibration and naive baseline comparisons
- Separate `Best Stocks` and `Best Portfolio` workflows
- Ledoit-Wolf covariance estimation and constrained mean-risk optimization
- Position, sector, and beta constraints
- Portfolio-matched historical testing with turnover and transaction costs
- VaR, CVaR, beta, volatility, drawdown, and scenario analysis
- Server-side secrets, bounded uploads, ticker validation, fixed news endpoints, XML hardening,
  request timeouts, retries, and generic upstream errors
- Automated tests, linting, dependency auditing, CodeQL, and security scanning

## Architecture

```text
app.py
  -> data.py          market-data validation
  -> features.py      point-in-time features and targets
  -> model.py         purged validation, calibration, forecasts
  -> portfolio.py     ranking and constrained optimization
  -> analytics.py     evidence gate, market regime, strategy comparison
  -> risk.py          portfolio risk and scenarios
  -> backtest.py      complete walk-forward pipeline
  -> integrations.py current-events research
  -> reporting.py     downloadable PDF research brief
  -> security.py      input and URL validation
```

## Run locally

Use Python 3.12. Open the extracted project folder in VS Code, then open a PowerShell terminal in
that folder.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
streamlit run app.py
```

On later launches, only activate the environment and run Streamlit:

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
streamlit run app.py
```

## Validate before deployment

```bash
ruff check .
pytest --cov=quantdash
bandit -r src -c pyproject.toml
pip-audit -r requirements.txt
```

## Secret management

Current events require no API key, and the public interface never asks for credentials. Never
place any real credentials in `.env`, Git, screenshots, or Streamlit widgets.

## Deployment checklist

1. Run all tests and security checks above.
2. Enable GitHub secret scanning, Dependabot, CodeQL, and protected branches.
3. Deploy behind HTTPS with restrictive access controls while testing.
4. Add hosting-level request limits, timeouts, and spending limits.
5. Configure uptime and error monitoring without logging portfolio inputs or secrets.
6. Review the model evidence gate and data age before describing the deployment as current.

The included Dockerfile runs as an unprivileged user and includes a health check.

## Important limitations

- Yahoo Finance is an unofficial research source and can be delayed or incomplete.
- RSS headlines can be incomplete, duplicated, delayed, or classified imperfectly.
- The default universe has survivorship and selection bias.
- Corporate actions, delistings, taxes, borrow fees, and market impact are not fully modeled.
- Probability calibration and feature relationships can change across market regimes.
- The portfolio comparison uses current selections over historical data, so it is descriptive and
  selection-biased. It is not an out-of-sample performance claim.
- Transaction costs are estimates, not executable quotes.
- No output is a personalized recommendation or guarantee.

## Resume description

Built a Python and Streamlit quantitative research platform that ranks equities using calibrated,
purged walk-forward machine-learning signals and constructs diversified portfolios through
constrained mean-risk optimization. Added validation-based model governance, market-regime
context, explainable score attribution, current-event relevance scoring, portfolio comparison,
automated PDF research briefs, VaR/CVaR stress analysis, secure server-side integrations, CI,
CodeQL, and dependency auditing.
