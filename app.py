from __future__ import annotations

import html
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quantdash.analytics import evidence_gate, market_regime, strategy_comparison
from quantdash.backtest import run_walk_forward
from quantdash.config import DEFAULT_UNIVERSE, SECTORS, Mandate
from quantdash.data import close_matrix, data_health, download_prices, return_matrix
from quantdash.features import build_features
from quantdash.integrations import current_events
from quantdash.model import fit_predict
from quantdash.portfolio import optimize_portfolio, rank_signals
from quantdash.reporting import build_research_brief
from quantdash.risk import stress_tests
from quantdash.security import is_valid_ticker, normalize_ticker, validate_tickers, validate_upload_size

st.set_page_config(page_title="Quant Signal Workspace", page_icon="📈", layout="wide")

CSS = """
<style>
:root { --surface:#11161d; --line:#39434e; --text:#f1eee7; --muted:#afb4b9; --gold:#bd9650; }
.stApp { background:linear-gradient(180deg,#05070a 0%,#0a0e14 100%); color:var(--text); }
html, body, [class*="css"] { font-family:Garamond, Georgia, serif; }
[data-testid="stSidebar"] { background:#0a0d11; border-right:1px solid var(--line); }
.hero { border:1px solid var(--line); border-top:3px solid var(--gold); background:var(--surface); border-radius:7px; padding:28px 30px; margin:4px 0 18px; }
.eyebrow,.section-kicker { color:var(--gold); font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { color:var(--text); margin:.35rem 0 .7rem; font-size:2.65rem; line-height:1.02; }
.hero p { color:#b9c8d5; font-size:1rem; line-height:1.5; max-width:950px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.chip { background:#0d1a26; border:1px solid var(--line); color:#b8c9d7; border-radius:999px; padding:5px 9px; font-size:.73rem; }
.insight { border-left:3px solid var(--gold); background:rgba(189,150,80,.10); padding:13px 16px; color:#c8d6e2; margin:8px 0 18px; }
.metric-card { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px; min-height:98px; }
.metric-card span { color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-card strong { display:block; color:var(--text); font-size:1.45rem; margin-top:5px; }
.metric-card small { color:#8fa4b8; display:block; margin-top:4px; }
.quant-loader { display:flex; align-items:center; gap:11px; color:#b9c8d5; font-size:1rem; padding:18px 0; }
.quant-loader-circle { width:20px; height:20px; border:3px solid #39434e; border-top-color:#bd9650; border-radius:50%; animation:quant-spin .75s linear infinite; flex:0 0 auto; }
@keyframes quant-spin { to { transform:rotate(360deg); } }
/* Hide Streamlit's illustrated loading animation so this dashboard uses only the plain loader above. */
[data-testid="stSpinner"], [data-testid="stStatusWidget"] { display:none !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def pct(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def num(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def metrics(items: list[tuple[str, str, str]]) -> None:
    for column, (label, value, detail) in zip(st.columns(len(items)), items, strict=True):
        column.markdown(
            f'<div class="metric-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong><small>{html.escape(detail)}</small></div>',
            unsafe_allow_html=True,
        )


def show_loader(message: str):
    placeholder = st.empty()
    update_loader(placeholder, message)
    return placeholder


def update_loader(placeholder, message: str) -> None:
    placeholder.markdown(
        f'<div class="quant-loader"><span class="quant-loader-circle"></span><span>{html.escape(message)}</span></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800, show_spinner=False, max_entries=8)
def cached_download(tickers: tuple[str, ...], period: str):
    return download_prices(list(tickers), period)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=8)
def cached_features(data: dict[str, pd.DataFrame], benchmark: str) -> pd.DataFrame:
    return build_features(data, benchmark)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=16)
def cached_predictions(panel: pd.DataFrame, horizon: int):
    return fit_predict(panel, horizon)


@st.cache_data(ttl=900, show_spinner=False, max_entries=64)
def cached_current_events(ticker: str, sector: str):
    return current_events(ticker, sector)


def controls() -> tuple[list[str], str, Mandate, bool]:
    st.sidebar.markdown("### Quant research\n**Signal Workspace**")
    st.sidebar.caption("Set the mandate. The dashboard handles the detail.")
    st.sidebar.caption("Change any settings below, then click Apply settings once.")
    with st.sidebar.form("research_settings", enter_to_submit=False):
        period = st.selectbox("Price history", ["3y", "5y", "10y"], index=1)
        horizon = st.selectbox("Time horizon", [3, 5, 10, 20], index=1, format_func=lambda x: f"{x} trading days")
        target = st.select_slider("Return target", [0.00, 0.01, 0.02, 0.03, 0.05], value=0.03, format_func=lambda x: f"{x:.0%}")
        manual = st.text_area("Add stocks to this model run", placeholder="TSM, SKHY, 7203.T, VOW3.DE")
        st.caption("Use Yahoo Finance symbols. International stocks need their exchange suffix, such as 7203.T or VOW3.DE.")
        with st.expander("Screen filters", expanded=False):
            min_probability = st.slider("Minimum probability positive", 0.40, 0.70, 0.50, 0.01)
        with st.expander("Portfolio construction", expanded=False):
            max_holdings = st.slider("Maximum holdings", 5, 12, 8)
            max_weight = st.slider("Maximum position", 0.10, 0.35, 0.20, 0.01)
            sector_cap = st.slider("Maximum sector exposure", 0.25, 0.60, 0.40, 0.05)
            beta_cap = st.slider("Maximum portfolio beta", 0.60, 1.40, 1.05, 0.05)
            costs = st.slider("Transaction cost assumption (bps)", 0, 50, 10)
        with st.expander("Universe and data", expanded=False):
            upload = st.file_uploader("Upload ticker CSV", type=["csv"])
        st.form_submit_button("Apply settings", type="primary", width="stretch")
    # User-requested symbols come first so partial upstream responses and the
    # bounded recovery path prioritize the exact stocks the user asked for.
    requested = [x for x in manual.replace("\n", ",").split(",") if x.strip()]
    if upload is not None:
        try:
            validate_upload_size(upload.size)
            uploaded = pd.read_csv(io.BytesIO(upload.getvalue()), nrows=101)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
            st.sidebar.error("The ticker file must be a valid CSV smaller than 100 KB.")
            st.stop()
        if not uploaded.empty:
            requested.extend(uploaded.iloc[:100, 0].dropna().astype(str).tolist())
    values = requested + list(DEFAULT_UNIVERSE)
    mandate = Mandate(
        horizon=horizon,
        target_return=target,
        max_holdings=max_holdings,
        max_weight=max_weight,
        sector_cap=sector_cap,
        beta_cap=beta_cap,
        min_probability=min_probability,
        transaction_cost_bps=float(costs),
    )
    return validate_tickers(values), period, mandate, st.sidebar.button("Refresh market data", type="primary", width="stretch")


def opportunity_map(signals: pd.DataFrame, holdings: pd.DataFrame, leader_ticker: str) -> go.Figure:
    frame = signals[~signals["Defensive"]].copy()
    weight_map = holdings.set_index("Ticker")["Weight"] if not holdings.empty else pd.Series(dtype=float)
    frame["Portfolio Weight"] = frame["Ticker"].map(weight_map).fillna(0.0)
    fig = px.scatter(
        frame,
        x="Risk Score",
        y="Expected Return",
        size="Probability Positive",
        color="Sector",
        hover_name="Ticker",
        hover_data={"Probability Positive": ":.1%", "Expected Return": ":.2%", "Risk Score": ":.1%", "Portfolio Weight": ":.1%"},
        text="Ticker",
        title="Opportunity Map: expected return versus standalone risk",
    )
    fig.update_traces(textposition="top center", marker={"line": {"width": 1, "color": "#d9e2ec"}})
    portfolio_rows = frame[frame["Portfolio Weight"] > 0]
    if not portfolio_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_rows["Risk Score"],
                y=portfolio_rows["Expected Return"],
                mode="markers",
                name="Portfolio holding",
                text=portfolio_rows["Ticker"],
                customdata=portfolio_rows[["Portfolio Weight"]],
                hovertemplate="<b>%{text}</b><br>Portfolio weight: %{customdata[0]:.1%}<extra></extra>",
                marker={"size": 27, "symbol": "circle-open", "color": "#BD9650", "line": {"width": 3, "color": "#BD9650"}},
            )
        )
    leader_row = frame[frame["Ticker"].eq(leader_ticker)].head(1)
    if not leader_row.empty:
        fig.add_trace(
            go.Scatter(
                x=leader_row["Risk Score"],
                y=leader_row["Expected Return"],
                mode="markers",
                name="Top-ranked signal",
                text=leader_row["Ticker"],
                hovertemplate="<b>%{text}</b><br>Highest-ranked current signal<extra></extra>",
                marker={"size": 16, "symbol": "diamond", "color": "#F4C66A", "line": {"width": 1, "color": "#0A0E14"}},
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#70869e")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=540)
    fig.update_xaxes(tickformat=".0%", title="Standalone risk score: lower is better")
    fig.update_yaxes(tickformat=".2%", title="Expected return over selected horizon")
    return fig


def price_chart(closes: pd.DataFrame, ticker: str) -> go.Figure:
    series = closes[ticker].dropna()
    fig = px.line(
        pd.DataFrame({"Price": series, "20-day MA": series.rolling(20).mean(), "50-day MA": series.rolling(50).mean()}),
        title=f"{ticker}: price and moving-average context",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=420)
    return fig


def score_breakdown_chart(row: pd.Series) -> go.Figure:
    components = [
        ("Probability rank", float(row["Probability Contribution"]), "#BD9650"),
        ("Expected-return rank", float(row["Return Contribution"]), "#70869E"),
        ("Lower-risk credit", float(row["Risk Contribution"]), "#8AA88B"),
    ]
    fig = go.Figure()
    for name, value, color in components:
        fig.add_trace(go.Bar(x=[value], y=[row["Ticker"]], name=name, orientation="h", marker_color=color, text=[f"{value:.3f}"], textposition="inside"))
    fig.update_layout(
        barmode="stack",
        title=f"Signal score {row['Signal Score']:.3f}: transparent contribution breakdown",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        xaxis={"range": [0, 1], "title": "Weighted score contribution"},
    )
    return fig


def comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    plot = comparison.copy()
    plot["Drawdown Magnitude"] = plot["Maximum Drawdown"].abs().clip(lower=0.01)
    fig = px.scatter(
        plot,
        x="Annual Volatility",
        y="Historical Annual Return",
        text="Strategy",
        color="Strategy",
        size="Drawdown Magnitude",
        hover_data={"Maximum Drawdown": ":.1%", "Daily VaR 95%": ":.2%", "Observations": True, "Drawdown Magnitude": False},
        title="Current-selection historical risk and return comparison",
    )
    fig.update_traces(textposition="top center", marker={"line": {"width": 1, "color": "#D9E2EC"}})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=470, showlegend=False)
    fig.update_xaxes(tickformat=".0%", title="Annualized historical volatility")
    fig.update_yaxes(tickformat=".0%", title="Annualized historical return")
    return fig


def explain_signal(row: pd.Series) -> list[str]:
    output = []
    if row.get("Probability Positive", 0) >= 0.55:
        output.append(f"Probability-positive estimate is {pct(row['Probability Positive'], 0)}, above the high-conviction threshold.")
    if row.get("Expected Return", 0) > 0:
        output.append(f"Expected {pct(row['Expected Return'])} return is positive over the selected horizon.")
    if row.get("relative20", 0) > 0:
        output.append(f"Recent relative performance versus the benchmark is positive at {pct(row['relative20'])}.")
    if row.get("drawdown60", 0) < -0.05:
        output.append(f"The stock is {pct(row['drawdown60'])} below its 60-day high, creating a potential rebound setup.")
    return output[:4] or ["The model does not identify a strong, distinct driver beyond the combined signal."]


def show_evidence(gate) -> None:
    if gate.level == "positive":
        st.success(f"**Model evidence gate: {gate.label}.** {gate.detail}")
    elif gate.level == "warning":
        st.warning(f"**Model evidence gate: {gate.label}.** {gate.detail}")
    else:
        st.error(f"**Model evidence gate: {gate.label}.** {gate.detail}")


def render_events(events: list[dict]) -> None:
    if not events:
        st.info("No current events are available for this stock right now.")
        return
    for event in events:
        with st.container(border=True):
            left, right = st.columns([0.82, 0.18])
            with left:
                st.subheader(str(event["Event"]))
                st.caption(f"{event['Source']}  |  {event['Published']}  |  {event['Relevance']}")
                st.write(event["Why It Matters"])
            with right:
                st.metric("Relevance", f"{float(event['Relevance Score']):.0%}")
                st.link_button("Read source", str(event["Link"]), width="stretch")


def main() -> None:
    tickers, period, mandate, refresh = controls()
    if refresh:
        cached_download.clear()
        cached_features.clear()
        cached_predictions.clear()
    loader = show_loader("Loading market data...")
    try:
        data, data_warnings = cached_download(tuple(tickers), period)
        if mandate.benchmark not in data or len(data) < 5:
            st.error("Market data could not be loaded. Check your internet connection, then click Refresh market data in the sidebar.")
            for warning in data_warnings[:5]:
                st.caption(f"• {warning}")
            st.stop()
        update_loader(loader, "Calculating market indicators...")
        panel = cached_features(data, mandate.benchmark)
        update_loader(loader, f"Updating the {mandate.horizon}-day research model...")
        predictions, diagnostics = cached_predictions(panel, mandate.horizon)
    finally:
        loader.empty()
    if predictions.empty:
        st.error("The model did not have enough validated history to publish research signals.")
        st.stop()
    signals = rank_signals(predictions, panel, mandate, diagnostics.return_error)
    opportunities = signals[signals["Eligible Long"]].copy()
    returns, closes = return_matrix(data), close_matrix(data)
    fallback_stocks = signals[(~signals["Defensive"]) & signals["Ticker"].ne(mandate.benchmark)]
    if fallback_stocks.empty:
        st.error("No non-benchmark stock has enough validated data for the research view. Refresh the data or try again later.")
        st.stop()
    research_candidates = opportunities.copy()
    if research_candidates.empty:
        research_candidates = fallback_stocks[fallback_stocks["Feature Coverage"] >= 0.65].copy()
        if research_candidates.empty:
            research_candidates = fallback_stocks.copy()

    portfolio = optimize_portfolio(signals, returns, mandate)
    portfolio_uses_research_candidates = False
    if portfolio.holdings.empty:
        augmented_signals = signals.copy()
        augmented_tickers = research_candidates["Ticker"].head(max(mandate.max_holdings * 3, 12))
        augmented_signals.loc[augmented_signals["Ticker"].isin(augmented_tickers), "Eligible Long"] = True
        candidate_portfolio = optimize_portfolio(augmented_signals, returns, mandate)
        if not candidate_portfolio.holdings.empty:
            portfolio = candidate_portfolio
            portfolio_uses_research_candidates = True
    leader = opportunities.iloc[0] if not opportunities.empty else fallback_stocks.iloc[0]
    selected = str(leader["Ticker"])
    gate = evidence_gate(diagnostics)
    regime = market_regime(closes[mandate.benchmark])
    comparison, cumulative = strategy_comparison(signals, portfolio, returns, mandate.benchmark)

    if opportunities.empty:
        title = "No stock passes the current long screen."
        copy = "The highest-ranked name remains visible for research, but it does not satisfy the selected probability and return filters."
    elif gate.level == "positive":
        title = f"{html.escape(leader['Ticker'])} leads a baseline-validated {mandate.horizon}-day screen."
        copy = f"The model estimates a <b>{pct(leader['Probability Positive'], 0)} probability of a positive return</b> and a <b>{pct(leader['Expected Return'])} expected return</b>. The signal passed both naive out-of-fold baselines, but it remains uncertain and is not a trade instruction."
    else:
        title = f"{html.escape(leader['Ticker'])} is the highest-ranked current research signal."
        copy = f"The ranking estimates a <b>{pct(leader['Probability Positive'], 0)} probability of a positive return</b> and a <b>{pct(leader['Expected Return'])} expected return</b>. The evidence gate is <b>{html.escape(gate.label.lower())}</b>, so this should be treated as exploratory research rather than an actionable call."
    st.markdown(
        f'<section class="hero"><div class="eyebrow">Highest-ranked current setup</div><h1>{title}</h1><p>{copy}</p><div class="chips"><span class="chip">As of {leader["As Of"].date()}</span><span class="chip">{diagnostics.folds} purged date folds</span><span class="chip">{len(data)} symbols loaded</span><span class="chip">Evidence: {html.escape(gate.label)}</span><span class="chip">Regime: {html.escape(regime.label)}</span></div></section>',
        unsafe_allow_html=True,
    )
    metrics(
        [
            ("Probability positive", pct(leader["Probability Positive"], 0), f"Range {pct(leader['Chance Low'], 0)} to {pct(leader['Chance High'], 0)}"),
            ("Expected return", pct(leader["Expected Return"]), f"Band {pct(leader['Return Low'])} to {pct(leader['Return High'])}"),
            ("Standalone risk", pct(leader["Risk Score"], 0), "Lower is more favorable"),
            ("Market regime", regime.label, f"20-day vol {pct(regime.volatility20, 0)}"),
            ("Evidence gate", gate.label, "Compared with naive baselines"),
        ]
    )

    tabs = st.tabs(
        [
            "Decision Brief",
            "Best Stocks",
            "Best Portfolio",
            "Stock Explorer",
            "Current Events",
            "Data and Method",
            "Model Math",
        ]
    )
    with tabs[0]:
        st.markdown('<p class="section-kicker">Decision view</p>', unsafe_allow_html=True)
        show_evidence(gate)
        st.info(f"**Market regime: {regime.label}.** {regime.detail} Regime is context only and does not mechanically change the forecast.")
        st.subheader("Opportunity Map")
        st.caption(
            "Higher means stronger expected return, farther left means lower standalone risk, larger bubbles mean higher probability-positive, gold rings mark portfolio holdings, and the gold diamond marks the top-ranked signal."
        )
        st.plotly_chart(opportunity_map(signals, portfolio.holdings, str(leader["Ticker"])), width="stretch")
        st.markdown(
            '<div class="insight"><b>How to use this plot:</b> start with the upper-left area, then check the gold portfolio rings. A highly ranked stock is not automatically a large portfolio weight because correlation, downside risk, sector exposure, and beta are evaluated separately.</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Portfolio comparison")
        st.caption(
            "This compares current selections over their historical return series. It is descriptive and selection-biased, not an out-of-sample claim that one strategy will outperform."
        )
        if comparison.empty:
            st.info("Not enough aligned history is available for the strategy comparison.")
        else:
            st.plotly_chart(comparison_chart(comparison), width="stretch")
            format_columns = {"Historical Annual Return": "{:.1%}", "Annual Volatility": "{:.1%}", "Maximum Drawdown": "{:.1%}", "Daily VaR 95%": "{:.2%}"}
            st.dataframe(comparison.style.format(format_columns), width="stretch", hide_index=True)
            if not cumulative.empty:
                with st.expander("View cumulative historical paths"):
                    st.line_chart(cumulative)
        pdf_bytes = build_research_brief(
            leader,
            {"label": gate.label, "detail": gate.detail},
            {"label": regime.label, "detail": regime.detail},
            portfolio.holdings,
            comparison,
            research_candidates.head(8),
            mandate.horizon,
            mandate.target_return,
        )
        st.download_button(
            "Download investment-research PDF brief",
            pdf_bytes,
            file_name=f"{leader['Ticker']}_quant_research_brief.pdf",
            mime="application/pdf",
            type="primary",
        )

    with tabs[1]:
        st.markdown('<p class="section-kicker">Individual research signals</p>', unsafe_allow_html=True)
        st.subheader("Best Stocks")
        st.caption("Rankings are research estimates. The evidence gate above determines how much weight the model deserves today.")
        if opportunities.empty:
            st.warning(
                "No stock currently passes the strict long screen. The highest-ranked research candidates remain visible below for comparison and further analysis."
            )
        columns = [
            "Ticker",
            "Sector",
            "Expected Return",
            "Probability Positive",
            "Probability > 2%",
            "Probability > 5%",
            "Probability Target",
            "Risk Score",
            "Signal Score",
            "High Conviction",
            "Reliability",
            "Feature Coverage",
        ]
        percent_columns = {
            key: "{:.1%}" for key in columns if "Probability" in key or key in {"Expected Return", "Risk Score", "Reliability", "Feature Coverage"}
        }
        st.dataframe(research_candidates[columns].head(20).style.format(percent_columns).format({"Signal Score": "{:.3f}"}), width="stretch", hide_index=True)
        limited_history = signals[signals["Feature Coverage"] < 0.65][
            ["Ticker", "Sector", "close", "Expected Return", "Probability Positive", "Risk Score", "Feature Coverage"]
        ]
        if not limited_history.empty:
            with st.expander("New and limited-history stocks"):
                st.dataframe(
                    limited_history.style.format(
                        {
                            "close": "${:,.2f}",
                            "Expected Return": "{:.1%}",
                            "Probability Positive": "{:.1%}",
                            "Risk Score": "{:.1%}",
                            "Feature Coverage": "{:.1%}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("These stocks are searchable, but they cannot enter the long screen until at least 65% of the model features are available.")
        with st.expander("Bearish and hedge research"):
            research = signals[~signals["Eligible Long"]][["Ticker", "Sector", "Expected Return", "Probability Positive", "Risk Score", "Defensive"]]
            st.dataframe(
                research.style.format({"Expected Return": "{:.1%}", "Probability Positive": "{:.1%}", "Risk Score": "{:.1%}"}),
                width="stretch",
                hide_index=True,
            )
            st.caption("These names are retained for context or portfolio hedging. They are not presented as long recommendations.")

    with tabs[2]:
        st.markdown('<p class="section-kicker">Diversified allocation</p>', unsafe_allow_html=True)
        st.subheader("Best Portfolio")
        st.caption("The portfolio can include a lower-ranked candidate or defensive asset when it improves diversification or controls downside risk.")
        if portfolio_uses_research_candidates:
            st.warning(
                "No stock passed the strict long screen, so this exploratory research portfolio uses the highest-ranked candidates. All selected position, sector, and beta limits still apply."
            )
        if portfolio.holdings.empty:
            st.warning("No feasible portfolio was produced under the current constraints.")
        else:
            metrics(
                [
                    ("Expected basket return", pct(portfolio.metrics.get("Expected Horizon Return")), f"Next {mandate.horizon} trading days"),
                    ("Weighted probability", pct(portfolio.metrics.get("Weighted Probability"), 0), "Weighted positive-return estimate"),
                    ("Daily VaR", pct(portfolio.metrics.get("Daily VaR 95%")), "Historical 95% VaR"),
                    ("Tail loss / CVaR", pct(portfolio.metrics.get("Daily CVaR 95%")), "Average loss beyond VaR"),
                ]
            )
            left, right = st.columns([0.8, 1.2])
            with left:
                chart = px.pie(portfolio.holdings, names="Ticker", values="Weight", hole=0.55, title="Portfolio allocation")
                chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(chart, width="stretch")
            with right:
                st.dataframe(
                    portfolio.holdings.style.format(
                        {"Weight": "{:.1%}", "Expected Return": "{:.2%}", "Probability Positive": "{:.1%}", "Risk Score": "{:.1%}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
            weights = portfolio.holdings.set_index("Ticker")["Weight"]
            st.subheader("Downside scenarios")
            st.dataframe(
                stress_tests(weights, signals.set_index("Ticker")["beta60"], returns).style.format({"Estimated Return": "{:.2%}"}),
                width="stretch",
                hide_index=True,
            )
        for warning in portfolio.warnings:
            st.warning(warning)

    with tabs[3]:
        st.markdown('<p class="section-kicker">Single-name view</p>', unsafe_allow_html=True)
        stock = st.selectbox("Stock for detailed research", signals["Ticker"].tolist(), index=signals["Ticker"].tolist().index(selected))
        row = signals.loc[signals["Ticker"].eq(stock)].iloc[0]
        metrics(
            [
                ("Latest price", f"USD {row['close']:,.2f}", f"1-day return {pct(row['ret_1'])}"),
                ("Probability positive", pct(row["Probability Positive"], 0), f"Range {pct(row['Chance Low'], 0)} to {pct(row['Chance High'], 0)}"),
                ("Expected return", pct(row["Expected Return"]), f"Band {pct(row['Return Low'])} to {pct(row['Return High'])}"),
                ("Risk level", pct(row["Risk Score"], 0), f"Beta {num(row['beta60'])} | VaR {pct(row['var95'])}"),
                ("Data coverage", pct(row["Feature Coverage"], 0), "65% required for the long screen"),
            ]
        )
        st.plotly_chart(price_chart(closes, stock), width="stretch")
        st.plotly_chart(score_breakdown_chart(row), width="stretch")
        st.caption(
            "Signal score = 45% probability rank + 40% expected-return rank + 15% lower-risk credit. The chart shows the exact weighted contribution of each component."
        )
        left, right = st.columns(2)
        with left:
            st.subheader("Why the model sees this setup")
            for point in explain_signal(row):
                st.write(f"• {point}")
        with right:
            st.subheader("Risk check")
            for point in [
                f"60-day drawdown: {pct(row['drawdown60'])}.",
                f"Annualized 20-day volatility: {pct(row['vol20'])}.",
                f"Historical daily CVaR: {pct(row['cvar95'])}.",
                "Short-horizon forecasts are uncertain. Use the return band, not only the point estimate.",
            ]:
                st.write(f"• {point}")

    with tabs[4]:
        st.markdown('<p class="section-kicker">Current events</p>', unsafe_allow_html=True)
        st.subheader("News that may change the story")
        st.caption("Current events add qualitative context to the selected stock. They do not modify the quantitative forecast or confirm that it is correct.")
        research_input = st.text_input(
            "Search any stock ticker for current events",
            value=selected,
            key="events_research",
            help="This search is not limited to the model universe. International symbols must include their Yahoo Finance exchange suffix.",
        )
        research_ticker = normalize_ticker(research_input)
        valid_event_ticker = is_valid_ticker(research_ticker)
        matching_rows = signals.loc[signals["Ticker"].eq(research_ticker)]
        if not valid_event_ticker:
            st.error("Enter a valid ticker using letters, numbers, periods, ampersands, or hyphens.")
        elif not matching_rows.empty:
            research_row = matching_rows.iloc[0]
            metrics(
                [
                    ("Stock", research_ticker, str(research_row["Sector"])),
                    ("Current rank", f"#{int(matching_rows.index[0]) + 1}", f"Score {research_row['Signal Score']:.3f}"),
                    ("Expected return", pct(research_row["Expected Return"]), f"{mandate.horizon}-day model"),
                    ("Risk", pct(research_row["Risk Score"], 0), "Headlines can change this context"),
                ]
            )
        else:
            metrics(
                [
                    ("Stock", research_ticker, SECTORS.get(research_ticker, "Outside active model")),
                    ("Model status", "News lookup", "Add it in the sidebar for quantitative analysis"),
                ]
            )
        event_key = f"current_events_{research_ticker}"
        if st.button("Update current events", type="primary", width="stretch", disabled=not valid_event_ticker):
            loader = show_loader("Loading current events...")
            try:
                event_sector = str(matching_rows.iloc[0]["Sector"]) if not matching_rows.empty else SECTORS.get(research_ticker, "Other")
                st.session_state[event_key] = cached_current_events(research_ticker, event_sector)
            finally:
                loader.empty()
        if event_key in st.session_state:
            events, event_warnings = st.session_state[event_key]
            for warning in event_warnings[:3]:
                st.caption(f"• {warning}")
            render_events(events)
            if events:
                with st.expander("Event relevance table"):
                    event_frame = pd.DataFrame(events).drop(columns=["Link"])
                    st.dataframe(event_frame.style.format({"Relevance Score": "{:.0%}"}), width="stretch", hide_index=True)
        else:
            st.info("Select a stock, then click Update current events to load recent headlines, sources, relevance, and why each event matters.")
    with tabs[5]:
        st.markdown('<p class="section-kicker">Evidence and controls</p>', unsafe_allow_html=True)
        show_evidence(gate)
        evidence = pd.DataFrame(
            [
                ("Probability loss (Brier)", diagnostics.brier, diagnostics.baseline_brier, gate.probability_pass),
                ("Return error (MAE)", diagnostics.mae, diagnostics.baseline_mae, gate.return_pass),
                ("Return rank correlation", diagnostics.rank_ic, np.nan, np.nan),
                ("Out-of-fold samples", diagnostics.samples, np.nan, np.nan),
            ],
            columns=["Metric", "Model", "Naive baseline", "Beats baseline"],
        )
        st.subheader("Model validation")
        st.dataframe(evidence.style.format({"Model": "{:.4f}", "Naive baseline": "{:.4f}"}), width="stretch", hide_index=True)
        if not diagnostics.calibration.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=diagnostics.calibration["predicted"], y=diagnostics.calibration["observed"], mode="lines+markers", name="Observed"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot"}, name="Perfect calibration"))
            fig.update_layout(
                title="Probability calibration",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Predicted probability",
                yaxis_title="Observed frequency",
            )
            st.plotly_chart(fig, width="stretch")
        if st.button("Run full portfolio-matched walk-forward backtest"):
            loader = show_loader("Running walk-forward backtest...")
            try:
                result = run_walk_forward(data, mandate)
            finally:
                loader.empty()
            if not result.equity.empty:
                st.line_chart(result.equity)
                st.dataframe(pd.DataFrame(result.metrics.items(), columns=["Metric", "Value"]), width="stretch", hide_index=True)
            for warning in result.warnings:
                st.warning(warning)
        st.subheader("Data health")
        st.dataframe(data_health(data), width="stretch", hide_index=True)
        for warning in [*data_warnings, *diagnostics.warnings]:
            st.caption(f"• {warning}")

    with tabs[6]:
        st.markdown('<p class="section-kicker">Transparent methodology</p>', unsafe_allow_html=True)
        st.subheader("The math behind the dashboard")
        st.caption("These are the calculations used by the live dashboard. They explain the research outputs, not guaranteed future results.")

        st.markdown("#### 1. Market measurements")
        st.latex(r"R_{t,n} = \frac{P_t}{P_{t-n}} - 1")
        st.write("Returns compare the latest price with the price 1, 5, 20, or 60 trading days earlier.")
        st.latex(r"\sigma_{20} = \operatorname{stdev}(R_{daily,20}) \times \sqrt{252}")
        st.write("Annualized volatility scales recent daily-return variation to a trading year.")
        st.latex(r"\beta = \frac{\operatorname{Cov}(R_{stock}, R_{SPY})}{\operatorname{Var}(R_{SPY})}")
        st.write("Beta measures the stock's historical sensitivity to the benchmark. Drawdown, historical VaR, and CVaR measure downside behavior.")

        st.markdown("#### 2. Forecast target and validation")
        st.latex(r"R_{t,h} = \frac{P_{t+h}}{P_t} - 1")
        st.write(
            "A gradient-boosting classifier estimates the probability that the future return is positive. A separate gradient-boosting regressor estimates the expected future return over the selected holding horizon."
        )
        st.info(
            "The models are tested with four purged date folds. A purge prevents the training set from using observations whose forward return overlaps the validation period. Probability estimates are calibrated from genuine out-of-fold predictions."
        )

        st.markdown("#### 3. Stock ranking")
        st.latex(r"\text{Signal Score} = 0.45(\text{probability rank}) + 0.40(\text{return rank}) + 0.15(\text{lower-risk credit})")
        st.write(
            "Ranks are cross-sectional percentile ranks within the current model run. The risk component combines volatility, CVaR, and beta distance from one."
        )

        st.markdown("#### 4. Portfolio construction")
        st.latex(r"\max_w \left(w^T\mu - \lambda w^T\Sigma w - 0.10\sum_i w_i^2\right)")
        st.write(
            "The optimizer seeks stronger expected return while penalizing covariance risk and concentration. The covariance estimate is based on historical returns and is scaled to the selected horizon."
        )
        st.latex(r"\sum_i w_i = 1, \quad 0 \leq w_i \leq \text{position limit}")
        st.latex(r"\text{sector weight} \leq \text{sector cap}, \quad \sum_i w_i\beta_i \leq \text{beta cap}")
        st.write(
            "If the selected constraints cannot be satisfied, the dashboard intentionally publishes no portfolio instead of showing weights that break the limits."
        )

        st.markdown("#### 5. Evidence gate and interpretation")
        st.write(
            "The evidence gate compares out-of-fold probability loss (Brier score) with a naive probability baseline and return error (MAE) with a naive median-return baseline. A validated edge requires both forecast components to beat their respective baselines."
        )
        st.warning(
            "Current Events add qualitative context only. Headlines do not change the quantitative forecast, and no dashboard output is an investment recommendation or a guaranteed return."
        )
    st.divider()
    st.caption(
        "Research and education only. Market data and news may be delayed, incomplete, or wrong. Outputs are estimates, not investment recommendations or guaranteed returns."
    )


if __name__ == "__main__":
    main()
