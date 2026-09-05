from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)
MINIMUM_PRICE_ROWS = 20
MODEL_READY_ROWS = 300
MINIMUM_SYMBOLS = 5
BENCHMARK_TICKER = "SPY"


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(0):
            frame = frame.xs(ticker, axis=1, level=0)
        elif ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
    if "Adj Close" not in frame and "Close" in frame:
        frame["Adj Close"] = frame["Close"]
    keep = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if c in frame]
    if "Close" not in keep:
        return pd.DataFrame()
    frame = frame[keep].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _add_available(raw: pd.DataFrame, tickers: list[str], data: dict[str, pd.DataFrame]) -> None:
    for ticker in tickers:
        frame = _normalize(raw, ticker)
        if len(frame) >= MINIMUM_PRICE_ROWS:
            data[ticker] = frame


def _download_batch(tickers: list[str], period: str, threads: bool, timeout: int) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        repair=False,
        progress=False,
        threads=threads,
        timeout=timeout,
        group_by="ticker",
    )


def download_prices(tickers: list[str], period: str = "5y") -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load market data quickly, then recover the benchmark and a usable minimum universe."""
    warnings: list[str] = []
    data: dict[str, pd.DataFrame] = {}
    ordered = list(dict.fromkeys(tickers))
    if BENCHMARK_TICKER in ordered:
        ordered.remove(BENCHMARK_TICKER)
        ordered.insert(0, BENCHMARK_TICKER)
    try:
        raw = _download_batch(ordered, period, threads=True, timeout=10)
        _add_available(raw, ordered, data)
    except Exception as exc:  # upstream failures are non-fatal
        LOGGER.warning("Market-data batch failed: %s", type(exc).__name__)
        warnings.append("Primary market-data request failed. Trying a smaller recovery set.")

    # A single fast batch can be rate-limited or partially fail. Recover only what is needed
    # to keep the model valid, placing SPY first because it is required for all benchmark features.
    recovery = [ticker for ticker in ordered if ticker not in data]
    required_recoveries = max(0, MINIMUM_SYMBOLS - len(data))
    if BENCHMARK_TICKER not in data and BENCHMARK_TICKER in recovery:
        required_recoveries = max(required_recoveries, 1)
    for ticker in recovery[: max(required_recoveries + 1, 1)]:
        try:
            frame = _normalize(yf.Ticker(ticker).history(period=period, auto_adjust=False, actions=False, timeout=10), ticker)
            if len(frame) >= MINIMUM_PRICE_ROWS:
                data[ticker] = frame
            else:
                warnings.append(f"{ticker}: insufficient history")
        except Exception:
            warnings.append(f"{ticker}: unavailable")
        if BENCHMARK_TICKER in data and len(data) >= MINIMUM_SYMBOLS:
            break
    skipped = [ticker for ticker in ordered if ticker not in data]
    if skipped:
        warnings.append(f"{len(skipped)} symbols were unavailable and were skipped for this refresh.")
    limited = [ticker for ticker, frame in data.items() if len(frame) < MODEL_READY_ROWS]
    if limited:
        warnings.append(
            f"Limited history for {', '.join(limited[:8])}. These stocks remain available for research but are excluded from high-conviction signals."
        )
    return data, warnings


def close_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {ticker: frame["Adj Close"].where(frame["Adj Close"].notna(), frame["Close"]) for ticker, frame in data.items()}
    return pd.DataFrame(closes).sort_index()


def return_matrix(data: dict[str, pd.DataFrame], min_coverage: float = 0.90) -> pd.DataFrame:
    """Never convert missing prices into artificial zero returns."""
    closes = close_matrix(data)
    returns = closes.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA)
    required = max(60, int(len(returns) * min_coverage))
    return returns.dropna(axis=1, thresh=required)


def data_health(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    rows = []
    for ticker, frame in data.items():
        last = frame.index.max()
        rows.append({"Ticker": ticker, "Rows": len(frame), "Latest": last.date(), "Age": (now - last.normalize()).days})
    return pd.DataFrame(rows).sort_values("Ticker")
