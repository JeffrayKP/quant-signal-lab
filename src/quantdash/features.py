from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "dist_ma20",
    "dist_ma50",
    "rsi14",
    "vol20",
    "vol60",
    "downside_vol",
    "drawdown60",
    "volume_z",
    "beta60",
    "corr60",
    "relative20",
    "var95",
    "cvar95",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _rolling_cvar(values: pd.Series, window: int = 60) -> pd.Series:
    def tail_mean(x: np.ndarray) -> float:
        q = np.quantile(x, 0.05)
        tail = x[x <= q]
        return float(tail.mean()) if len(tail) else float(q)

    return values.rolling(window, min_periods=40).apply(tail_mean, raw=True)


def build_features(data: dict[str, pd.DataFrame], benchmark: str = "SPY") -> pd.DataFrame:
    benchmark_close = None
    if benchmark in data:
        benchmark_close = data[benchmark]["Adj Close"].dropna()
    frames: list[pd.DataFrame] = []
    for ticker, bars in data.items():
        close = bars["Adj Close"].where(bars["Adj Close"].notna(), bars["Close"])
        ret = close.pct_change(fill_method=None)
        out = pd.DataFrame(index=close.index)
        out["ticker"] = ticker
        out["close"] = close
        for window in (1, 5, 20, 60):
            out[f"ret_{window}"] = close.pct_change(window, fill_method=None)
        ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()
        out["dist_ma20"] = close / ma20 - 1
        out["dist_ma50"] = close / ma50 - 1
        out["rsi14"] = _rsi(close)
        out["vol20"] = ret.rolling(20).std() * np.sqrt(252)
        out["vol60"] = ret.rolling(60).std() * np.sqrt(252)
        out["downside_vol"] = ret.clip(upper=0).rolling(60).std() * np.sqrt(252)
        out["drawdown60"] = close / close.rolling(60).max() - 1
        volume_source = bars["Volume"] if "Volume" in bars else pd.Series(np.nan, index=bars.index)
        volume = pd.to_numeric(volume_source, errors="coerce")
        out["volume_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        out[["beta60", "corr60", "relative20"]] = np.nan
        if benchmark_close is not None:
            bench = benchmark_close.reindex(close.index)
            bench_ret = bench.pct_change(fill_method=None)
            out["beta60"] = ret.rolling(60).cov(bench_ret) / bench_ret.rolling(60).var()
            out["corr60"] = ret.rolling(60).corr(bench_ret)
            out["relative20"] = out["ret_20"] - bench.pct_change(20, fill_method=None)
        out["var95"] = ret.rolling(60).quantile(0.05)
        out["cvar95"] = _rolling_cvar(ret)
        out.index.name = "date"
        frames.append(out.reset_index())
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def add_targets(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = panel.sort_values(["ticker", "date"]).copy()
    future = frame.groupby("ticker")["close"].shift(-horizon)
    frame["future_return"] = future / frame["close"] - 1
    frame["target_positive"] = (frame["future_return"] > 0).astype(float)
    frame.loc[frame["future_return"].isna(), "target_positive"] = np.nan
    return frame
