import pandas as pd

from quantdash import data as market_data


def _bars(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    return pd.DataFrame({"Close": 100.0, "Adj Close": 100.0, "Volume": 1_000_000}, index=index)


def test_data_recovery_prioritizes_benchmark_and_minimum_universe(monkeypatch):
    def fail_batch(*args, **kwargs):
        raise RuntimeError("upstream unavailable")

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            return _bars()

    monkeypatch.setattr(market_data, "_download_batch", fail_batch)
    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)
    prices, warnings = market_data.download_prices(["AAPL", "MSFT", "NVDA", "AMZN", "SPY"])
    assert "SPY" in prices
    assert len(prices) >= market_data.MINIMUM_SYMBOLS
    assert any("Primary market-data request failed" in warning for warning in warnings)


def test_new_listing_with_short_history_remains_available(monkeypatch):
    short_bars = _bars(40)

    def partial_batch(*args, **kwargs):
        columns = pd.MultiIndex.from_product([["SKHY"], short_bars.columns])
        return pd.DataFrame(short_bars.to_numpy(), index=short_bars.index, columns=columns)

    monkeypatch.setattr(market_data, "_download_batch", partial_batch)
    prices, warnings = market_data.download_prices(["SKHY"])
    assert len(prices["SKHY"]) == 40
    assert any("Limited history for SKHY" in warning for warning in warnings)
