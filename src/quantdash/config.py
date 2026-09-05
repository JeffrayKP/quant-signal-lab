from dataclasses import dataclass

TRADING_DAYS = 252
DEFAULT_BENCHMARK = "SPY"
DEFAULT_UNIVERSE = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "TSM",
    "SKHY",
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",
    "LLY",
    "JNJ",
    "XOM",
    "CVX",
    "COST",
    "WMT",
    "HD",
    "MCD",
    "GE",
    "CAT",
    "RTX",
    "NEE",
    "XLV",
    "XLP",
    "GLD",
    "TLT",
    "SHY",
    "SPY",
)
DEFENSIVE = frozenset({"GLD", "TLT", "SHY", "XLV", "XLP"})
SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer",
    "GOOGL": "Communication",
    "META": "Communication",
    "AVGO": "Technology",
    "TSM": "Technology",
    "SKHY": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "LLY": "Healthcare",
    "JNJ": "Healthcare",
    "XLV": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "COST": "Consumer",
    "WMT": "Consumer",
    "HD": "Consumer",
    "MCD": "Consumer",
    "XLP": "Consumer",
    "GE": "Industrials",
    "CAT": "Industrials",
    "RTX": "Industrials",
    "NEE": "Utilities",
    "GLD": "Defensive",
    "TLT": "Defensive",
    "SHY": "Defensive",
    "SPY": "Benchmark",
}

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet Google",
    "META": "Meta Platforms",
    "AVGO": "Broadcom",
    "TSM": "Taiwan Semiconductor Manufacturing Company",
    "SKHY": "SK hynix",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "V": "Visa",
    "MA": "Mastercard",
    "LLY": "Eli Lilly",
    "JNJ": "Johnson & Johnson",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "COST": "Costco",
    "WMT": "Walmart",
    "HD": "Home Depot",
    "MCD": "McDonald's",
    "GE": "GE Aerospace",
    "CAT": "Caterpillar",
    "RTX": "RTX Raytheon",
    "NEE": "NextEra Energy",
    "XLV": "Health Care Select Sector SPDR",
    "XLP": "Consumer Staples Select Sector SPDR",
    "GLD": "SPDR Gold Shares",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
}


@dataclass(frozen=True)
class Mandate:
    horizon: int = 5
    target_return: float = 0.03
    max_holdings: int = 8
    max_weight: float = 0.20
    sector_cap: float = 0.40
    beta_cap: float = 1.05
    risk_aversion: float = 5.0
    min_probability: float = 0.50
    min_expected_return: float = 0.0
    transaction_cost_bps: float = 10.0
    benchmark: str = DEFAULT_BENCHMARK
