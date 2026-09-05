import re
from urllib.parse import urlparse

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.&-]{0,19}$")
US_SHARE_CLASS_RE = re.compile(r"^[A-Z]{1,6}\.[AB]$")
MAX_TICKERS = 100
MAX_UPLOAD_BYTES = 100_000


def normalize_ticker(value: str) -> str:
    """Normalize user input without destroying Yahoo's international exchange suffixes."""
    ticker = str(value).strip().upper()
    return ticker.replace(".", "-") if US_SHARE_CLASS_RE.fullmatch(ticker) else ticker


def is_valid_ticker(value: str) -> bool:
    return bool(TICKER_RE.fullmatch(normalize_ticker(value)))


def validate_tickers(values: list[str], benchmark: str = "SPY") -> list[str]:
    output: list[str] = []
    for raw in values:
        ticker = normalize_ticker(raw)
        if not TICKER_RE.fullmatch(ticker):
            continue
        if ticker not in output:
            output.append(ticker)
        if len(output) >= MAX_TICKERS:
            break
    if benchmark not in output:
        output.append(benchmark)
    return output


def safe_external_url(value: str) -> str | None:
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    return value if parsed.scheme in {"http", "https"} and bool(parsed.netloc) else None


def validate_upload_size(size: int) -> None:
    if size > MAX_UPLOAD_BYTES:
        raise ValueError("Ticker upload exceeds the 100 KB safety limit.")
