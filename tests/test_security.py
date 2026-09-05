import pytest

from quantdash.security import normalize_ticker, safe_external_url, validate_tickers, validate_upload_size


def test_ticker_validation_deduplicates_and_rejects_payloads():
    assert validate_tickers([" aapl ", "AAPL", "<script>", "BRK.B", "7203.T", "VOW3.DE", "SHEL.L", "M&M.NS"]) == [
        "AAPL",
        "BRK-B",
        "7203.T",
        "VOW3.DE",
        "SHEL.L",
        "M&M.NS",
        "SPY",
    ]
    assert normalize_ticker("000660.KS") == "000660.KS"


def test_upload_limit():
    with pytest.raises(ValueError):
        validate_upload_size(100_001)


def test_external_url_allowlist():
    assert safe_external_url("https://example.com/x") == "https://example.com/x"
    assert safe_external_url("javascript:alert(1)") is None
