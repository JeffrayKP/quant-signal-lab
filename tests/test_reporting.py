import pandas as pd
from pypdf import PdfReader

from quantdash.reporting import build_research_brief


def test_research_brief_is_two_page_pdf():
    leader = pd.Series(
        {
            "Ticker": "AAPL",
            "Expected Return": 0.02,
            "Probability Positive": 0.61,
            "Risk Score": 0.40,
            "Probability Target": 0.35,
            "Return Low": -0.03,
            "Return High": 0.07,
            "Reliability": 0.30,
        }
    )
    holdings = pd.DataFrame([{"Ticker": "AAPL", "Sector": "Technology", "Weight": 0.20, "Expected Return": 0.02, "Role": "Return candidate"}])
    comparison = pd.DataFrame(
        [{"Strategy": "SPY", "Historical Annual Return": 0.10, "Annual Volatility": 0.18, "Maximum Drawdown": -0.20, "Daily VaR 95%": -0.02}]
    )
    pdf = build_research_brief(
        leader,
        {"label": "Validated edge", "detail": "Both baselines passed."},
        {"label": "Risk-on", "detail": "Trend is positive."},
        holdings,
        comparison,
        pd.DataFrame([leader]),
        5,
        0.03,
    )
    reader = PdfReader(__import__("io").BytesIO(pdf))
    assert len(reader.pages) == 2
    assert "AAPL Decision Brief" in reader.pages[0].extract_text()
