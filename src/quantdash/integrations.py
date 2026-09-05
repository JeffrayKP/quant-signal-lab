from __future__ import annotations

import re
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests
from defusedxml import ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import COMPANY_NAMES
from .security import is_valid_ticker, normalize_ticker, safe_external_url


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.4, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SECTOR_TERMS = {
    "technology": ("semiconductor", "software", "cloud", "artificial intelligence", "chip"),
    "communication": ("advertising", "social media", "search", "streaming"),
    "consumer": ("consumer spending", "retail", "ecommerce", "housing"),
    "financials": ("interest rate", "banking", "credit", "capital markets"),
    "healthcare": ("fda", "drug", "clinical trial", "medicare"),
    "energy": ("oil", "natural gas", "opec", "energy prices"),
    "industrials": ("manufacturing", "aerospace", "defense", "construction"),
    "utilities": ("electricity", "renewable", "power demand", "interest rate"),
    "defensive": ("treasury", "gold", "inflation", "interest rate"),
    "benchmark": ("s&p 500", "federal reserve", "inflation", "jobs report"),
}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def _event_relevance(title: str, description: str, ticker: str, sector: str) -> tuple[float, str, str]:
    text = f"{title} {description}".lower()
    company = COMPANY_NAMES.get(ticker, ticker)
    direct_terms = {part.lower() for part in company.split() if len(part) >= 4}
    ticker_hit = len(ticker) >= 3 and bool(re.search(rf"\b{re.escape(ticker.lower())}\b", text))
    sector_terms = SECTOR_TERMS.get(sector.lower(), ())
    direct_hits = sum(term in text for term in direct_terms) + int(ticker_hit)
    sector_hits = sum(term in text for term in sector_terms)
    catalyst_terms = (
        "earnings",
        "guidance",
        "revenue",
        "profit",
        "acquisition",
        "lawsuit",
        "regulator",
        "launch",
        "contract",
        "downgrade",
        "upgrade",
        "dividend",
    )
    catalyst_hits = sum(term in text for term in catalyst_terms)
    if direct_hits:
        score = min(1.0, 0.68 + 0.08 * min(direct_hits, 2) + 0.04 * min(catalyst_hits, 2))
        why = "Direct company news that may affect earnings expectations, valuation, or near-term sentiment."
        return score, "Company-specific", why
    if sector_hits:
        score = min(0.78, 0.46 + 0.07 * min(sector_hits, 3) + 0.03 * min(catalyst_hits, 2))
        why = f"A {sector.lower()} development that may change demand, costs, regulation, or peer valuations."
        return score, "Sector context", why
    why = "Broad market context that may influence discount rates, volatility, or investor risk appetite."
    return 0.30, "Market context", why


def current_events(ticker: str, sector: str, limit: int = 12) -> tuple[list[dict], list[str]]:
    """Return recent, relevance-scored headlines from fixed RSS endpoints."""
    clean_ticker = normalize_ticker(ticker)
    if not clean_ticker or not is_valid_ticker(clean_ticker):
        return [], ["The ticker was not valid for current-event research."]
    ticker = clean_ticker
    company = COMPANY_NAMES.get(ticker, ticker)
    feeds = [
        ("Yahoo Finance", f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(ticker, safe='.-')}&region=US&lang=en-US", None),
        ("Google News", "https://news.google.com/rss/search", {"q": f'"{company}" {ticker} stock when:7d', "hl": "en-US", "gl": "US", "ceid": "US:en"}),
    ]
    rows: list[dict] = []
    warnings: list[str] = []
    seen: set[str] = set()
    session = _session()
    for feed_name, url, params in feeds:
        try:
            response = session.get(url, params=params, timeout=(4, 12), headers={"User-Agent": "QuantSignalLab/4.0"})
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                warnings.append(f"{feed_name} returned an oversized response and was skipped.")
                continue
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError):
            warnings.append(f"{feed_name} is temporarily unavailable.")
            continue
        for item in root.findall(".//item"):
            title = _clean_text(item.findtext("title") or "")
            normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            if not title or normalized in seen:
                continue
            link = safe_external_url((item.findtext("link") or "").strip())
            if link is None:
                continue
            seen.add(normalized)
            description = _clean_text(item.findtext("description") or "")
            source = _clean_text(item.findtext("source") or "") or feed_name
            published_raw = (item.findtext("pubDate") or "").strip()
            try:
                published = parsedate_to_datetime(published_raw).astimezone().strftime("%Y-%m-%d %H:%M %Z")
            except (TypeError, ValueError, IndexError, OverflowError):
                published = published_raw or "Time unavailable"
            score, relevance, why = _event_relevance(title, description, ticker, sector)
            rows.append(
                {
                    "Event": title,
                    "Source": source,
                    "Published": published,
                    "Relevance": relevance,
                    "Relevance Score": score,
                    "Why It Matters": why,
                    "Link": link,
                }
            )
    rows.sort(key=lambda row: row["Relevance Score"], reverse=True)
    if not rows and not warnings:
        warnings.append("No recent headlines were returned for this stock.")
    return rows[: min(max(limit, 1), 25)], warnings
