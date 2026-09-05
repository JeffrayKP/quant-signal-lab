from quantdash.integrations import _event_relevance, current_events


def test_company_event_scores_above_market_context():
    direct = _event_relevance("Apple raises iPhone revenue guidance", "", "AAPL", "Technology")
    market = _event_relevance("Treasury yields move after jobs report", "", "AAPL", "Technology")
    assert direct[0] > market[0]
    assert direct[1] == "Company-specific"


def test_invalid_event_ticker_is_rejected_without_network_request():
    events, warnings = current_events("BAD;DROP", "Technology")
    assert events == []
    assert "not valid" in warnings[0]


def test_international_ticker_is_encoded_and_not_limited_to_model_universe(monkeypatch):
    class FakeResponse:
        content = b"""<rss><channel><item><title>M&amp;M reports revenue growth</title><link>https://example.com/story</link></item></channel></rss>"""

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url, **kwargs):
            if "feeds.finance.yahoo.com" in url:
                assert "M%26M.NS" in url
            return FakeResponse()

    monkeypatch.setattr("quantdash.integrations._session", lambda: FakeSession())
    events, warnings = current_events("M&M.NS", "Consumer")
    assert events
    assert not warnings
