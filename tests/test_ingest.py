"""Tests for feed ingestion and politeness controls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from newscheck.config import Settings
from newscheck.ingest.feeds import FeedReader, _clean_html, compute_url_hash
from newscheck.ingest.respect import RateLimiter, RobotsChecker
from newscheck.models import ReliabilityTier, Source


class TestRobotsChecker:
    def test_default_allowed_on_failure(self) -> None:
        checker = RobotsChecker("NewsCheck/0.1")
        # Should not raise exception and return boolean
        allowed = checker.is_allowed("https://example.com/some/article")
        assert isinstance(allowed, bool)


class TestRateLimiter:
    def test_record_and_wait(self) -> None:
        limiter = RateLimiter(min_delay=0.01)
        limiter.record("example.com")
        limiter.wait("example.com")
        # Ensure it works without hanging


class TestFeedReader:
    def test_clean_html(self) -> None:
        html = "<p>This is <b>bold</b> and <a href='#'>a link</a>.</p>"
        cleaned = _clean_html(html)
        assert cleaned == "This is bold and a link."

    def test_compute_url_hash(self) -> None:
        h1 = compute_url_hash("https://example.com/a")
        h2 = compute_url_hash("https://example.com/a")
        h3 = compute_url_hash("https://example.com/b")
        assert h1 == h2
        assert h1 != h3

    def test_parse_fixture_feed(self, settings: Settings) -> None:
        import feedparser

        fixture_path = Path(__file__).parent / "fixtures" / "sample_feed.xml"
        feed_xml = fixture_path.read_text(encoding="utf-8")
        parsed_feed = feedparser.parse(feed_xml)

        reader = FeedReader(settings)
        source = Source(
            id=1,
            name="Fixture Source",
            domain="example-news.com",
            feed_url="https://example-news.com/feed.xml",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
        )

        with (
            patch.object(reader._robots, "is_allowed", return_value=True),
            patch("feedparser.parse", return_value=parsed_feed),
        ):
            items = reader._read_one(source)

        assert len(items) == 5
        assert items[0].title.startswith("Major earthquake strikes coastal region")
        assert items[0].url == "https://example-news.com/earthquake-coastal-region"
        assert items[0].published_at is not None
        assert "earthquake struck" in items[0].snippet
