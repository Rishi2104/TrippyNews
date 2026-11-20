"""Tests for article content extraction."""

from __future__ import annotations

from unittest.mock import patch

from newscheck.config import Settings
from newscheck.extract.content import ArticleExtractor, _extract_xml_field
from newscheck.ingest.feeds import FeedItem
from newscheck.models import ExtractionStatus, ReliabilityTier, Source


class TestArticleExtractor:
    def test_extract_xml_field(self) -> None:
        xml_data = "<doc><title>Breaking News</title><author>Jane Doe</author></doc>"
        assert _extract_xml_field(xml_data, "title") == "Breaking News"
        assert _extract_xml_field(xml_data, "author") == "Jane Doe"
        assert _extract_xml_field(xml_data, "nonexistent") is None

    def test_extract_fallback_on_failure(self, settings: Settings) -> None:
        extractor = ArticleExtractor(settings)
        source = Source(
            id=1,
            name="Source",
            domain="example.com",
            feed_url="https://example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
        )
        item = FeedItem(
            source=source,
            url="https://example.com/blocked",
            title="Headline Only",
            snippet="Snippet text",
            published_at=None,
            author="Reporter",
        )

        with patch("trafilatura.fetch_url", return_value=None):
            article = extractor.extract(item, source_id=1, run_id=1)

        assert article.title == "Headline Only"
        assert article.full_text is None
        assert article.extraction_status == ExtractionStatus.LIMITED
        assert article.snippet == "Snippet text"
        assert article.content_hash != ""

    def test_extract_success(self, settings: Settings) -> None:
        extractor = ArticleExtractor(settings)
        source = Source(
            id=1,
            name="Source",
            domain="example.com",
            feed_url="https://example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
        )
        item = FeedItem(
            source=source,
            url="https://example.com/article",
            title="Feed Headline",
            snippet="Feed snippet",
            published_at=None,
            author=None,
        )

        html_content = "<html><body><article><p>This is a fully extracted story body.</p></article></body></html>"

        with (
            patch.object(extractor._robots, "is_allowed", return_value=True),
            patch("trafilatura.fetch_url", return_value=html_content),
            patch("trafilatura.extract", side_effect=["This is a fully extracted story body.", None]),
        ):
            article = extractor.extract(item, source_id=1, run_id=1)

        assert article.full_text == "This is a fully extracted story body."
        assert article.extraction_status == ExtractionStatus.FULL
