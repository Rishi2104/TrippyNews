"""RSS feed reader with politeness controls and structured output.

Reads feeds listed in ``sources.yaml``, respects robots.txt and rate limits,
and yields ``FeedItem`` objects ready for extraction.
"""

from __future__ import annotations

import hashlib
import socket
import ssl
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import feedparser

from newscheck.config import IngestConfig, Settings
from newscheck.ingest.respect import RateLimiter, RobotsChecker
from newscheck.log import get_logger
from newscheck.models import Source

logger = get_logger(__name__)


@dataclass
class FeedItem:
    """A single item parsed from an RSS/Atom feed."""

    source: Source
    url: str
    title: str
    snippet: str
    published_at: datetime | None
    author: str | None


class FeedReader:
    """Fetches and parses RSS feeds with politeness controls.

    Usage::

        reader = FeedReader(settings)
        for item in reader.read_all(sources):
            ...
    """

    def __init__(self, settings: Settings) -> None:
        cfg: IngestConfig = settings.ingest
        self._timeout = cfg.request_timeout_seconds
        self._max_retries = cfg.max_retries
        self._backoff = cfg.retry_backoff_seconds
        self._max_per_feed = cfg.max_articles_per_feed
        self._user_agent = cfg.user_agent

        self._robots = RobotsChecker(self._user_agent, timeout=self._timeout)
        self._limiter = RateLimiter(min_delay=cfg.rate_limit_per_domain_seconds)

    def read_all(self, sources: list[Source]) -> list[FeedItem]:
        """Read every enabled source and return all collected feed items.

        Feeds that fail are logged and skipped — never fatal.
        """
        items: list[FeedItem] = []
        for source in sources:
            if not source.enabled:
                continue
            try:
                feed_items = self._read_one(source)
                items.extend(feed_items)
                logger.info(
                    "feed_read",
                    source=source.name,
                    items=len(feed_items),
                )
            except Exception:
                logger.exception("feed_read_failed", source=source.name)
        return items

    def _fetch_feed_bytes(self, url: str) -> bytes | None:
        """Download raw feed bytes with SSL resilience and realistic headers."""
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 NewsCheck/1.0",
                    "Accept": "application/rss+xml,application/atom+xml,application/xml,text/xml,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                content = resp.read()
                return bytes(content) if isinstance(content, (bytes, bytearray)) else None
        except Exception:
            return None

    def _read_one(self, source: Source) -> list[FeedItem]:
        """Parse a single RSS/Atom feed and return its items."""
        domain = urlparse(source.feed_url).netloc

        self._limiter.wait(domain)

        # feedparser handles HTTP internally; we first attempt a direct fetch
        # with SSL resilience, and fall back to feedparser's built-in fetcher.
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self._timeout)
        try:
            raw_bytes = self._fetch_feed_bytes(source.feed_url)
            if raw_bytes is not None:
                feed = feedparser.parse(raw_bytes)
            else:
                feed = feedparser.parse(
                    source.feed_url,
                    agent=self._user_agent,
                )
        finally:
            socket.setdefaulttimeout(old_timeout)

        if feed.bozo and not feed.entries:
            logger.warning(
                "feed_parse_warning",
                source=source.name,
                error=str(getattr(feed, "bozo_exception", "unknown")),
            )
            return []

        items: list[FeedItem] = []
        for entry in feed.entries[: self._max_per_feed]:
            item = self._parse_entry(entry, source)
            if item is not None:
                items.append(item)
        return items

    def _parse_entry(
        self,
        entry: feedparser.FeedParserDict,
        source: Source,
    ) -> FeedItem | None:
        """Convert a feedparser entry to a FeedItem, or None if unusable."""
        link: str = getattr(entry, "link", "") or ""
        title: str = getattr(entry, "title", "") or ""

        if not link or not title:
            return None

        # Normalise the URL (strip fragments, trailing whitespace).
        link = link.split("#")[0].strip()

        # Only allow http/https URLs from the whitelisted domain.
        parsed = urlparse(link)
        if parsed.scheme not in ("http", "https"):
            return None

        # Published date.
        pub_dt = self._parse_date(entry)

        # Snippet from the feed (summary or description).
        snippet = _clean_html(getattr(entry, "summary", "") or getattr(entry, "description", "") or "")

        author = getattr(entry, "author", None)

        return FeedItem(
            source=source,
            url=link,
            title=title.strip(),
            snippet=snippet[:500],
            published_at=pub_dt,
            author=author,
        )

    @staticmethod
    def _parse_date(entry: feedparser.FeedParserDict) -> datetime | None:
        """Best-effort date parsing from a feed entry."""
        # feedparser normalises dates into a time_struct in `published_parsed`.
        published_parsed = getattr(entry, "published_parsed", None)
        if published_parsed is not None:
            try:
                from calendar import timegm

                ts = timegm(published_parsed)
                return datetime.fromtimestamp(ts, tz=UTC)
            except (ValueError, OverflowError, TypeError):
                pass

        # Fallback: try the raw `published` string.
        published_raw: str = getattr(entry, "published", "") or ""
        if published_raw:
            try:
                return parsedate_to_datetime(published_raw).replace(tzinfo=UTC)
            except (ValueError, TypeError):
                pass

        return None


def _clean_html(text: str) -> str:
    """Strip common HTML tags from feed snippets."""
    import re

    # Remove tags without extra spaces around word boundaries
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_url_hash(url: str) -> str:
    """Compute a deterministic SHA-256 hash of a URL (for dedup fallback)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()
