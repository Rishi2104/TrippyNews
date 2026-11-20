"""Full-text article extraction using trafilatura.

Extracts the main body text, author, and date from an article URL.
Handles paywalls and extraction failures by falling back to feed-level
metadata and flagging the article as ``limited``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

import trafilatura

from newscheck.config import ExtractConfig, Settings, load_blocklist
from newscheck.ingest.feeds import FeedItem
from newscheck.ingest.respect import RateLimiter, RobotsChecker
from newscheck.log import get_logger
from newscheck.models import Article, ExtractionStatus, SourceMode

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Outcome of attempting to extract full text from an article URL."""

    full_text: str | None
    author: str | None
    title: str | None
    status: ExtractionStatus


class ArticleExtractor:
    """Downloads and extracts article content with politeness controls.

    Usage::

        extractor = ArticleExtractor(settings)
        article = extractor.extract(feed_item, source_id=1, run_id=42)
    """

    def __init__(self, settings: Settings) -> None:
        cfg: ExtractConfig = settings.extract
        self._max_length = cfg.max_content_length
        self._hash_algo = cfg.hash_algorithm
        self._user_agent = settings.ingest.user_agent
        self._timeout = settings.ingest.request_timeout_seconds

        self._robots = RobotsChecker(self._user_agent, timeout=self._timeout)
        self._limiter = RateLimiter(min_delay=settings.ingest.rate_limit_per_domain_seconds)
        blocklist_path = settings.ingest.blocklist_file
        import pathlib

        p = pathlib.Path(blocklist_path)
        self._blocklist = load_blocklist(p if p.is_file() else pathlib.Path.cwd() / blocklist_path)

    def extract(self, item: FeedItem, *, source_id: int, run_id: int) -> Article:
        """Fetch and extract full text for a feed item, returning an Article.

        On failure or headline_only mode, returns an Article with ``extraction_status == LIMITED``
        containing only the headline, snippet, and URL from the feed.
        """
        # L8: If source is headline_only mode, do not fetch full text
        if getattr(item.source, "mode", SourceMode.CLAIMS) == SourceMode.HEADLINE_ONLY:
            return Article(
                source_id=source_id,
                url=item.url,
                title=item.title,
                snippet="",
                full_text=None,
                author=item.author,
                published_at=item.published_at,
                fetched_at=datetime.now(UTC),
                content_hash=self._compute_hash(f"{item.title}|{item.url}"),
                extraction_status=ExtractionStatus.LIMITED,
                wire_origin=None,
                run_id=run_id,
            )

        # L9: Check blocklist
        parsed = urlparse(item.url)
        if parsed.netloc.lower() in self._blocklist["domains"] or item.url in self._blocklist["urls"]:
            logger.info("article_skipped_blocklisted", url=item.url)
            return Article(
                source_id=source_id,
                url=item.url,
                title=item.title,
                snippet="",
                full_text=None,
                author=item.author,
                published_at=item.published_at,
                fetched_at=datetime.now(UTC),
                content_hash=self._compute_hash(f"{item.title}|{item.url}"),
                extraction_status=ExtractionStatus.FAILED,
                wire_origin=None,
                run_id=run_id,
            )

        result = self._try_extract(item.url)

        # Prefer extracted title, fall back to feed title.
        title = result.title or item.title

        # Use extracted author or feed-level author.
        author = result.author or item.author

        # Build the content hash from available text.
        hash_input = result.full_text or f"{title}|{item.url}"
        content_hash = self._compute_hash(hash_input)

        return Article(
            source_id=source_id,
            url=item.url,
            title=title,
            snippet=item.snippet,
            full_text=result.full_text,
            author=author,
            published_at=item.published_at,
            fetched_at=datetime.now(UTC),
            content_hash=content_hash,
            extraction_status=result.status,
            wire_origin=None,  # set by the wire detector later
            run_id=run_id,
        )

    def _try_extract(self, url: str) -> ExtractionResult:
        """Attempt full-text extraction for a single URL."""
        domain = urlparse(url).netloc

        # Check robots.txt for the article URL.
        if not self._robots.is_allowed(url):
            logger.info("article_blocked_by_robots", url=url)
            return ExtractionResult(full_text=None, author=None, title=None, status=ExtractionStatus.LIMITED)

        self._limiter.wait(domain)

        try:
            downloaded = trafilatura.fetch_url(url)
        except Exception:
            logger.warning("article_download_failed", url=url)
            return ExtractionResult(full_text=None, author=None, title=None, status=ExtractionStatus.FAILED)

        if downloaded is None:
            logger.info("article_download_empty", url=url)
            return ExtractionResult(full_text=None, author=None, title=None, status=ExtractionStatus.LIMITED)

        # Extract body text.
        try:
            body = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_recall=False,
            )
        except Exception:
            logger.warning("article_extract_error", url=url)
            return ExtractionResult(full_text=None, author=None, title=None, status=ExtractionStatus.FAILED)

        if not body:
            logger.info("article_no_body", url=url)
            return ExtractionResult(full_text=None, author=None, title=None, status=ExtractionStatus.LIMITED)

        # Extract metadata (title, author).
        metadata = trafilatura.extract(
            downloaded,
            output_format="xml",
            include_comments=False,
        )
        meta_title: str | None = None
        meta_author: str | None = None
        if metadata:
            meta_title = _extract_xml_field(metadata, "title")
            meta_author = _extract_xml_field(metadata, "author")

        # Truncate if too long.
        if len(body) > self._max_length:
            body = body[: self._max_length]
            logger.info("article_truncated", url=url, length=len(body))

        return ExtractionResult(
            full_text=body,
            author=meta_author,
            title=meta_title,
            status=ExtractionStatus.FULL,
        )

    def _compute_hash(self, text: str) -> str:
        """Compute a content hash for deduplication."""
        h = hashlib.new(self._hash_algo)
        h.update(text.encode("utf-8"))
        return h.hexdigest()


def _extract_xml_field(xml_str: str, field: str) -> str | None:
    """Pull a simple text field from trafilatura's XML output.

    This avoids importing a full XML parser for two fields.
    """
    import re

    pattern = rf"<{field}>(.*?)</{field}>"
    match = re.search(pattern, xml_str, re.DOTALL)
    if match:
        value = match.group(1).strip()
        return value if value else None
    return None
