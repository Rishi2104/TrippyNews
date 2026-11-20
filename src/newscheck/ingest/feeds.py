"""RSS feed reader with politeness controls and structured output.

Reads feeds listed in ``sources.yaml``, respects robots.txt and rate limits,
verifies TLS strictly, uses honest User-Agent identity, normalizes and cleans URLs,
and yields ``FeedItem`` objects ready for extraction.
"""

from __future__ import annotations

import hashlib
import html
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import feedparser

from newscheck.config import IngestConfig, Settings, load_blocklist
from newscheck.ingest.respect import RateLimiter, RobotsChecker
from newscheck.log import get_logger
from newscheck.models import LicenseStatus, Source

logger = get_logger(__name__)

# Tracking query parameters to strip during URL normalisation
_TRACKING_PARAMS: set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "traffic_source",
    "ref",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "source",
}


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
    """Fetches and parses RSS feeds with strict politeness and legal controls."""

    def __init__(self, settings: Settings) -> None:
        cfg: IngestConfig = settings.ingest
        self._timeout = cfg.request_timeout_seconds
        self._max_retries = cfg.max_retries
        self._backoff = cfg.retry_backoff_seconds
        self._max_per_feed = cfg.max_articles_per_feed
        self._user_agent = cfg.user_agent

        self._limiter = RateLimiter(min_delay=cfg.rate_limit_per_domain_seconds)
        self._robots = RobotsChecker(self._user_agent, timeout=self._timeout, rate_limiter=self._limiter)

        # HTTP caching metadata per feed URL: url -> {"etag": ..., "last_modified": ...}
        self._http_cache: dict[str, dict[str, str]] = {}
        # Track sources/origins that returned 403/429
        self.blocked_sources: set[str] = set()

        # Load blocklist
        blocklist_path = settings.ingest.blocklist_file
        import pathlib

        p = pathlib.Path(blocklist_path)
        self._blocklist = load_blocklist(p if p.is_file() else pathlib.Path.cwd() / blocklist_path)

    def read_all(self, sources: list[Source]) -> list[FeedItem]:
        """Read every enabled, non-prohibited source and return all collected feed items."""
        items: list[FeedItem] = []
        for source in sources:
            if not source.enabled:
                continue

            # L8: Ingest nothing from prohibited sources
            if source.license_status == LicenseStatus.PROHIBITED:
                logger.info("source_skipped_license_prohibited", source=source.name)
                continue

            # L9: Check blocklist
            if source.domain.lower() in self._blocklist["domains"]:
                logger.info("source_skipped_domain_blocklisted", source=source.name, domain=source.domain)
                continue
            if source.feed_url in self._blocklist["urls"]:
                logger.info("source_skipped_feed_url_blocklisted", source=source.name, url=source.feed_url)
                continue

            try:
                feed_items = self._read_one(source)
                items.extend(feed_items)
                logger.info("feed_read", source=source.name, items=len(feed_items))
            except Exception:
                logger.exception("feed_read_failed", source=source.name)

        return items

    def _fetch_feed_bytes(self, url: str, source_domain: str) -> tuple[bytes | None, int | None]:
        """Download raw feed bytes with honest UA, strict TLS verification, retries, and backoff.

        Returns (content_bytes_or_none, http_status_code).
        """
        # Strict TLS verification
        ctx = ssl.create_default_context()

        headers: dict[str, str] = {
            "User-Agent": self._user_agent,
            "Accept": "application/rss+xml,application/atom+xml,application/xml,text/xml,*/*",
        }

        # C6: ETag / If-Modified-Since caching
        cached_meta = self._http_cache.get(url, {})
        if "etag" in cached_meta:
            headers["If-None-Match"] = cached_meta["etag"]
        if "last_modified" in cached_meta:
            headers["If-Modified-Since"] = cached_meta["last_modified"]

        for attempt in range(self._max_retries + 1):
            try:
                self._limiter.wait(source_domain)
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                    status = resp.status if hasattr(resp, "status") else 200

                    # Store caching headers
                    etag = resp.headers.get("ETag")
                    last_mod = resp.headers.get("Last-Modified")
                    if etag or last_mod:
                        self._http_cache[url] = {
                            k: v
                            for k, v in {"etag": etag, "last_modified": last_mod}.items()
                            if v
                        }

                    content = resp.read()
                    return bytes(content) if isinstance(content, (bytes, bytearray)) else None, status

            except urllib.error.HTTPError as exc:
                status = exc.code
                logger.warning(
                    "feed_fetch_http_error",
                    url=url,
                    status=status,
                    attempt=attempt + 1,
                )

                # 304 Not Modified
                if status == 304:
                    logger.info("feed_not_modified_304", url=url)
                    return b"", 304

                # L5: On 403 Forbidden or 429 Too Many Requests -> mark blocked and back off
                if status in (403, 429):
                    self.blocked_sources.add(source_domain)
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait_sec = min(float(retry_after), 10.0) if retry_after and retry_after.isdigit() else 5.0
                    logger.warning("source_blocked_or_throttled", domain=source_domain, status=status, wait=wait_sec)
                    time.sleep(wait_sec)
                    return None, status

                # 5xx: Server errors -> retry with exponential backoff
                if 500 <= status < 600 and attempt < self._max_retries:
                    wait_sec = self._backoff * (2**attempt)
                    logger.info("retrying_after_server_error", url=url, wait=wait_sec)
                    time.sleep(wait_sec)
                    continue

                return None, status

            except ssl.SSLError as exc:
                # L6: On certificate error, do not bypass; skip the source and log
                logger.error("feed_fetch_ssl_error", url=url, error=str(exc))
                return None, None

            except Exception as exc:
                logger.warning("feed_fetch_network_error", url=url, error=str(exc), attempt=attempt + 1)
                if attempt < self._max_retries:
                    time.sleep(self._backoff * (2**attempt))
                    continue
                return None, None

        return None, None

    def _read_one(self, source: Source) -> list[FeedItem]:
        """Parse a single RSS/Atom feed and return its items."""
        domain = urllib.parse.urlparse(source.feed_url).netloc

        # L4: Call is_allowed() before every feed fetch
        if not self._robots.is_allowed(source.feed_url):
            logger.warning("feed_blocked_by_robots_txt", source=source.name, url=source.feed_url)
            return []

        raw_bytes, status = self._fetch_feed_bytes(source.feed_url, domain)
        if raw_bytes is None:
            logger.warning("feed_bytes_unavailable", source=source.name, status=status)
            return []

        if status == 304:
            return []

        feed = feedparser.parse(raw_bytes)
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
        link: str = (getattr(entry, "link", None) or (entry.get("link", "") if isinstance(entry, dict) else "")) or ""
        title: str = (getattr(entry, "title", None) or (entry.get("title", "") if isinstance(entry, dict) else "")) or ""

        if not link or not title:
            return None

        # C2: URL normalisation
        norm_url = normalize_url(link)
        parsed = urllib.parse.urlparse(norm_url)

        # Require http or https scheme
        if parsed.scheme not in ("http", "https"):
            return None

        # C1: Enforce link host in source's allowed_domains or domain
        link_host = parsed.netloc.lower()
        # Strip port if present
        if ":" in link_host:
            link_host = link_host.split(":")[0]

        valid_domains = {d.lower() for d in source.allowed_domains}
        valid_domains.add(source.domain.lower())

        is_allowed_domain = False
        for allowed in valid_domains:
            if link_host == allowed or link_host.endswith("." + allowed):
                is_allowed_domain = True
                break

        if not is_allowed_domain:
            logger.debug(
                "entry_rejected_unvetted_domain",
                source=source.name,
                link_host=link_host,
                allowed=list(valid_domains),
            )
            return None

        # L9: Check blocklist for article URL and domain
        if link_host in self._blocklist["domains"] or norm_url in self._blocklist["urls"]:
            logger.info("entry_skipped_blocklisted", url=norm_url)
            return None

        # L4: Check robots.txt for the article URL before queuing
        if not self._robots.is_allowed(norm_url):
            logger.debug("entry_skipped_robots_disallowed", url=norm_url)
            return None

        # C3: Published date parsing with fallback, offset awareness, future clamp
        pub_dt = self._parse_date(entry)

        # C4: Clean HTML with html.unescape and script/style removal
        raw_snippet = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        snippet = clean_html(raw_snippet)

        author = getattr(entry, "author", None)
        if author:
            author = clean_html(str(author))

        return FeedItem(
            source=source,
            url=norm_url,
            title=title.strip(),
            snippet=snippet[:500],
            published_at=pub_dt,
            author=author,
        )

    @staticmethod
    def _parse_date(entry: feedparser.FeedParserDict) -> datetime | None:
        """Best-effort date parsing from a feed entry with timezone and future safety."""
        now = datetime.now(UTC)

        # 1. Try `published_parsed` then `updated_parsed`
        for attr in ("published_parsed", "updated_parsed"):
            st = getattr(entry, attr, None)
            if st is not None:
                try:
                    from calendar import timegm

                    ts = timegm(st)
                    dt = datetime.fromtimestamp(ts, tz=UTC)
                    # C3: Clamp future dates beyond 1 hour
                    if dt > now + timedelta(hours=1):
                        return now
                    return dt
                except (ValueError, OverflowError, TypeError):
                    pass

        # 2. Try raw published / updated string
        for attr in ("published", "updated"):
            raw_str: str = getattr(entry, attr, "") or ""
            if raw_str:
                try:
                    dt = parsedate_to_datetime(raw_str)
                    dt = dt.astimezone(UTC) if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
                    if dt > now + timedelta(hours=1):
                        return now
                    return dt
                except (ValueError, TypeError):
                    pass

        return None


def normalize_url(raw_url: str) -> str:
    """Normalise a URL: strip fragments, lowercase host, and remove tracking query params."""
    if not raw_url:
        return ""

    # Strip fragments and surrounding whitespace
    url = raw_url.split("#")[0].strip()
    parsed = urllib.parse.urlsplit(url)

    # Lowercase hostname
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path

    # Clean query parameters
    if parsed.query:
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        filtered_pairs = [
            (k, v)
            for k, v in query_pairs
            if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        new_query = urllib.parse.urlencode(filtered_pairs)
    else:
        new_query = ""

    return urllib.parse.urlunsplit((scheme, netloc, path, new_query, ""))


def clean_html(text: str) -> str:
    """Clean HTML from text: remove scripts, styles, decode entities, and normalize whitespace."""
    if not text:
        return ""

    # 1. Strip script and style blocks including their contents
    text = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", text, flags=re.IGNORECASE)

    # 2. Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Unescape HTML entities (&amp;, &#8217;, &quot;, etc.)
    text = html.unescape(text)

    # 4. Remove space before punctuation
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)

    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_url_hash(url: str) -> str:
    """Compute deterministic SHA-256 hash of normalized URL."""
    norm = normalize_url(url)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# Backwards compatibility alias
_clean_html = clean_html
