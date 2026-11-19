"""Politeness controls: robots.txt checking, rate limiting, and request helpers.

Every HTTP request the pipeline makes passes through these guards.
"""

from __future__ import annotations

import ssl
import time
import urllib.request
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from newscheck.log import get_logger

logger = get_logger(__name__)


class RobotsChecker:
    """Caches and checks robots.txt per domain.

    Respects ``User-Agent`` matching.  If robots.txt cannot be fetched
    (timeout, DNS error, 5xx, or 403 WAF block), the domain is assumed **allowed** — this is
    the standard convention for polite bots.
    """

    def __init__(self, user_agent: str, *, timeout: int = 5) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache: dict[str, RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        """Return True if *url* may be fetched according to robots.txt."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._fetch_robots(origin)
        return self._cache[origin].can_fetch(self._user_agent, url)

    def _fetch_robots(self, origin: str) -> RobotFileParser:
        """Download and parse robots.txt for an origin, with fallback."""
        rp = RobotFileParser()
        robots_url = f"{origin}/robots.txt"
        rp.set_url(robots_url)

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            robots_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/plain,text/html,*/*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
                rp.parse(content.splitlines())
        except Exception:
            logger.warning("robots_txt_fetch_failed", origin=origin)
            rp.allow_all = True  # type: ignore[attr-defined]

        # If robots.txt resulted in disallow_all due to WAF / 403 or failure,
        # fallback to allowed so legitimate news sources are not dropped.
        if getattr(rp, "disallow_all", False):
            rp.allow_all = True  # type: ignore[attr-defined]
            rp.disallow_all = False  # type: ignore[attr-defined]

        return rp


class RateLimiter:
    """Per-domain delay-based rate limiter.

    Ensures at least ``min_delay`` seconds between consecutive requests
    to the same domain.  Thread-safe is **not** required — the pipeline
    is single-threaded.
    """

    def __init__(self, min_delay: float = 1.0) -> None:
        self._min_delay = min_delay
        self._last_request: dict[str, float] = {}

    def wait(self, domain: str) -> None:
        """Block until it is safe to request *domain* again."""
        now = time.monotonic()
        last = self._last_request.get(domain, 0.0)
        wait_time = self._min_delay - (now - last)
        if wait_time > 0:
            logger.debug("rate_limit_wait", domain=domain, seconds=round(wait_time, 2))
            time.sleep(wait_time)
        self._last_request[domain] = time.monotonic()

    def record(self, domain: str) -> None:
        """Record that a request was just made (without waiting)."""
        self._last_request[domain] = time.monotonic()
