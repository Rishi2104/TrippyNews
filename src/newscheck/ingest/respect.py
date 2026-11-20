"""Politeness controls: robots.txt checking, rate limiting, and request helpers.

Implements RFC 9309 (Robots Exclusion Protocol):
- 4xx status (e.g. 404): robots.txt is considered absent -> allow all (cache 24h).
- 5xx status or network/TLS error: server unreachable -> disallow for this run (cache 1h).
- 403 / 429 status: mark origin blocked, disallow (cache 1h).
- 200 status: parse rules, respect Crawl-delay via RateLimiter, cache 24h.
- Strict TLS verification and honest user agent identity (never spoof browser UA).
"""

from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from newscheck.log import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Per-domain delay-based rate limiter.

    Ensures at least ``min_delay`` seconds between consecutive requests
    to the same domain. Supports dynamic Crawl-delay updates from robots.txt.
    """

    def __init__(self, min_delay: float = 1.0) -> None:
        self._default_min_delay = min_delay
        self._domain_delays: dict[str, float] = {}
        self._last_request: dict[str, float] = {}

    def set_delay(self, domain: str, delay: float) -> None:
        """Set custom delay (e.g. from robots.txt Crawl-delay) for a domain."""
        if delay > 0:
            self._domain_delays[domain] = max(self._default_min_delay, delay)
            logger.debug("rate_limit_delay_updated", domain=domain, delay=delay)

    def get_delay(self, domain: str) -> float:
        """Get effective delay for a domain."""
        return self._domain_delays.get(domain, self._default_min_delay)

    def wait(self, domain: str) -> None:
        """Block until it is safe to request *domain* again."""
        now = time.monotonic()
        last = self._last_request.get(domain, 0.0)
        delay = self.get_delay(domain)
        wait_time = delay - (now - last)
        if wait_time > 0:
            logger.debug("rate_limit_wait", domain=domain, seconds=round(wait_time, 2))
            time.sleep(wait_time)
        self._last_request[domain] = time.monotonic()

    def record(self, domain: str) -> None:
        """Record that a request was just made (without waiting)."""
        self._last_request[domain] = time.monotonic()


class RobotsChecker:
    """Caches and checks robots.txt per domain complying with RFC 9309.

    - 4xx -> Allow all (24h cache TTL)
    - 5xx / Network / TLS errors -> Disallow for run (1h cache TTL)
    - 403 / 429 -> Blocked / Disallow (1h cache TTL)
    - 200 -> Parse rules and Crawl-delay (24h cache TTL)
    """

    CACHE_TTL_SUCCESS: float = 24 * 3600.0  # 24 hours
    CACHE_TTL_ERROR: float = 3600.0         # 1 hour

    def __init__(
        self,
        user_agent: str,
        *,
        timeout: int = 10,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._limiter = rate_limiter
        # Cache entries: origin -> (RobotFileParser, expiry_time, is_allowed_fallback)
        self._cache: dict[str, tuple[RobotFileParser | None, float, bool | None]] = {}
        self.blocked_origins: set[str] = set()

    def is_allowed(self, url: str) -> bool:
        """Return True if *url* may be fetched according to robots.txt."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        domain = parsed.netloc

        now = time.monotonic()
        cached = self._cache.get(origin)
        if cached is not None:
            rp, expiry, fallback = cached
            if now < expiry:
                if fallback is not None:
                    return fallback
                if rp is not None:
                    return rp.can_fetch(self._user_agent, url)

        # Fetch or refresh robots.txt
        rp, expiry, fallback = self._fetch_robots(origin, domain)
        self._cache[origin] = (rp, expiry, fallback)

        if fallback is not None:
            return fallback
        if rp is not None:
            return rp.can_fetch(self._user_agent, url)
        return True

    def _fetch_robots(
        self,
        origin: str,
        domain: str,
    ) -> tuple[RobotFileParser | None, float, bool | None]:
        """Download and parse robots.txt according to RFC 9309 rules."""
        robots_url = f"{origin}/robots.txt"
        now = time.monotonic()

        # Strict TLS verification
        ctx = ssl.create_default_context()

        req = urllib.request.Request(
            robots_url,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "text/plain,*/*",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                status = resp.status if hasattr(resp, "status") else 200
                if status == 200:
                    content = resp.read().decode("utf-8", errors="ignore")
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.parse(content.splitlines())

                    # Check for Crawl-delay directive in robots.txt
                    if self._limiter is not None:
                        try:
                            cd = rp.crawl_delay(self._user_agent) or rp.crawl_delay("*")
                            if cd:
                                self._limiter.set_delay(domain, float(cd))
                        except Exception:
                            pass

                    logger.debug("robots_txt_parsed", origin=origin)
                    return rp, now + self.CACHE_TTL_SUCCESS, None

                if 400 <= status < 500:
                    # RFC 9309: 4xx indicates robots.txt does not exist -> allow
                    logger.debug("robots_txt_4xx_allow_all", origin=origin, status=status)
                    return None, now + self.CACHE_TTL_SUCCESS, True

                # 5xx status -> disallow for this run (cache 1h)
                logger.warning("robots_txt_5xx_disallow", origin=origin, status=status)
                return None, now + self.CACHE_TTL_ERROR, False

        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                # 403 Forbidden or 429 Too Many Requests -> mark origin blocked
                logger.warning("robots_txt_blocked", origin=origin, status=exc.code)
                self.blocked_origins.add(origin)
                return None, now + self.CACHE_TTL_ERROR, False
            if 400 <= exc.code < 500:
                # RFC 9309: other 4xx (404 Not Found, etc.) -> allow all
                logger.debug("robots_txt_not_found_allow_all", origin=origin, status=exc.code)
                return None, now + self.CACHE_TTL_SUCCESS, True
            # 5xx server errors -> disallow (RFC 9309)
            logger.warning("robots_txt_server_error_disallow", origin=origin, status=exc.code)
            return None, now + self.CACHE_TTL_ERROR, False

        except ssl.SSLError as exc:
            logger.warning("robots_txt_ssl_error_disallow", origin=origin, error=str(exc))
            return None, now + self.CACHE_TTL_ERROR, False

        except Exception as exc:
            # Network / connection error -> disallow for this run (cache 1h)
            logger.warning("robots_txt_network_error_disallow", origin=origin, error=str(exc))
            return None, now + self.CACHE_TTL_ERROR, False
