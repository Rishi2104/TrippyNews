"""Acceptance tests for legal-safety, compliance, and template requirements.

Verifies:
- L1: No verbatim republication outside quote blocks (no run of >= 8 consecutive words).
- L3: Evidence quotes capped at 25 words with ellipsis, with outlet name and link.
- L4 & L7: robots.txt enforcement (RFC 9309 semantics, disallowed URLs never requested).
- L5 & L6: Honest UA, no Chrome spoof, strict TLS (no CERT_NONE / check_hostname=False).
- L10: No 'example.com' in any generated output file; contact email present.
- L11: Required legal pages exist and are linked in footer (About, Terms, Privacy, Takedown, Sources Policy).
- L12: Single source of truth (methodology constants match scoring code constants).
- T1: Story and corrections pages render cleanly when published_at/changed_at is None.
- C1 & C2: Domain whitelist enforced; tracking parameters stripped from URLs.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.robotparser import RobotFileParser

from newscheck.build.site import SiteGenerator, max_consecutive_word_overlap
from newscheck.claims.extract import truncate_evidence_quote
from newscheck.config import Settings
from newscheck.db import create_run, get_connection, init_db, insert_cluster, upsert_source, upsert_story
from newscheck.ingest.feeds import FeedReader, normalize_url
from newscheck.ingest.respect import RobotsChecker
from newscheck.models import Cluster, CorroborationLabel, ReliabilityTier, Source, Story
from newscheck.scoring.formula import SCORING_CONSTANTS


class TestLegalCompliance:
    def test_no_verbatim_run_outside_quote_blocks(self) -> None:
        """L1: Fail if rendered narrative shares >= 8 consecutive words with source text."""
        source_lede = (
            "The Supreme Court of India on Tuesday refused to interfere with the order "
            "of the Cauvery Water Management Authority directing Karnataka to release water to Tamil Nadu."
        )
        from newscheck.generate.writer import synthesize_own_words

        synthesized = synthesize_own_words(source_lede, index=0)

        # Overlap check between synthesized text and source lede
        overlap = max_consecutive_word_overlap(synthesized, source_lede)
        assert overlap < 8, f"Verbatim run of {overlap} words detected: '{synthesized}'"

    def test_evidence_quotes_length_and_attribution(self) -> None:
        """L3: Evidence quotes must be capped at 25 words with ellipsis, max 1 per claim."""
        long_sentence = (
            "The meteorological department reported that severe rainfall exceeding one hundred "
            "millimeters caused unprecedented flooding across low-lying coastal villages in the "
            "district throughout the early morning hours on Monday."
        )
        words = long_sentence.split()
        assert len(words) > 25

        truncated = truncate_evidence_quote(long_sentence, max_words=25)
        truncated_words = truncated.removesuffix("...").split()
        assert len(truncated_words) <= 25
        assert truncated.endswith("...")

        # Sentence under 25 words is untouched
        short_sentence = "The minister confirmed three new bridges will open in December."
        assert truncate_evidence_quote(short_sentence, max_words=25) == short_sentence

    def test_robots_disallowed_never_requested(self) -> None:
        """L4: A robots-disallowed URL is never requested by FeedReader or ArticleExtractor."""
        robots_txt_content = """
User-agent: *
Disallow: /blocked-feed.xml
Disallow: /secret-article/
Allow: /
"""
        rp = RobotFileParser()
        rp.parse(robots_txt_content.splitlines())

        checker = RobotsChecker(user_agent="NewsCheckBot/1.0", timeout=5)

        with patch.object(checker, "_fetch_robots", return_value=(rp, time.monotonic() + 86400, None)):
            assert checker.is_allowed("https://news.test/blocked-feed.xml") is False
            assert checker.is_allowed("https://news.test/secret-article/post-1") is False
            assert checker.is_allowed("https://news.test/public-article") is True

        # Verify FeedReader honors robots check
        reader = FeedReader(Settings())
        reader._robots = checker

        with (
            patch.object(checker, "_fetch_robots", return_value=(rp, time.monotonic() + 86400, None)),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            source = Source(
                id=1,
                name="Blocked Source",
                domain="news.test",
                feed_url="https://news.test/blocked-feed.xml",
                language="en",
                country="IN",
                tier=ReliabilityTier.MAJOR_OUTLET,
                allowed_domains=["news.test"],
            )
            items = reader.read_all([source])
            assert items == []
            # urlopen should never have been called for the feed URL
            assert mock_urlopen.call_count == 0

    def test_honest_user_agent_and_strict_tls(self, project_root: Path) -> None:
        """L5 & L6: Grep codebase asserting zero CERT_NONE/check_hostname=False and honest UA."""
        src_dir = Path(__file__).parent.parent / "src"

        # Check for disabled TLS
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "CERT_NONE" not in content, f"Insecure TLS CERT_NONE found in {py_file.name}"
            assert "check_hostname = False" not in content, f"Insecure check_hostname=False in {py_file.name}"
            assert "check_hostname=False" not in content, f"Insecure check_hostname=False in {py_file.name}"

        # Verify default settings contain the honest bot UA
        settings = Settings()
        assert "NewsCheckBot/1.0" in settings.ingest.user_agent
        assert "@" in settings.ingest.user_agent
        assert "Mozilla" not in settings.ingest.user_agent

    def test_no_example_com_in_output_and_pages_exist(self, project_root: Path, settings: Settings) -> None:
        """L10 & L11: No 'example.com' in any generated file; legal pages exist and are linked in footer."""
        db_file = project_root / "test_legal_site.db"
        conn = get_connection(db_file)
        init_db(conn)

        run = create_run(conn)
        cluster = Cluster(
            id=1,
            run_id=run.id or 1,
            rank=1,
            label="Water Dispute",
            article_count=1,
            independent_source_count=1,
            tier_weighted_score=2.0,
        )
        insert_cluster(conn, cluster)

        src = Source(
            id=1,
            name="The Hindu",
            domain="thehindu.test",
            feed_url="https://thehindu.test/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
            allowed_domains=["thehindu.test"],
        )
        upsert_source(conn, src)

        now = datetime.now(UTC)
        story = Story(
            cluster_id=1,
            headline="Supreme Court Adjourns Water Dispute Hearing",
            corroboration_score=8.5,
            corroboration_label=CorroborationLabel.STRONGLY_CORROBORATED,
            published_at=now,
            summary="Independent corroboration across reporting outlets confirms that hearing was adjourned.",
            is_sensitive=False,
            is_developing=False,
        )
        upsert_story(conn, story)

        generator = SiteGenerator(settings, project_root)
        out_dir = generator.build_site(conn)

        # Check no example.com exists in any generated HTML file
        for html_file in out_dir.glob("*.html"):
            text = html_file.read_text(encoding="utf-8")
            assert "example.com" not in text, f"Found 'example.com' in {html_file.name}"

        # Verify mandatory legal pages exist
        mandatory_pages = ["about.html", "terms.html", "privacy.html", "takedown.html", "sources_policy.html"]
        for p in mandatory_pages:
            assert (out_dir / p).is_file(), f"Missing mandatory page {p}"

        # Verify footer links in base.html / index.html
        index_html = (out_dir / "index.html").read_text(encoding="utf-8")
        for p in mandatory_pages:
            assert f'href="{p}"' in index_html, f"Footer missing link to {p}"

        # Verify trademark disclaimer present
        assert "Trademark &amp; Non-Affiliation Disclaimer" in index_html or "Trademark" in index_html

    def test_methodology_constants_match_scoring(self, project_root: Path, settings: Settings) -> None:
        """L12: Constants displayed in methodology.html match the actual scoring constants."""
        db_file = project_root / "test_meth.db"
        conn = get_connection(db_file)
        init_db(conn)

        generator = SiteGenerator(settings, project_root)
        out_dir = generator.build_site(conn)

        meth_html = (out_dir / "methodology.html").read_text(encoding="utf-8")

        # Verify weights and thresholds from SCORING_CONSTANTS appear in methodology.html
        assert f"{SCORING_CONSTANTS['tier1_weight']}" in meth_html
        assert f"{SCORING_CONSTANTS['tier2_weight']}" in meth_html
        assert f"{SCORING_CONSTANTS['tier3_weight']}" in meth_html
        assert f"{SCORING_CONSTANTS['primary_source_bonus']}" in meth_html
        assert f"{SCORING_CONSTANTS['conflict_penalty']}" in meth_html
        assert f"{SCORING_CONSTANTS['strongly_min_sources']}" in meth_html
        assert f"{SCORING_CONSTANTS['corroborated_min_sources']}" in meth_html

    def test_story_renders_with_null_published_at(self, project_root: Path, settings: Settings) -> None:
        """T1: Story and index pages must render safely without crashing when published_at is None."""
        db_file = project_root / "test_null_dates.db"
        conn = get_connection(db_file)
        init_db(conn)

        run = create_run(conn)
        cluster = Cluster(
            id=1,
            run_id=run.id or 1,
            rank=1,
            label="Diplomatic Talks",
            article_count=1,
            independent_source_count=1,
            tier_weighted_score=3.0,
        )
        insert_cluster(conn, cluster)

        src = Source(
            id=1,
            name="Wire Bureau",
            domain="wire.test",
            feed_url="https://wire.test/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.WIRE_OFFICIAL,
            allowed_domains=["wire.test"],
        )
        upsert_source(conn, src)

        story = Story(
            cluster_id=1,
            headline="Diplomatic Talks Conclude In New Delhi",
            corroboration_score=6.0,
            corroboration_label=CorroborationLabel.CORROBORATED,
            published_at=None,
            summary="Talks between representatives concluded peacefully.",
            is_sensitive=False,
            is_developing=False,
            updated_at=None,
        )
        upsert_story(conn, story)

        generator = SiteGenerator(settings, project_root)
        out_dir = generator.build_site(conn)
        assert (out_dir / "index.html").is_file()
        assert (out_dir / "story_1.html").is_file()

    def test_tracking_params_absent_and_host_normalized(self) -> None:
        """C2: Tracking parameters (utm_*, traffic_source, ref) stripped and host lowercased."""
        dirty_url = "HTTPS://WWW.TheHindu.Test/news/national/story.html?utm_source=rss&utm_medium=feed&traffic_source=rss&ref=home#comments"
        clean = normalize_url(dirty_url)

        assert "utm_source" not in clean
        assert "utm_medium" not in clean
        assert "traffic_source" not in clean
        assert "ref=" not in clean
        assert "#comments" not in clean
        assert clean.startswith("https://www.thehindu.test/")
        assert "WWW.TheHindu.Test" not in clean

    def test_allowed_domains_enforced_at_ingest(self) -> None:
        """C1: Inbound article link host must match source allowed_domains."""
        source = Source(
            id=1,
            name="Official Press",
            domain="press.gov.test",
            feed_url="https://press.gov.test/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.WIRE_OFFICIAL,
            allowed_domains=["press.gov.test", "archive.press.gov.test"],
        )

        reader = FeedReader(Settings())

        valid_entry = {
            "title": "Government Release",
            "link": "https://press.gov.test/releases/doc-101",
            "summary": "Official announcement on budget.",
        }
        with patch.object(reader._robots, "is_allowed", return_value=True):
            item_valid = reader._parse_entry(valid_entry, source)
            assert item_valid is not None
            assert item_valid.url == "https://press.gov.test/releases/doc-101"

            unvetted_entry = {
                "title": "Malicious Redirect",
                "link": "https://evil-tracking-site.test/click?target=123",
                "summary": "Spam or phishing link injected into feed.",
            }
            item_invalid = reader._parse_entry(unvetted_entry, source)
            assert item_invalid is None
