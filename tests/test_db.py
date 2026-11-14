"""Tests for the database layer."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from newscheck.db import (
    count_distinct_sources,
    create_run,
    finish_run,
    get_all_sources,
    get_canonical_id,
    get_recent_articles,
    get_source_by_domain,
    insert_article,
    insert_dedup_group,
    upsert_source,
    url_exists,
)
from newscheck.models import (
    Article,
    DedupGroup,
    DedupMethod,
    ExtractionStatus,
    ReliabilityTier,
    RunStatus,
    Source,
)


class TestSourceCRUD:
    """Tests for source upsert and query."""

    def test_upsert_and_get(self, db_conn: sqlite3.Connection) -> None:
        src = Source(
            name="Test Source",
            domain="test.example.com",
            feed_url="https://test.example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
        )
        row_id = upsert_source(db_conn, src)
        assert row_id > 0

        result = get_source_by_domain(db_conn, "test.example.com")
        assert result is not None
        assert result.name == "Test Source"
        assert result.tier == ReliabilityTier.MAJOR_OUTLET

    def test_upsert_updates_existing(self, db_conn: sqlite3.Connection) -> None:
        src = Source(
            name="Original",
            domain="dup.example.com",
            feed_url="https://dup.example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.OTHER_WHITELISTED,
        )
        upsert_source(db_conn, src)

        src.name = "Updated"
        src.tier = ReliabilityTier.WIRE_OFFICIAL
        upsert_source(db_conn, src)

        result = get_source_by_domain(db_conn, "dup.example.com")
        assert result is not None
        assert result.name == "Updated"
        assert result.tier == ReliabilityTier.WIRE_OFFICIAL

    def test_get_all_sources_enabled_only(self, db_conn: sqlite3.Connection) -> None:
        upsert_source(
            db_conn,
            Source(
                name="Enabled",
                domain="en.example.com",
                feed_url="https://en.example.com/rss",
                language="en",
                country="IN",
                tier=ReliabilityTier.MAJOR_OUTLET,
                enabled=True,
            ),
        )
        upsert_source(
            db_conn,
            Source(
                name="Disabled",
                domain="dis.example.com",
                feed_url="https://dis.example.com/rss",
                language="en",
                country="IN",
                tier=ReliabilityTier.MAJOR_OUTLET,
                enabled=False,
            ),
        )
        sources = get_all_sources(db_conn, enabled_only=True)
        assert len(sources) == 1
        assert sources[0].name == "Enabled"


class TestRunCRUD:
    """Tests for pipeline run lifecycle."""

    def test_create_and_finish_run(self, db_conn: sqlite3.Connection) -> None:
        run = create_run(db_conn)
        assert run.id is not None
        assert run.status == RunStatus.RUNNING

        finish_run(
            db_conn,
            run,
            status=RunStatus.SUCCESS,
            articles_fetched=42,
            clusters_formed=5,
        )

        row = db_conn.execute("SELECT * FROM runs WHERE id = ?", (run.id,)).fetchone()
        assert row["status"] == "success"
        assert row["articles_fetched"] == 42
        assert row["finished_at"] is not None


class TestArticleCRUD:
    """Tests for article insert and query."""

    def _insert_source(self, conn: sqlite3.Connection) -> int:
        return upsert_source(
            conn,
            Source(
                name="Src",
                domain="src.example.com",
                feed_url="https://src.example.com/rss",
                language="en",
                country="IN",
                tier=ReliabilityTier.MAJOR_OUTLET,
            ),
        )

    def test_insert_and_query(self, db_conn: sqlite3.Connection) -> None:
        src_id = self._insert_source(db_conn)
        run = create_run(db_conn)

        art = Article(
            source_id=src_id,
            url="https://src.example.com/article-1",
            title="Test Article",
            snippet="A snippet",
            full_text="Full text of the article.",
            content_hash="abc123",
            extraction_status=ExtractionStatus.FULL,
            run_id=run.id or 0,
        )
        row_id = insert_article(db_conn, art)
        db_conn.commit()
        assert row_id is not None

        # Verify it's retrievable.
        assert url_exists(db_conn, "https://src.example.com/article-1")

    def test_duplicate_url_is_skipped(self, db_conn: sqlite3.Connection) -> None:
        src_id = self._insert_source(db_conn)
        run = create_run(db_conn)

        art = Article(
            source_id=src_id,
            url="https://src.example.com/dup-article",
            title="First",
            content_hash="hash1",
            extraction_status=ExtractionStatus.FULL,
            run_id=run.id or 0,
        )
        first_id = insert_article(db_conn, art)
        db_conn.commit()
        assert first_id is not None

        art2 = Article(
            source_id=src_id,
            url="https://src.example.com/dup-article",
            title="Second (same URL)",
            content_hash="hash2",
            extraction_status=ExtractionStatus.FULL,
            run_id=run.id or 0,
        )
        second_id = insert_article(db_conn, art2)
        assert second_id is None  # skipped

    def test_get_recent_articles(self, db_conn: sqlite3.Connection) -> None:
        src_id = self._insert_source(db_conn)
        run = create_run(db_conn)
        now = datetime.now(UTC)

        art = Article(
            source_id=src_id,
            url="https://src.example.com/recent",
            title="Recent Article",
            content_hash="recent_hash",
            extraction_status=ExtractionStatus.FULL,
            fetched_at=now,
            run_id=run.id or 0,
        )
        insert_article(db_conn, art)
        db_conn.commit()

        from datetime import timedelta

        cutoff = now - timedelta(hours=1)
        recent = get_recent_articles(db_conn, since=cutoff)
        assert len(recent) == 1
        assert recent[0].title == "Recent Article"


class TestDedupCRUD:
    """Tests for dedup group storage and canonical lookup."""

    def test_canonical_id_for_ungrouped(self, db_conn: sqlite3.Connection) -> None:
        """An article not in any dedup group is its own canonical."""
        result = get_canonical_id(db_conn, 999)
        assert result == 999

    def test_insert_and_lookup(self, db_conn: sqlite3.Connection) -> None:
        # Need source and articles first.
        src_id = upsert_source(
            db_conn,
            Source(
                name="S",
                domain="s.example.com",
                feed_url="https://s.example.com/rss",
                language="en",
                country="IN",
                tier=ReliabilityTier.MAJOR_OUTLET,
            ),
        )
        run = create_run(db_conn)
        for i in range(3):
            insert_article(
                db_conn,
                Article(
                    source_id=src_id,
                    url=f"https://s.example.com/art-{i}",
                    title=f"Article {i}",
                    content_hash=f"hash_{i}",
                    extraction_status=ExtractionStatus.FULL,
                    run_id=run.id or 0,
                ),
            )
        db_conn.commit()

        # Article 2 is a duplicate of article 1.
        insert_dedup_group(
            db_conn,
            DedupGroup(
                article_id=2,
                canonical_id=1,
                similarity=0.85,
                method=DedupMethod.MINHASH,
            ),
        )
        db_conn.commit()

        assert get_canonical_id(db_conn, 2) == 1
        assert get_canonical_id(db_conn, 1) == 1  # itself


class TestSourceDiversity:
    """Tests for the source diversity check."""

    def test_count_distinct_sources(self, db_conn: sqlite3.Connection) -> None:
        now = datetime.now(UTC)

        for i, domain in enumerate(["a.com", "b.com", "c.com"]):
            src_id = upsert_source(
                db_conn,
                Source(
                    name=f"Src {i}",
                    domain=domain,
                    feed_url=f"https://{domain}/rss",
                    language="en",
                    country="IN",
                    tier=ReliabilityTier.MAJOR_OUTLET,
                ),
            )
            run = create_run(db_conn)
            insert_article(
                db_conn,
                Article(
                    source_id=src_id,
                    url=f"https://{domain}/art-1",
                    title=f"Article from {domain}",
                    content_hash=f"hash_{domain}",
                    extraction_status=ExtractionStatus.FULL,
                    fetched_at=now,
                    run_id=run.id or 0,
                ),
            )
        db_conn.commit()

        from datetime import timedelta

        count = count_distinct_sources(db_conn, since=now - timedelta(hours=1))
        assert count == 3
