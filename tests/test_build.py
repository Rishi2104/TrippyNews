"""Tests for the static site generator and error alerting."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from newscheck.build.alerts import send_pipeline_failure_alert
from newscheck.build.site import SiteGenerator
from newscheck.config import AlertingConfig, Settings
from newscheck.db import (
    create_run,
    get_connection,
    init_db,
    insert_cluster,
    upsert_source,
    upsert_story,
)
from newscheck.models import Cluster, CorroborationLabel, ReliabilityTier, Source, Story


class TestSiteGenerator:
    def test_build_site_atomic(self, project_root: Path, settings: Settings) -> None:
        db_file = project_root / "test_site.db"
        conn = get_connection(db_file)
        init_db(conn)

        run = create_run(conn)
        assert run.id is not None

        # Insert a source, cluster, and story
        src = Source(
            id=1,
            name="Source A",
            domain="a.com",
            feed_url="https://a.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
        )
        upsert_source(conn, src)

        now = datetime.now(UTC)
        cluster = Cluster(
            run_id=run.id,
            rank=1,
            label="Space Agency Discovers New Exoplanet",
            article_count=3,
            independent_source_count=3,
            tier_weighted_score=8.5,
        )
        cluster_id = insert_cluster(conn, cluster)

        story = Story(
            cluster_id=cluster_id,
            headline="Space Agency Discovers New Exoplanet",
            summary="Astronomers confirmed the discovery of an Earth-sized exoplanet in the habitable zone.",
            corroboration_score=8.5,
            corroboration_label=CorroborationLabel.STRONGLY_CORROBORATED,
            is_sensitive=False,
            is_developing=False,
            published_at=now,
        )
        story_id = upsert_story(conn, story)
        conn.commit()

        generator = SiteGenerator(settings, project_root)
        out_dir = generator.build_site(conn)
        conn.close()

        assert out_dir.exists()
        assert (out_dir / "index.html").exists()
        assert (out_dir / "methodology.html").exists()
        assert (out_dir / "corrections.html").exists()
        assert (out_dir / f"story_{story_id}.html").exists()

        index_html = (out_dir / "index.html").read_text(encoding="utf-8")
        assert "Space Agency Discovers New Exoplanet" in index_html
        assert "Strongly corroborated" in index_html

    def test_pagination_and_latest_ordering(self, project_root: Path, settings: Settings) -> None:
        from datetime import timedelta

        db_file = project_root / "test_pagination.db"
        conn = get_connection(db_file)
        init_db(conn)

        run = create_run(conn)
        assert run.id is not None

        src = Source(
            id=1,
            name="Source A",
            domain="a.com",
            feed_url="https://a.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
        )
        upsert_source(conn, src)

        base_time = datetime.now(UTC)
        # Insert 25 stories, each 1 hour apart
        for i in range(1, 26):
            cluster = Cluster(
                run_id=run.id,
                rank=i,
                label=f"Headline Story Number {i}",
                article_count=3,
                independent_source_count=3,
                tier_weighted_score=float(i),
            )
            cid = insert_cluster(conn, cluster)
            story = Story(
                cluster_id=cid,
                headline=f"Headline Story Number {i}",
                summary=f"Summary for story {i}",
                corroboration_score=float(i),
                corroboration_label=CorroborationLabel.CORROBORATED,
                is_sensitive=False,
                is_developing=False,
                published_at=base_time + timedelta(hours=i),
            )
            upsert_story(conn, story)
        conn.commit()

        generator = SiteGenerator(settings, project_root)
        out_dir = generator.build_site(conn)
        conn.close()

        # 25 stories / 10 per page = 3 pages
        assert (out_dir / "index.html").exists()
        assert (out_dir / "page_1.html").exists()
        assert (out_dir / "page_2.html").exists()
        assert (out_dir / "page_3.html").exists()

        p1_text = (out_dir / "index.html").read_text(encoding="utf-8")
        # Latest on top: story 25 has the latest published_at (base_time + 25h)
        assert "Headline Story Number 25" in p1_text
        assert "Headline Story Number 16" in p1_text
        # Story 1 is oldest, should not be on page 1
        assert "Headline Story Number 1<" not in p1_text

        # Verify pagination links exist
        assert "page_2.html" in p1_text
        assert "page_3.html" in p1_text
        assert "Page 1 of 3" in p1_text

        # Verify page 3 has oldest story (Story 1)
        p3_text = (out_dir / "page_3.html").read_text(encoding="utf-8")
        assert "Headline Story Number 1" in p3_text
        assert "Page 3 of 3" in p3_text


class TestAlerting:
    def test_alerting_disabled_does_nothing(self) -> None:
        cfg = Settings(alerting=AlertingConfig(enabled=False))
        sent = send_pipeline_failure_alert(cfg, "Some failure")
        assert sent is False

    @patch("newscheck.build.alerts._send_telegram")
    def test_alerting_telegram(self, mock_tg: MagicMock) -> None:
        mock_tg.return_value = True
        cfg = Settings(alerting=AlertingConfig(enabled=True, method="telegram"))
        sent = send_pipeline_failure_alert(cfg, "Fatal DB error", run_id=12)
        assert sent is True
        mock_tg.assert_called_once()
