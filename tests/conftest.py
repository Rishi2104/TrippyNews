"""Shared test fixtures for the NewsCheck test suite."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from newscheck.config import Settings, load_settings
from newscheck.db import get_connection, init_db
from newscheck.models import (
    Article,
    ExtractionStatus,
    ReliabilityTier,
    Source,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project root with settings.toml and sources.yaml."""
    settings_content = """
[pipeline]
stories_per_run = 5
min_sources_to_publish = 2
min_sources_for_valid_run = 2
max_story_age_hours = 48

[ingest]
request_timeout_seconds = 5
max_retries = 0
rate_limit_per_domain_seconds = 0.0
max_articles_per_feed = 10
feeds_file = "sources.yaml"

[dedup]
minhash_threshold = 0.55
minhash_num_perm = 64
minhash_shingle_size = 5

[cluster]
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
min_cluster_size = 2
min_samples = 1
"""
    (tmp_path / "settings.toml").write_text(settings_content, encoding="utf-8")

    sources_content = """
sources:
  - name: "Test Source A"
    domain: "source-a.example.com"
    feed_url: "https://source-a.example.com/rss"
    language: "en"
    country: "IN"
    tier: 1

  - name: "Test Source B"
    domain: "source-b.example.com"
    feed_url: "https://source-b.example.com/rss"
    language: "en"
    country: "IN"
    tier: 2

  - name: "Test Source C"
    domain: "source-c.example.com"
    feed_url: "https://source-c.example.com/rss"
    language: "en"
    country: "IN"
    tier: 3
"""
    (tmp_path / "sources.yaml").write_text(sources_content, encoding="utf-8")

    src_root = Path(__file__).parent.parent
    if (src_root / "templates").exists():
        shutil.copytree(src_root / "templates", tmp_path / "templates")
    if (src_root / "static").exists():
        shutil.copytree(src_root / "static", tmp_path / "static")

    return tmp_path


@pytest.fixture
def settings(project_root: Path) -> Settings:
    """Load test settings from the temporary project root."""
    return load_settings(project_root=project_root)


@pytest.fixture
def db_conn(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Create an in-memory database with the full schema."""
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_sources() -> list[Source]:
    """Return a small list of test sources."""
    return [
        Source(
            id=1,
            name="Wire Agency",
            domain="wire.example.com",
            feed_url="https://wire.example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.WIRE_OFFICIAL,
        ),
        Source(
            id=2,
            name="Major Paper",
            domain="major.example.com",
            feed_url="https://major.example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.MAJOR_OUTLET,
        ),
        Source(
            id=3,
            name="Small Outlet",
            domain="small.example.com",
            feed_url="https://small.example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.OTHER_WHITELISTED,
        ),
    ]


@pytest.fixture
def sample_articles() -> list[Article]:
    """Return a set of test articles for dedup and clustering tests."""
    now = datetime.now(UTC)
    return [
        Article(
            id=1,
            source_id=1,
            url="https://wire.example.com/earthquake-1",
            title="Major earthquake strikes coastal region",
            snippet="A 6.5-magnitude earthquake struck the coast early morning.",
            full_text=(
                "A 6.5-magnitude earthquake struck the coastal region early this morning, "
                "causing widespread damage. Rescue teams have been deployed to the affected areas. "
                "The national disaster management authority confirmed the magnitude and said "
                "aftershocks are expected. At least 50 buildings have been damaged. "
                "Officials said there are no confirmed casualties so far."
            ),
            author="PTI",
            published_at=now,
            fetched_at=now,
            content_hash="hash_eq_1",
            extraction_status=ExtractionStatus.FULL,
            wire_origin="PTI",
            run_id=1,
        ),
        Article(
            id=2,
            source_id=2,
            url="https://major.example.com/earthquake-report",
            title="Earthquake rocks coastal area, rescue ops begin",
            snippet="A powerful earthquake shook the coastal district today.",
            full_text=(
                "A powerful earthquake measuring 6.5 on the Richter scale rocked the coastal "
                "district in the early hours today. Search and rescue operations are underway. "
                "The disaster management authority has issued an advisory. Multiple buildings "
                "sustained damage. No casualties have been reported yet according to officials."
            ),
            author="Staff Reporter",
            published_at=now,
            fetched_at=now,
            content_hash="hash_eq_2",
            extraction_status=ExtractionStatus.FULL,
            wire_origin=None,
            run_id=1,
        ),
        Article(
            id=3,
            source_id=3,
            url="https://small.example.com/quake-update",
            title="6.5 earthquake hits coast, NDMA deploys teams",
            snippet="Earthquake of 6.5 magnitude hits coastal region.",
            full_text=(
                "An earthquake of 6.5 magnitude hit the coastal region early morning today. "
                "The NDMA has deployed rescue teams. At least 50 buildings are damaged. "
                "Officials confirmed no casualties so far. Aftershocks expected."
            ),
            author="PTI",
            published_at=now,
            fetched_at=now,
            content_hash="hash_eq_3",
            extraction_status=ExtractionStatus.FULL,
            wire_origin="PTI",
            run_id=1,
        ),
        Article(
            id=4,
            source_id=1,
            url="https://wire.example.com/markets-rally",
            title="Stock markets surge on GDP data",
            snippet="BSE Sensex gains 500 points on positive GDP numbers.",
            full_text=(
                "Indian stock markets surged on Friday after the release of positive GDP growth "
                "data. The BSE Sensex gained over 500 points. The Nifty 50 crossed the 20,000 "
                "mark for the first time. Analysts said the rally was broad-based."
            ),
            author=None,
            published_at=now,
            fetched_at=now,
            content_hash="hash_mkt_1",
            extraction_status=ExtractionStatus.FULL,
            wire_origin=None,
            run_id=1,
        ),
        Article(
            id=5,
            source_id=2,
            url="https://major.example.com/sensex-rally",
            title="Sensex jumps 500 points as GDP growth beats expectations",
            snippet="Markets rallied after strong GDP data was released.",
            full_text=(
                "The BSE Sensex jumped over 500 points on Friday, driven by better-than-expected "
                "GDP growth figures. The Nifty 50 index also hit a record high, crossing 20,000 "
                "for the first time. Market analysts attributed the rally to strong economic data."
            ),
            author=None,
            published_at=now,
            fetched_at=now,
            content_hash="hash_mkt_2",
            extraction_status=ExtractionStatus.FULL,
            wire_origin=None,
            run_id=1,
        ),
        Article(
            id=6,
            source_id=3,
            url="https://small.example.com/cricket-win",
            title="India wins cricket series in thrilling finish",
            snippet="India clinched the series with a last-ball six.",
            full_text=(
                "India clinched the cricket series with a dramatic last-ball victory in the "
                "deciding match. The team chased down 320 runs with captain scoring a century. "
                "Fans celebrated across the country."
            ),
            author="Sports Desk",
            published_at=now,
            fetched_at=now,
            content_hash="hash_cric_1",
            extraction_status=ExtractionStatus.FULL,
            wire_origin=None,
            run_id=1,
        ),
    ]
