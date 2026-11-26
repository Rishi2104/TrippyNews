"""SQLite database layer for NewsCheck.

Provides schema creation, connection helpers, and typed CRUD operations.
All timestamps are stored as ISO-8601 UTC strings.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from newscheck.log import get_logger
from newscheck.models import (
    Article,
    Claim,
    Cluster,
    ClusterArticle,
    CrossCheck,
    DedupGroup,
    DedupMethod,
    ExtractionStatus,
    LicenseStatus,
    ReliabilityTier,
    Run,
    RunStatus,
    Source,
    SourceMode,
    Story,
    StoryChangelog,
)

logger = get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    domain          TEXT    NOT NULL UNIQUE,
    feed_url        TEXT    NOT NULL,
    language        TEXT    NOT NULL DEFAULT 'en',
    country         TEXT    NOT NULL DEFAULT 'IN',
    tier            INTEGER NOT NULL CHECK (tier IN (1, 2, 3)),
    enabled         INTEGER NOT NULL DEFAULT 1,
    terms_url       TEXT    NOT NULL DEFAULT '',
    license_status  TEXT    NOT NULL DEFAULT 'permitted',
    last_reviewed   TEXT    NOT NULL DEFAULT '2026-11-20',
    mode            TEXT    NOT NULL DEFAULT 'claims',
    allowed_domains TEXT    NOT NULL DEFAULT '[]',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT    NOT NULL,
    finished_at       TEXT,
    status            TEXT    NOT NULL DEFAULT 'running'
                              CHECK (status IN ('running', 'success', 'failed')),
    articles_fetched  INTEGER NOT NULL DEFAULT 0,
    clusters_formed   INTEGER NOT NULL DEFAULT 0,
    stories_published INTEGER NOT NULL DEFAULT 0,
    error_message     TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id         INTEGER NOT NULL REFERENCES sources(id),
    url               TEXT    NOT NULL UNIQUE,
    title             TEXT    NOT NULL,
    snippet           TEXT    NOT NULL DEFAULT '',
    full_text         TEXT,
    author            TEXT,
    published_at      TEXT,
    fetched_at        TEXT    NOT NULL,
    content_hash      TEXT    NOT NULL,
    extraction_status TEXT    NOT NULL DEFAULT 'full'
                              CHECK (extraction_status IN ('full', 'limited', 'failed')),
    wire_origin       TEXT,
    run_id            INTEGER NOT NULL REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_source_id    ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_run_id       ON articles(run_id);
CREATE INDEX IF NOT EXISTS idx_articles_fetched_at   ON articles(fetched_at);

CREATE TABLE IF NOT EXISTS dedup_groups (
    article_id   INTEGER NOT NULL REFERENCES articles(id),
    canonical_id INTEGER NOT NULL REFERENCES articles(id),
    similarity   REAL    NOT NULL,
    method       TEXT    NOT NULL CHECK (method IN ('minhash', 'wire', 'both')),
    PRIMARY KEY (article_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   INTEGER NOT NULL REFERENCES runs(id),
    rank                     INTEGER NOT NULL,
    label                    TEXT    NOT NULL,
    article_count            INTEGER NOT NULL DEFAULT 0,
    independent_source_count INTEGER NOT NULL DEFAULT 0,
    tier_weighted_score      REAL    NOT NULL DEFAULT 0.0,
    created_at               TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cluster_articles (
    cluster_id INTEGER NOT NULL REFERENCES clusters(id),
    article_id INTEGER NOT NULL REFERENCES articles(id),
    PRIMARY KEY (cluster_id, article_id)
);

-- Phase 2 tables (created now for forward-compatibility).
CREATE TABLE IF NOT EXISTS claims (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      INTEGER NOT NULL REFERENCES articles(id),
    cluster_id      INTEGER NOT NULL REFERENCES clusters(id),
    claim_text      TEXT    NOT NULL,
    source_sentence TEXT    NOT NULL,
    claim_type      TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cross_checks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id),
    claim_b_id INTEGER NOT NULL REFERENCES claims(id),
    label      TEXT    NOT NULL CHECK (label IN ('agreed', 'disputed', 'single_source')),
    nli_score  REAL    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stories (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id          INTEGER NOT NULL REFERENCES clusters(id),
    headline            TEXT    NOT NULL,
    summary             TEXT,
    corroboration_score REAL    NOT NULL DEFAULT 0.0,
    corroboration_label TEXT    NOT NULL DEFAULT 'Unconfirmed',
    is_sensitive        INTEGER NOT NULL DEFAULT 0,
    is_developing       INTEGER NOT NULL DEFAULT 0,
    published_at        TEXT,
    updated_at          TEXT,
    content_hash        TEXT
);

CREATE TABLE IF NOT EXISTS story_changelog (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id      INTEGER NOT NULL REFERENCES stories(id),
    changed_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    field_changed TEXT    NOT NULL,
    old_value     TEXT,
    new_value     TEXT,
    reason        TEXT
);
"""

DEFAULT_DB_PATH = Path("newscheck.db")


def _dt_to_iso(dt: datetime | None) -> str | None:
    """Convert a datetime to an ISO-8601 UTC string, or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _iso_to_dt(s: str | None) -> datetime | None:
    """Parse an ISO-8601 string back to a UTC datetime, or None."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection.

    Args:
        db_path: Path to the database file.  Defaults to ``newscheck.db``
                 in the current working directory.

    Returns:
        A :class:`sqlite3.Connection` with WAL mode and foreign keys enabled.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not already exist, and run schema migrations.

    Safe to call multiple times (idempotent).
    """
    conn.executescript(_SCHEMA_SQL)
    # Check if sources table has new columns for L8 / C1
    cur = conn.execute("PRAGMA table_info(sources)")
    existing_cols = {row["name"] for row in cur.fetchall()}
    new_cols = {
        "terms_url": "TEXT NOT NULL DEFAULT ''",
        "license_status": "TEXT NOT NULL DEFAULT 'permitted'",
        "last_reviewed": "TEXT NOT NULL DEFAULT '2026-11-20'",
        "mode": "TEXT NOT NULL DEFAULT 'claims'",
        "allowed_domains": "TEXT NOT NULL DEFAULT '[]'",
    }
    for col_name, col_def in new_cols.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE sources ADD COLUMN {col_name} {col_def}")
    conn.commit()
    logger.info("database_initialised")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that commits on success and rolls back on error."""
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


# ── Source CRUD ────────────────────────────────────────────────────────────────


def upsert_source(conn: sqlite3.Connection, src: Source) -> int:
    """Insert or update a source, returning its row id.

    Uses the ``domain`` as the natural key for upsert.
    """
    allowed_json = json.dumps(src.allowed_domains)
    cur = conn.execute(
        """
        INSERT INTO sources (name, domain, feed_url, language, country, tier, enabled,
                             terms_url, license_status, last_reviewed, mode, allowed_domains)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(domain) DO UPDATE SET
            name            = excluded.name,
            feed_url        = excluded.feed_url,
            language        = excluded.language,
            country         = excluded.country,
            tier            = excluded.tier,
            enabled         = excluded.enabled,
            terms_url       = excluded.terms_url,
            license_status  = excluded.license_status,
            last_reviewed   = excluded.last_reviewed,
            mode            = excluded.mode,
            allowed_domains = excluded.allowed_domains
        """,
        (
            src.name,
            src.domain,
            src.feed_url,
            src.language,
            src.country,
            int(src.tier),
            src.enabled,
            src.terms_url,
            str(src.license_status),
            src.last_reviewed,
            str(src.mode),
            allowed_json,
        ),
    )
    conn.commit()
    return cur.lastrowid or 0


def _row_to_source(row: sqlite3.Row) -> Source:
    d = dict(row)
    allowed_raw = d.get("allowed_domains", "[]")
    try:
        allowed = json.loads(allowed_raw) if allowed_raw else []
    except Exception:
        allowed = []

    license_str = d.get("license_status", "permitted")
    mode_str = d.get("mode", "claims")

    return Source(
        id=d["id"],
        name=d["name"],
        domain=d["domain"],
        feed_url=d["feed_url"],
        language=d["language"],
        country=d["country"],
        tier=ReliabilityTier(d["tier"]),
        enabled=bool(d["enabled"]),
        terms_url=d.get("terms_url", ""),
        license_status=LicenseStatus(license_str) if license_str in LicenseStatus else LicenseStatus.PERMITTED,
        last_reviewed=d.get("last_reviewed", "2026-11-20"),
        mode=SourceMode(mode_str) if mode_str in SourceMode else SourceMode.CLAIMS,
        allowed_domains=allowed,
    )


def get_all_sources(conn: sqlite3.Connection, *, enabled_only: bool = True) -> list[Source]:
    """Return all sources, optionally filtering to enabled ones."""
    sql = "SELECT * FROM sources"
    if enabled_only:
        sql += " WHERE enabled = 1"
    rows = conn.execute(sql).fetchall()
    return [_row_to_source(row) for row in rows]


def get_source_by_domain(conn: sqlite3.Connection, domain: str) -> Source | None:
    """Look up a source by its domain name."""
    row = conn.execute("SELECT * FROM sources WHERE domain = ?", (domain,)).fetchone()
    if row is None:
        return None
    return _row_to_source(row)


# ── Run CRUD ───────────────────────────────────────────────────────────────────


def create_run(conn: sqlite3.Connection) -> Run:
    """Insert a new pipeline run and return it with its id."""
    now = datetime.now(UTC)
    cur = conn.execute(
        "INSERT INTO runs (started_at, status) VALUES (?, ?)",
        (_dt_to_iso(now), RunStatus.RUNNING.value),
    )
    conn.commit()
    return Run(id=cur.lastrowid, started_at=now, status=RunStatus.RUNNING)


def finish_run(
    conn: sqlite3.Connection,
    run: Run,
    *,
    status: RunStatus,
    articles_fetched: int = 0,
    clusters_formed: int = 0,
    stories_published: int = 0,
    error_message: str | None = None,
) -> None:
    """Mark a run as finished (success or failed)."""
    now = datetime.now(UTC)
    conn.execute(
        """
        UPDATE runs
           SET finished_at       = ?,
               status            = ?,
               articles_fetched  = ?,
               clusters_formed   = ?,
               stories_published = ?,
               error_message     = ?
         WHERE id = ?
        """,
        (
            _dt_to_iso(now),
            status.value,
            articles_fetched,
            clusters_formed,
            stories_published,
            error_message,
            run.id,
        ),
    )
    conn.commit()


# ── Article CRUD ───────────────────────────────────────────────────────────────


def insert_article(conn: sqlite3.Connection, art: Article) -> int | None:
    """Insert an article. Returns the row id, or None if the URL already exists.

    Idempotent: duplicate URLs are silently skipped.
    """
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO articles
            (source_id, url, title, snippet, full_text, author,
             published_at, fetched_at, content_hash, extraction_status,
             wire_origin, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            art.source_id,
            art.url,
            art.title,
            art.snippet,
            art.full_text,
            art.author,
            _dt_to_iso(art.published_at),
            _dt_to_iso(art.fetched_at),
            art.content_hash,
            art.extraction_status.value,
            art.wire_origin,
            art.run_id,
        ),
    )
    if cur.rowcount == 0:
        return None  # duplicate URL
    return cur.lastrowid


def get_recent_articles(
    conn: sqlite3.Connection,
    *,
    since: datetime,
    with_text: bool = False,
) -> list[Article]:
    """Return articles fetched on or after *since*.

    Args:
        since: UTC cutoff datetime.
        with_text: If False, ``full_text`` is omitted (saves memory).
    """
    cols = (
        "id, source_id, url, title, snippet, author, published_at, "
        "fetched_at, content_hash, extraction_status, wire_origin, run_id"
    )
    if with_text:
        cols = "*"

    rows = conn.execute(
        f"SELECT {cols} FROM articles WHERE fetched_at >= ? ORDER BY fetched_at DESC",  # noqa: S608
        (_dt_to_iso(since),),
    ).fetchall()
    return [_row_to_article(row, with_text=with_text) for row in rows]


def get_articles_by_ids(
    conn: sqlite3.Connection,
    article_ids: Sequence[int],
    *,
    with_text: bool = False,
) -> list[Article]:
    """Return articles matching the given ids."""
    if not article_ids:
        return []
    placeholders = ",".join("?" for _ in article_ids)
    cols = (
        "*"
        if with_text
        else (
            "id, source_id, url, title, snippet, author, published_at, "
            "fetched_at, content_hash, extraction_status, wire_origin, run_id"
        )
    )
    rows = conn.execute(
        f"SELECT {cols} FROM articles WHERE id IN ({placeholders})",  # noqa: S608
        list(article_ids),
    ).fetchall()
    return [_row_to_article(row, with_text=with_text) for row in rows]


def url_exists(conn: sqlite3.Connection, url: str) -> bool:
    """Check whether an article URL is already stored."""
    row = conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone()
    return row is not None


def _row_to_article(row: sqlite3.Row, *, with_text: bool = False) -> Article:
    """Map a database row to an :class:`Article`."""
    return Article(
        id=row["id"],
        source_id=row["source_id"],
        url=row["url"],
        title=row["title"],
        snippet=row["snippet"],
        full_text=row["full_text"] if with_text else None,
        author=row["author"],
        published_at=_iso_to_dt(row["published_at"]),
        fetched_at=_iso_to_dt(row["fetched_at"]) or datetime.now(UTC),
        content_hash=row["content_hash"],
        extraction_status=ExtractionStatus(row["extraction_status"]),
        wire_origin=row["wire_origin"],
        run_id=row["run_id"],
    )


# ── Dedup CRUD ─────────────────────────────────────────────────────────────────


def insert_dedup_group(conn: sqlite3.Connection, group: DedupGroup) -> None:
    """Record a dedup relationship. Idempotent (REPLACE)."""
    conn.execute(
        """
        INSERT OR REPLACE INTO dedup_groups (article_id, canonical_id, similarity, method)
        VALUES (?, ?, ?, ?)
        """,
        (group.article_id, group.canonical_id, group.similarity, group.method.value),
    )


def get_canonical_id(conn: sqlite3.Connection, article_id: int) -> int:
    """Return the canonical article id for a given article.

    If the article is not in any dedup group, returns itself.
    """
    row = conn.execute(
        "SELECT canonical_id FROM dedup_groups WHERE article_id = ?",
        (article_id,),
    ).fetchone()
    if row is None:
        return article_id
    return int(row["canonical_id"])


def get_all_dedup_groups(conn: sqlite3.Connection) -> list[DedupGroup]:
    """Return all dedup group entries."""
    rows = conn.execute("SELECT * FROM dedup_groups").fetchall()
    return [
        DedupGroup(
            article_id=row["article_id"],
            canonical_id=row["canonical_id"],
            similarity=row["similarity"],
            method=DedupMethod(row["method"]),
        )
        for row in rows
    ]


# ── Cluster CRUD ───────────────────────────────────────────────────────────────


def insert_cluster(conn: sqlite3.Connection, cluster: Cluster) -> int:
    """Insert a cluster and return its id."""
    cur = conn.execute(
        """
        INSERT INTO clusters
            (run_id, rank, label, article_count, independent_source_count,
             tier_weighted_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cluster.run_id,
            cluster.rank,
            cluster.label,
            cluster.article_count,
            cluster.independent_source_count,
            cluster.tier_weighted_score,
            _dt_to_iso(cluster.created_at),
        ),
    )
    return cur.lastrowid or 0


def insert_cluster_article(conn: sqlite3.Connection, ca: ClusterArticle) -> None:
    """Associate an article with a cluster."""
    conn.execute(
        "INSERT OR IGNORE INTO cluster_articles (cluster_id, article_id) VALUES (?, ?)",
        (ca.cluster_id, ca.article_id),
    )


def get_clusters_for_run(conn: sqlite3.Connection, run_id: int) -> list[Cluster]:
    """Return all clusters for a given run, ordered by rank."""
    rows = conn.execute(
        "SELECT * FROM clusters WHERE run_id = ? ORDER BY rank ASC",
        (run_id,),
    ).fetchall()
    return [
        Cluster(
            id=row["id"],
            run_id=row["run_id"],
            rank=row["rank"],
            label=row["label"],
            article_count=row["article_count"],
            independent_source_count=row["independent_source_count"],
            tier_weighted_score=row["tier_weighted_score"],
            created_at=_iso_to_dt(row["created_at"]) or datetime.now(UTC),
        )
        for row in rows
    ]


def get_articles_for_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    *,
    with_text: bool = False,
) -> list[Article]:
    """Return all articles belonging to a cluster."""
    cols = (
        "*"
        if with_text
        else (
            "a.id, a.source_id, a.url, a.title, a.snippet, a.author, "
            "a.published_at, a.fetched_at, a.content_hash, a.extraction_status, "
            "a.wire_origin, a.run_id"
        )
    )
    rows = conn.execute(
        f"""
        SELECT {cols}
          FROM articles a
          JOIN cluster_articles ca ON ca.article_id = a.id
         WHERE ca.cluster_id = ?
        """,  # noqa: S608
        (cluster_id,),
    ).fetchall()
    return [_row_to_article(row, with_text=with_text) for row in rows]


def count_distinct_sources(conn: sqlite3.Connection, *, since: datetime) -> int:
    """Count how many distinct sources contributed articles since *since*."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT source_id) AS cnt FROM articles WHERE fetched_at >= ?",
        (_dt_to_iso(since),),
    ).fetchone()
    return int(row["cnt"]) if row else 0


# ── Claims & Cross-Checks CRUD ─────────────────────────────────────────────────


def insert_claim(conn: sqlite3.Connection, claim: Claim) -> int:
    """Insert a claim and return its id."""
    cur = conn.execute(
        """
        INSERT INTO claims (article_id, cluster_id, claim_text, source_sentence, claim_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            claim.article_id,
            claim.cluster_id,
            claim.claim_text,
            claim.source_sentence,
            claim.claim_type,
            _dt_to_iso(claim.created_at),
        ),
    )
    return cur.lastrowid or 0


def insert_cross_check(conn: sqlite3.Connection, cc: CrossCheck) -> int:
    """Insert a cross check comparison."""
    cur = conn.execute(
        """
        INSERT INTO cross_checks (claim_a_id, claim_b_id, label, nli_score, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            cc.claim_a_id,
            cc.claim_b_id,
            cc.label.value,
            cc.nli_score,
            _dt_to_iso(cc.created_at),
        ),
    )
    return cur.lastrowid or 0


# ── Story CRUD & Changelog ─────────────────────────────────────────────────────


def upsert_story(conn: sqlite3.Connection, story: Story) -> int:
    """Insert or update a published story, generating changelog entries on edits."""
    now = datetime.now(UTC)
    existing = conn.execute(
        "SELECT * FROM stories WHERE cluster_id = ?",
        (story.cluster_id,),
    ).fetchone()

    if existing is None:
        cur = conn.execute(
            """
            INSERT INTO stories
                (cluster_id, headline, summary, corroboration_score,
                 corroboration_label, is_sensitive, is_developing,
                 published_at, updated_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                story.cluster_id,
                story.headline,
                story.summary,
                story.corroboration_score,
                story.corroboration_label.value,
                int(story.is_sensitive),
                int(story.is_developing),
                _dt_to_iso(story.published_at),
                _dt_to_iso(now),
                story.content_hash,
            ),
        )
        return cur.lastrowid or 0

    story_id = existing["id"]
    # Check for changes and log them
    if existing["corroboration_label"] != story.corroboration_label.value:
        insert_story_changelog(
            conn,
            StoryChangelog(
                story_id=story_id,
                field_changed="corroboration_label",
                old_value=existing["corroboration_label"],
                new_value=story.corroboration_label.value,
                reason="Corroboration level updated from new reporting",
            ),
        )

    if existing["headline"] != story.headline:
        insert_story_changelog(
            conn,
            StoryChangelog(
                story_id=story_id,
                field_changed="headline",
                old_value=existing["headline"],
                new_value=story.headline,
                reason="Headline refined with new corroborating sources",
            ),
        )

    conn.execute(
        """
        UPDATE stories
           SET headline            = ?,
               summary             = ?,
               corroboration_score = ?,
               corroboration_label = ?,
               is_sensitive        = ?,
               is_developing       = ?,
               updated_at          = ?,
               content_hash        = ?
         WHERE id = ?
        """,
        (
            story.headline,
            story.summary,
            story.corroboration_score,
            story.corroboration_label.value,
            int(story.is_sensitive),
            int(story.is_developing),
            _dt_to_iso(now),
            story.content_hash,
            story_id,
        ),
    )
    return int(story_id)


def insert_story_changelog(conn: sqlite3.Connection, cl: StoryChangelog) -> int:
    """Insert an entry into the public story changelog."""
    cur = conn.execute(
        """
        INSERT INTO story_changelog (story_id, changed_at, field_changed, old_value, new_value, reason)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            cl.story_id,
            _dt_to_iso(cl.changed_at),
            cl.field_changed,
            cl.old_value,
            cl.new_value,
            cl.reason,
        ),
    )
    return cur.lastrowid or 0


def prune_old_data(conn: sqlite3.Connection, *, retention_days: int = 5) -> int:
    """Purge stale records older than `retention_days` and compact the database.

    Deletes historic runs, articles, clusters, claims, cross-checks, and stories
    past the retention window to keep database size healthy under git file size limits.
    """
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    logger.info("pruning_old_data", cutoff=cutoff, retention_days=retention_days)

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            """
            DELETE FROM cross_checks WHERE claim_a_id IN (
                SELECT id FROM claims WHERE created_at < ?
            ) OR claim_b_id IN (
                SELECT id FROM claims WHERE created_at < ?
            )
            """,
            (cutoff, cutoff),
        )
        conn.execute("DELETE FROM claims WHERE created_at < ?", (cutoff,))
        conn.execute(
            "DELETE FROM cluster_articles WHERE cluster_id IN (SELECT id FROM clusters WHERE created_at < ?)",
            (cutoff,),
        )
        conn.execute("DELETE FROM story_changelog WHERE story_id IN (SELECT id FROM stories WHERE published_at < ?)", (cutoff,))
        conn.execute("DELETE FROM stories WHERE published_at < ?", (cutoff,))
        conn.execute("DELETE FROM clusters WHERE created_at < ?", (cutoff,))
        conn.execute(
            "DELETE FROM dedup_groups WHERE article_id IN (SELECT id FROM articles WHERE published_at < ?)",
            (cutoff,),
        )
        cur = conn.execute("DELETE FROM articles WHERE published_at < ?", (cutoff,))
        deleted_count = cur.rowcount
        conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    logger.info("prune_complete", deleted_articles=deleted_count)
    return deleted_count

