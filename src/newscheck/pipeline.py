"""Pipeline orchestrator — runs all Phase 1 stages in sequence.

Stages:
  1. Ingest:  Read RSS feeds, collect feed items
  2. Extract: Fetch full text for each article
  3. Dedup:   MinHash near-duplicates + wire-service detection
  4. Cluster: Embed → HDBSCAN → rank → select top N

The orchestrator creates a pipeline Run record, executes stages,
and finalises the run as success or failure.  Errors in individual
articles or feeds never crash the pipeline — they are logged and skipped.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from newscheck.build.alerts import send_pipeline_failure_alert
from newscheck.build.site import SiteGenerator
from newscheck.claims.extract import get_claim_extractor
from newscheck.cluster.embed import ArticleEmbedder
from newscheck.cluster.group import cluster_articles, rank_and_select
from newscheck.config import Settings, load_settings, load_sources_yaml
from newscheck.crosscheck.nli import NLICrossChecker, cross_check_cluster_claims
from newscheck.db import (
    count_distinct_sources,
    create_run,
    finish_run,
    get_all_sources,
    get_connection,
    get_recent_articles,
    init_db,
    insert_article,
    insert_claim,
    insert_cluster,
    insert_cluster_article,
    insert_cross_check,
    insert_dedup_group,
    transaction,
    upsert_source,
    upsert_story,
)
from newscheck.dedup.minhash import MinHashDeduplicator
from newscheck.dedup.wire import WireDetector
from newscheck.extract.content import ArticleExtractor
from newscheck.generate.writer import ConstrainedWriter, get_agreed_claims
from newscheck.ingest.feeds import FeedReader
from newscheck.log import get_logger
from newscheck.models import (
    Article,
    Claim,
    ClaimLabel,
    Cluster,
    ClusterArticle,
    DedupGroup,
    DedupMethod,
    ReliabilityTier,
    Run,
    RunStatus,
    Source,
    Story,
)
from newscheck.safety.gates import SafetyAuditor
from newscheck.safety.sensitive import SensitiveTopicDetector
from newscheck.scoring.formula import compute_corroboration_score

logger = get_logger(__name__)


class PipelineResult:
    """Summary of a completed pipeline run."""

    def __init__(
        self,
        run: Run,
        articles_fetched: int,
        clusters: list[Cluster],
        *,
        success: bool,
        stories_published: int = 0,
        error: str | None = None,
    ) -> None:
        self.run = run
        self.articles_fetched = articles_fetched
        self.clusters = clusters
        self.success = success
        self.stories_published = stories_published
        self.error = error

    def __repr__(self) -> str:
        return (
            f"PipelineResult(success={self.success}, "
            f"articles={self.articles_fetched}, "
            f"clusters={len(self.clusters)}, "
            f"stories={self.stories_published})"
        )


def run_pipeline(
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
    db_path: Path | None = None,
) -> PipelineResult:
    """Execute the full Phase 1 pipeline.

    Args:
        settings:     Pre-loaded settings. If None, loaded from project_root.
        project_root: Project root directory.  Defaults to cwd.
        db_path:      SQLite database path.  Defaults to ``newscheck.db``.

    Returns:
        A :class:`PipelineResult` with run metadata and top-N clusters.
    """
    if project_root is None:
        project_root = Path.cwd()
    if settings is None:
        settings = load_settings(project_root=project_root)

    conn = get_connection(db_path)
    init_db(conn)
    run = create_run(conn)
    assert run.id is not None

    try:
        result = _execute(conn, run, settings, project_root)
        finish_run(
            conn,
            run,
            status=RunStatus.SUCCESS,
            articles_fetched=result.articles_fetched,
            clusters_formed=len(result.clusters),
            stories_published=result.stories_published,
        )
        return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("pipeline_failed", run_id=run.id)
        finish_run(
            conn,
            run,
            status=RunStatus.FAILED,
            error_message=error_msg,
        )
        send_pipeline_failure_alert(settings, error_msg, run_id=run.id)
        return PipelineResult(
            run=run,
            articles_fetched=0,
            clusters=[],
            success=False,
            error=error_msg,
        )
    finally:
        conn.close()


def _execute(
    conn: sqlite3.Connection,
    run: Run,
    settings: Settings,
    project_root: Path,
) -> PipelineResult:
    """Internal pipeline logic, separated for clean error handling."""
    assert run.id is not None

    # ── Stage 1: Sync sources ──────────────────────────────────────────
    sources = _sync_sources(conn, settings, project_root)
    sources_by_id: dict[int, Source] = {s.id: s for s in sources if s.id is not None}

    # ── Stage 2: Ingest feeds ──────────────────────────────────────────
    reader = FeedReader(settings)
    feed_items = reader.read_all(sources)
    logger.info("feed_items_collected", count=len(feed_items))

    # ── Stage 3: Extract full text ─────────────────────────────────────
    extractor = ArticleExtractor(settings)
    articles_inserted = 0

    # Build a source lookup by domain for quick mapping.
    source_by_domain: dict[str, Source] = {s.domain: s for s in sources}

    for item in feed_items:
        # Resolve source_id.
        src = item.source
        if src.id is None:
            db_src = source_by_domain.get(src.domain)
            if db_src is None or db_src.id is None:
                continue
            src = db_src
        if src.id is None:
            continue

        article = extractor.extract(item, source_id=src.id, run_id=run.id)
        row_id = insert_article(conn, article)
        if row_id is not None:
            article.id = row_id
            articles_inserted += 1

    conn.commit()
    logger.info("articles_inserted", count=articles_inserted)

    # ── Stage 4: Load recent articles ──────────────────────────────────
    cutoff = datetime.now(UTC) - timedelta(hours=settings.pipeline.max_story_age_hours)
    recent_articles = get_recent_articles(conn, since=cutoff, with_text=True)
    logger.info("recent_articles_loaded", count=len(recent_articles))

    if not recent_articles:
        logger.warning("no_recent_articles")
        return PipelineResult(run=run, articles_fetched=articles_inserted, clusters=[], success=True)

    # Check minimum source diversity.
    source_count = count_distinct_sources(conn, since=cutoff)
    if source_count < settings.pipeline.min_sources_for_valid_run:
        logger.warning(
            "insufficient_source_diversity",
            sources=source_count,
            required=settings.pipeline.min_sources_for_valid_run,
        )
        # Continue anyway but log — this is a soft gate for Phase 1.

    # ── Stage 5: Wire detection ────────────────────────────────────────
    wire_detector = WireDetector()
    wire_detector.annotate_articles(recent_articles)

    # Update wire_origin in DB for newly detected wires.
    for art in recent_articles:
        if art.wire_origin and art.id is not None:
            conn.execute(
                "UPDATE articles SET wire_origin = ? WHERE id = ? AND wire_origin IS NULL",
                (art.wire_origin, art.id),
            )
    conn.commit()

    # ── Stage 6: Dedup (MinHash + Wire) ────────────────────────────────
    minhash_deduper = MinHashDeduplicator(settings.dedup)
    minhash_groups = minhash_deduper.find_duplicates(recent_articles)

    wire_groups = wire_detector.build_wire_dedup_groups(recent_articles)

    # Merge dedup groups (wire groups take precedence if overlapping).
    all_dedup: dict[int, DedupGroup] = {}
    for g in minhash_groups:
        all_dedup[g.article_id] = g
    for g in wire_groups:
        existing = all_dedup.get(g.article_id)
        if existing is not None:
            # Promote method to BOTH.
            all_dedup[g.article_id] = DedupGroup(
                article_id=g.article_id,
                canonical_id=g.canonical_id,
                similarity=max(existing.similarity, g.similarity),
                method=DedupMethod.BOTH,
            )
        else:
            all_dedup[g.article_id] = g

    with transaction(conn):
        for group in all_dedup.values():
            insert_dedup_group(conn, group)

    logger.info("dedup_complete", total_groups=len(all_dedup))

    # ── Stage 7: Embed & Cluster ───────────────────────────────────────
    embedder = ArticleEmbedder(settings.cluster)
    embeddings = embedder.embed(recent_articles)

    cluster_map = cluster_articles(embeddings, recent_articles, settings.cluster)

    if not cluster_map:
        logger.warning("no_clusters_formed")
        return PipelineResult(run=run, articles_fetched=articles_inserted, clusters=[], success=True)

    # ── Stage 8: Rank & select top N ───────────────────────────────────
    top_clusters = rank_and_select(
        cluster_map,
        recent_articles,
        sources_by_id,
        conn,
        pipeline_config=settings.pipeline,
        scoring_config=settings.scoring,
    )

    # Persist clusters and their article associations.
    with transaction(conn):
        for cluster in top_clusters:
            cluster.run_id = run.id
            cluster_id = insert_cluster(conn, cluster)
            cluster.id = cluster_id

            # Find the original article indices for this cluster.
            # Match by rank: top_clusters[i] corresponds to the i-th best cluster.
            # We need to map back from the ranked cluster to the original cluster_map.
            # The rank_and_select function processes clusters in score order,
            # so we need to reconstruct the article mapping.
            pass  # associations are persisted below

    # Persist cluster ↔ article associations.
    # Reconstruct which articles belong to which cluster.
    scored_clusters = _get_scored_order(cluster_map, recent_articles, sources_by_id, conn, settings)
    for cluster, art_indices in zip(top_clusters, scored_clusters, strict=False):
        if cluster.id is None:
            continue
        for idx in art_indices:
            art = recent_articles[idx]
            if art.id is not None:
                insert_cluster_article(conn, ClusterArticle(cluster_id=cluster.id, article_id=art.id))
    conn.commit()

    # ── Stage 9: Claim Extraction, Cross-Checking & Story Generation ───
    claim_extractor = get_claim_extractor(settings)
    nli_checker = NLICrossChecker()
    topic_detector = SensitiveTopicDetector()
    constrained_writer = ConstrainedWriter(settings.generation, settings.safety, nli_checker)
    safety_auditor = SafetyAuditor()

    articles_by_id = {a.id: a for a in recent_articles if a.id is not None}
    stories_published = 0

    for cluster, art_indices in zip(top_clusters, scored_clusters, strict=False):
        if cluster.id is None:
            continue
        cluster_articles_list = [recent_articles[idx] for idx in art_indices]

        # 1. Extract claims for each article in cluster
        cluster_claims: list[Claim] = []
        for art in cluster_articles_list:
            art_claims = claim_extractor.extract_claims(art, cluster.id)
            for c in art_claims:
                cid = insert_claim(conn, c)
                c.id = cid
                cluster_claims.append(c)

        # 2. Pairwise Cross-check across independent sources
        cross_checks = cross_check_cluster_claims(cluster_claims, articles_by_id, nli_checker)
        disputed_count = 0
        for cc in cross_checks:
            insert_cross_check(conn, cc)
            if cc.label == ClaimLabel.DISPUTED:
                disputed_count += 1

        # 3. Transparent Corroboration Scoring
        cluster_source_ids = {a.source_id for a in cluster_articles_list}
        cluster_sources = [sources_by_id[sid] for sid in cluster_source_ids if sid in sources_by_id]

        pub_dates = [a.published_at for a in cluster_articles_list if a.published_at is not None]
        oldest_pub = min(pub_dates, default=None)
        newest_pub = max(pub_dates, default=None)

        score_breakdown = compute_corroboration_score(
            cluster_sources,
            disputed_claim_count=disputed_count,
            oldest_publication=oldest_pub,
            newest_publication=newest_pub,
            config=settings.scoring,
        )

        # 4. Sensitive Topic Detection
        is_sensitive, _ = topic_detector.is_sensitive(cluster_articles_list, cluster_claims)

        # 5. Constrained Generation from Agreed Claims Only
        agreed_claims = get_agreed_claims(cluster_claims, cross_checks)
        source_passages = [a.full_text or a.snippet for a in cluster_articles_list if (a.full_text or a.snippet)]

        raw_summary = None
        if not is_sensitive and score_breakdown.independent_sources >= settings.pipeline.min_sources_to_publish:
            raw_summary = constrained_writer.generate_summary(agreed_claims, source_passages)

        # 6. Safety Audit
        story = Story(
            cluster_id=cluster.id,
            headline=cluster.label,
            summary=raw_summary,
            corroboration_score=score_breakdown.final_score,
            corroboration_label=score_breakdown.label,
            is_sensitive=is_sensitive,
            is_developing=score_breakdown.is_developing,
            published_at=newest_pub or datetime.now(UTC),
        )
        audited_story = safety_auditor.audit_story(
            story,
            independent_source_count=score_breakdown.independent_sources,
            claims=cluster_claims,
            articles=cluster_articles_list,
        )

        upsert_story(conn, audited_story)
        stories_published += 1

    conn.commit()

    # ── Stage 10: Static Site Generation (Atomic Deploy) ───────────────
    site_gen = SiteGenerator(settings, project_root)
    site_gen.build_site(conn)

    logger.info(
        "pipeline_complete",
        run_id=run.id,
        articles=articles_inserted,
        clusters=len(top_clusters),
        stories_published=stories_published,
    )

    return PipelineResult(
        run=run,
        articles_fetched=articles_inserted,
        clusters=top_clusters,
        stories_published=stories_published,
        success=True,
    )


def _sync_sources(
    conn: sqlite3.Connection,
    settings: Settings,
    project_root: Path,
) -> list[Source]:
    """Load sources from YAML, upsert into DB, and return the DB-backed list."""
    feeds_path = project_root / settings.ingest.feeds_file
    raw_sources = load_sources_yaml(feeds_path)

    for raw in raw_sources:
        tier_val = int(str(raw.get("tier", 3)))
        src = Source(
            name=str(raw["name"]),
            domain=str(raw["domain"]),
            feed_url=str(raw["feed_url"]),
            language=str(raw.get("language", "en")),
            country=str(raw.get("country", "IN")),
            tier=ReliabilityTier(tier_val),
            enabled=bool(raw.get("enabled", True)),
        )
        upsert_source(conn, src)

    return get_all_sources(conn)


def _get_scored_order(
    cluster_map: dict[int, list[int]],
    articles: list[Article],
    sources_by_id: dict[int, Source],
    conn: sqlite3.Connection,
    settings: Settings,
) -> list[list[int]]:
    """Re-derive the article-index lists in the same order as rank_and_select.

    This is needed to associate the ranked Cluster objects with their articles.
    """
    from newscheck.cluster.group import count_independent_sources, score_cluster

    articles_by_id: dict[int, Article] = {a.id: a for a in articles if a.id is not None}

    scored: list[tuple[float, int, int, list[int]]] = []
    for label, indices in cluster_map.items():
        art_ids: list[int] = [aid for i in indices if (aid := articles[i].id) is not None]
        if not art_ids:
            continue
        indep_count, unique_src_ids = count_independent_sources(art_ids, articles_by_id, conn)
        cluster_arts = [articles[i] for i in indices]
        score = score_cluster(indep_count, unique_src_ids, sources_by_id, cluster_arts, settings.scoring)
        scored.append((score, indep_count, label, indices))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [indices for _, _, _, indices in scored[: settings.pipeline.stories_per_run]]
