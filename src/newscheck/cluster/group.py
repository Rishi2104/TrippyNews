"""HDBSCAN clustering and top-10 story ranking.

Groups article embeddings into event clusters, counts independent sources
(after dedup), scores each cluster, and selects the top N.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN

from newscheck.config import ClusterConfig, PipelineConfig, ScoringConfig
from newscheck.db import get_canonical_id
from newscheck.log import get_logger
from newscheck.models import (
    Article,
    Cluster,
    ReliabilityTier,
    Source,
)

logger = get_logger(__name__)


def cluster_articles(
    embeddings: NDArray[np.float32],
    articles: list[Article],
    config: ClusterConfig,
) -> dict[int, list[int]]:
    """Run HDBSCAN on embeddings and return cluster-label → article-index mapping.

    Articles assigned to noise (label -1) are excluded.

    Args:
        embeddings: 2-D array of shape ``(N, dim)``.
        articles: Corresponding list of articles (same order as rows).
        config: Clustering hyper-parameters.

    Returns:
        Dict mapping cluster label (int ≥ 0) to list of article indices.
    """
    if len(articles) < config.min_cluster_size:
        logger.warning(
            "too_few_articles_to_cluster",
            count=len(articles),
            min_cluster_size=config.min_cluster_size,
        )
        return {}

    clusterer = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        metric=config.metric,
    )

    labels: NDArray[np.intp] = clusterer.fit_predict(embeddings)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        label_int = int(label)
        if label_int == -1:
            continue  # noise
        clusters.setdefault(label_int, []).append(idx)

    logger.info(
        "hdbscan_complete",
        total_articles=len(articles),
        clusters_found=len(clusters),
        noise_articles=int(np.sum(labels == -1)),
    )
    return clusters


def count_independent_sources(
    article_ids: list[int],
    articles_by_id: dict[int, Article],
    conn: sqlite3.Connection,
) -> tuple[int, set[int]]:
    """Count independent sources for a set of articles, respecting dedup.

    Two articles from the same source, or articles that are dedup copies
    of each other, count as ONE source.

    Args:
        article_ids: Article ids in this cluster.
        articles_by_id: Lookup dict for article objects.
        conn: Database connection (for dedup lookups).

    Returns:
        Tuple of (independent_count, set_of_canonical_source_ids).
    """
    # Map each article to its canonical article (dedup-aware).
    canonical_sources: set[tuple[int, int]] = set()
    for aid in article_ids:
        canonical_aid = get_canonical_id(conn, aid)
        art = articles_by_id.get(canonical_aid) or articles_by_id.get(aid)
        if art is not None:
            canonical_sources.add((art.source_id, canonical_aid))

    # Unique source_ids among canonical articles.
    unique_sources = {src_id for src_id, _ in canonical_sources}
    return len(unique_sources), unique_sources


def score_cluster(
    independent_count: int,
    source_ids: set[int],
    sources_by_id: dict[int, Source],
    articles: list[Article],
    scoring_config: ScoringConfig,
) -> float:
    """Compute a tier-weighted score for a cluster.

    The score is:
        sum(tier_weight[source.tier] for unique source)
        + primary_source_bonus (if any tier-1 source is present)
        × recency_factor

    Args:
        independent_count: Number of independent sources.
        source_ids: Set of unique source ids.
        sources_by_id: Source lookup dict.
        articles: Articles in this cluster (for recency).
        scoring_config: Weight configuration.

    Returns:
        A float score (higher = more corroborated / newsworthy).
    """
    tier_weights = {
        ReliabilityTier.WIRE_OFFICIAL: scoring_config.tier1_weight,
        ReliabilityTier.MAJOR_OUTLET: scoring_config.tier2_weight,
        ReliabilityTier.OTHER_WHITELISTED: scoring_config.tier3_weight,
    }

    score = 0.0
    has_tier1 = False
    for sid in source_ids:
        src = sources_by_id.get(sid)
        if src is not None:
            score += tier_weights.get(src.tier, 1.0)
            if src.tier == ReliabilityTier.WIRE_OFFICIAL:
                has_tier1 = True

    if has_tier1:
        score += scoring_config.primary_source_bonus

    # Recency: boost recent stories.
    now = datetime.now(UTC)
    newest = _newest_published(articles)
    if newest is not None:
        age_hours = (now - newest).total_seconds() / 3600.0
        decay = max(0.1, 1.0 - (age_hours / (scoring_config.recency_decay_hours * 8)))
        score *= decay

    return round(score, 4)


def rank_and_select(
    cluster_map: dict[int, list[int]],
    articles: list[Article],
    sources_by_id: dict[int, Source],
    conn: sqlite3.Connection,
    *,
    pipeline_config: PipelineConfig,
    scoring_config: ScoringConfig,
) -> list[Cluster]:
    """Score all clusters, rank them, and return the top N as Cluster objects.

    Args:
        cluster_map: HDBSCAN label → list of article indices.
        articles: All articles (same order as embeddings).
        sources_by_id: Source lookup.
        conn: DB connection for dedup lookups.
        pipeline_config: For ``stories_per_run``.
        scoring_config: For tier weights.

    Returns:
        List of up to ``stories_per_run`` :class:`Cluster` objects,
        ordered by rank (1 = best).
    """
    articles_by_id: dict[int, Article] = {a.id: a for a in articles if a.id is not None}

    scored: list[tuple[int, float, int, set[int], list[int]]] = []
    for label, indices in cluster_map.items():
        art_ids: list[int] = [aid for i in indices if (aid := articles[i].id) is not None]
        if not art_ids:
            continue

        indep_count, unique_src_ids = count_independent_sources(art_ids, articles_by_id, conn)
        cluster_articles_list = [articles[i] for i in indices]
        score = score_cluster(indep_count, unique_src_ids, sources_by_id, cluster_articles_list, scoring_config)
        scored.append((label, score, indep_count, unique_src_ids, art_ids))

    # Sort by score descending, then by independent source count descending.
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    top_n = scored[: pipeline_config.stories_per_run]

    result: list[Cluster] = []
    for rank, (_label, score, indep_count, _, art_ids) in enumerate(top_n, start=1):
        # Use the title of the highest-tier article as the cluster label.
        cluster_label = _best_headline(art_ids, articles_by_id, sources_by_id)
        result.append(
            Cluster(
                run_id=0,  # set by the pipeline
                rank=rank,
                label=cluster_label,
                article_count=len(art_ids),
                independent_source_count=indep_count,
                tier_weighted_score=score,
            )
        )

    logger.info(
        "ranking_complete",
        total_clusters=len(cluster_map),
        selected=len(result),
    )
    return result


def _newest_published(articles: list[Article]) -> datetime | None:
    """Return the most recent ``published_at`` among articles, or None."""
    dates = [a.published_at for a in articles if a.published_at is not None]
    if not dates:
        dates = [a.fetched_at for a in articles]
    return max(dates) if dates else None


def _best_headline(
    art_ids: list[int],
    articles_by_id: dict[int, Article],
    sources_by_id: dict[int, Source],
) -> str:
    """Pick the headline from the highest-tier source in the cluster."""
    best_art: Article | None = None
    best_tier = 99

    for aid in art_ids:
        art = articles_by_id.get(aid)
        if art is None:
            continue
        src = sources_by_id.get(art.source_id)
        tier_val = int(src.tier) if src else 99
        if tier_val < best_tier:
            best_tier = tier_val
            best_art = art

    if best_art is not None:
        return best_art.title
    # Fallback: first available title.
    for aid in art_ids:
        art = articles_by_id.get(aid)
        if art is not None:
            return art.title
    return "Untitled cluster"
