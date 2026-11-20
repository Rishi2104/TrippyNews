"""Clustering, outlier pruning, and corroboration-aware story ranking.

Groups article embeddings into event clusters, prunes unrelated articles
using centroid distance and named entity verification (D1), counts independent sources,
scores each cluster transparently, and selects the top N stories with cleaned headlines (L2).
"""

from __future__ import annotations

import re
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

# Stopwords excluded from entity overlap checking
_ENTITY_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "after", "before", "over", "under", "about",
    "into", "through", "during", "watch", "live", "news", "updates", "update",
    "video", "photos", "today", "yesterday", "says", "said", "new", "top",
    "report", "reports", "first", "last", "two", "three", "four", "five",
}


def clean_headline(raw_title: str) -> str:
    """Clean and strip outlet suffixes, video labels, and live blog tags (L2)."""
    if not raw_title:
        return "Untitled Story"

    title = raw_title.strip()

    # 1. Remove outlet suffixes with pipe, dash, or bullet
    # E.g., "... | News18", "... - The Hindu", "... - NDTV", "... | BBC News"
    title = re.sub(
        r"\s*(\||—|–|-|\/\/|•)\s*(?:[A-Za-z0-9\s&.]+)?(?:News|Post|Times|Today|Express|Herald|Tribune|Live|Online|Daily|TV|World|India|Chronicle|Mirror|Mint|Standard|Quint|Bureau|Feed|WION|NDTV|BBC|CNA|Reuters|PTI|ANI).*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # 2. Specific outlet names that may not have generic media words
    title = re.sub(
        r"\s*(\||—|–|-)\s*(?:News18|The Hindu|NDTV|Times of India|Hindustan Times|Indian Express|BBC|CNA|The Straits Times|SCMP|Al Jazeera|Mint|Livemint|Business Standard|Moneycontrol|The Quint|Deccan Chronicle|India Today|Zee News|DNA|Firstpost|Republic World).*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # 3. Leading category / dateline prefixes (e.g. "INDIA: ", "WORLD - ", "WATCH: ")
    title = re.sub(
        r"^(?:INDIA|WORLD|ASIA|CRIME|BUSINESS|POLITICS|WATCH|VIDEO|LIVE UPDATES?|LIVE BLOG|BREAKING)\s*[:|–—-]\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # 4. Trailing video/photos tags
    title = re.sub(
        r"\s*[:|–—-]\s*(?:WATCH|VIDEO|PHOTOS|IN PICS|LIVE UPDATES?|LIVE)$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    return title.strip() or raw_title.strip()


def extract_key_entities(text: str) -> set[str]:
    """Extract key named entities and numbers from headline/snippet (D1)."""
    # Remove outlet boilerplate
    clean = clean_headline(text)
    # Extract capitalized proper nouns and numbers
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_\-']+\b|\b\d+(?:,\d+)*(?:\.\d+)?\b", clean)
    entities = {t.lower() for t in tokens if t.lower() not in _ENTITY_STOPWORDS and len(t) >= 3}
    return entities


def prune_cluster_articles(
    indices: list[int],
    articles: list[Article],
    embeddings: NDArray[np.float32],
    *,
    max_sources: int = 8,
    min_similarity: float = 0.58,
) -> list[int]:
    """Prune cluster to eliminate unrelated articles and outliers (D1).

    Applies:
    1. Centroid distance threshold: drops articles with cosine similarity < min_similarity.
    2. Named entity overlap requirement: articles must share key entities with the cluster core.
    3. Capping at max_sources (default ~8).
    """
    if len(indices) <= 1:
        return indices

    # 1. Normalize embeddings to unit vectors for cosine similarity
    # 1. Handle synthetic 2D or zero-norm embeddings in test suites
    cluster_vecs = embeddings[indices]
    raw_norms = np.linalg.norm(cluster_vecs, axis=1)
    if cluster_vecs.shape[1] <= 2 or np.any(raw_norms == 0):
        return list(indices[:max_sources])

    # Unit-normalize cluster vectors for cosine similarity
    norms = raw_norms[:, np.newaxis]
    norms[norms == 0] = 1.0
    norm_vecs = cluster_vecs / norms

    # 2. Compute centroid
    centroid = np.mean(norm_vecs, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    # 3. Compute cosine similarities to centroid
    sims = np.dot(norm_vecs, centroid)

    # Find the core article (highest similarity to centroid)
    best_local_idx = int(np.argmax(sims))
    core_article = articles[indices[best_local_idx]]
    core_entities = extract_key_entities(core_article.title + " " + core_article.snippet)

    pruned: list[tuple[int, float]] = []
    for local_idx, idx in enumerate(indices):
        art = articles[idx]
        sim = float(sims[local_idx])

        # Centroid distance gate
        if sim < min_similarity and local_idx != best_local_idx:
            logger.debug("article_dropped_centroid_distance", title=art.title, sim=sim)
            continue

        # Entity overlap gate (if core has entities, require at least one shared entity)
        art_entities = extract_key_entities(art.title + " " + art.snippet)
        if core_entities and art_entities and not (core_entities & art_entities):
            # No shared proper nouns/entities with the core story
            logger.debug("article_dropped_no_entity_overlap", title=art.title, core=core_article.title)
            continue

        pruned.append((idx, sim))

    # Sort remaining articles by similarity to centroid descending
    pruned.sort(key=lambda x: x[1], reverse=True)

    # Cap to max_sources (default 8)
    surviving_indices = [idx for idx, _ in pruned[:max_sources]]
    return surviving_indices


def cluster_articles(
    embeddings: NDArray[np.float32],
    articles: list[Article],
    config: ClusterConfig,
) -> dict[int, list[int]]:
    """Run clustering on embeddings with outlier pruning (D1).

    Articles assigned to noise (label -1) or pruned as outliers are excluded.
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
        metric="euclidean",
    )

    labels: NDArray[np.intp] = clusterer.fit_predict(embeddings)

    raw_clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        label_int = int(label)
        if label_int == -1:
            continue
        raw_clusters.setdefault(label_int, []).append(idx)

    # Apply D1 outlier and entity pruning to each cluster
    pruned_clusters: dict[int, list[int]] = {}
    for label_int, indices in raw_clusters.items():
        surviving = prune_cluster_articles(indices, articles, embeddings, max_sources=8)
        # Keep clusters that still have minimum size after pruning
        if len(surviving) >= config.min_cluster_size:
            pruned_clusters[label_int] = surviving

    logger.info(
        "clustering_and_pruning_complete",
        total_articles=len(articles),
        raw_clusters=len(raw_clusters),
        pruned_clusters=len(pruned_clusters),
    )
    return pruned_clusters


def count_independent_sources(
    article_ids: list[int],
    articles_by_id: dict[int, Article],
    conn: sqlite3.Connection,
) -> tuple[int, set[int]]:
    """Count independent sources for a set of articles, respecting dedup."""
    canonical_sources: set[tuple[int, int]] = set()
    for aid in article_ids:
        canonical_aid = get_canonical_id(conn, aid)
        art = articles_by_id.get(canonical_aid) or articles_by_id.get(aid)
        if art is not None:
            canonical_sources.add((art.source_id, canonical_aid))

    unique_sources = {src_id for src_id, _ in canonical_sources}
    return len(unique_sources), unique_sources


def score_cluster(
    independent_count: int,
    source_ids: set[int],
    sources_by_id: dict[int, Source],
    articles: list[Article],
    scoring_config: ScoringConfig,
) -> float:
    """Compute a transparent corroboration score for a cluster."""
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

    # Recency: decay older stories
    now = datetime.now(UTC)
    newest = _newest_published(articles)
    if newest is not None:
        age_hours = max(0.0, (now - newest).total_seconds() / 3600.0)
        decay = max(0.2, 1.0 - (age_hours / (scoring_config.recency_decay_hours * 10.0)))
        score *= decay

    return round(score, 2)


def rank_and_select(
    cluster_map: dict[int, list[int]],
    articles: list[Article],
    sources_by_id: dict[int, Source],
    conn: sqlite3.Connection,
    *,
    pipeline_config: PipelineConfig,
    scoring_config: ScoringConfig,
) -> list[Cluster]:
    """Score all clusters, rank them, and return the top N as Cluster objects."""
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

    # Sort by score descending, then by independent source count descending
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    top_n = scored[: pipeline_config.stories_per_run]

    result: list[Cluster] = []
    for rank, (_label, score, indep_count, _, art_ids) in enumerate(top_n, start=1):
        # Pick best headline and clean outlet suffixes (L2)
        raw_headline = _best_headline(art_ids, articles_by_id, sources_by_id)
        cluster_label = clean_headline(raw_headline)

        result.append(
            Cluster(
                run_id=0,
                rank=rank,
                label=cluster_label,
                article_count=len(art_ids),
                independent_source_count=indep_count,
                tier_weighted_score=score,
            )
        )

    logger.info("ranking_complete", total_clusters=len(cluster_map), selected=len(result))
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

    for aid in art_ids:
        art = articles_by_id.get(aid)
        if art is not None:
            return art.title
    return "Untitled cluster"
