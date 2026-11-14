"""Tests for embedding and clustering logic."""

from __future__ import annotations

import sqlite3

import numpy as np

from newscheck.cluster.group import (
    cluster_articles,
    count_independent_sources,
    rank_and_select,
    score_cluster,
)
from newscheck.config import ClusterConfig, PipelineConfig, ScoringConfig
from newscheck.models import Article, Source


class TestClustering:
    def test_count_independent_sources(self, db_conn: sqlite3.Connection, sample_articles: list[Article]) -> None:
        art_map = {a.id: a for a in sample_articles if a.id is not None}
        # Articles 1 and 2 from source 1 and source 2
        count, sources = count_independent_sources([1, 2], art_map, db_conn)
        assert count == 2
        assert sources == {1, 2}

    def test_score_cluster(self, sample_sources: list[Source], sample_articles: list[Article]) -> None:
        scoring_cfg = ScoringConfig(
            tier1_weight=3.0,
            tier2_weight=2.0,
            tier3_weight=1.0,
            primary_source_bonus=2.0,
        )
        sources_by_id = {s.id: s for s in sample_sources if s.id is not None}

        # Cluster with source 1 (tier 1 wire) and source 2 (tier 2)
        score = score_cluster(
            independent_count=2,
            source_ids={1, 2},
            sources_by_id=sources_by_id,
            articles=[sample_articles[0], sample_articles[1]],
            scoring_config=scoring_cfg,
        )
        # 3.0 (tier 1) + 2.0 (tier 2) + 2.0 (primary bonus) = 7.0 (with slight recency factor)
        assert score >= 6.0

    def test_cluster_articles_hdbscan(self) -> None:
        # Create synthetic 2D embeddings with 2 tight clusters
        cluster_cfg = ClusterConfig(min_cluster_size=2, min_samples=1)

        # 2 points at (0, 0), 2 points at (10, 10)
        embeddings = np.array(
            [
                [0.0, 0.0],
                [0.05, 0.05],
                [10.0, 10.0],
                [10.05, 10.05],
            ],
            dtype=np.float32,
        )

        # Mock articles corresponding to points
        articles = [
            Article(
                id=i,
                source_id=i,
                url=f"http://test.com/{i}",
                title=f"Title {i}",
                content_hash=f"h{i}",
                extraction_status="full",  # type: ignore[arg-type]
                run_id=1,
            )
            for i in range(1, 5)
        ]

        clusters = cluster_articles(embeddings, articles, cluster_cfg)
        assert len(clusters) == 2

    def test_rank_and_select(
        self,
        db_conn: sqlite3.Connection,
        sample_sources: list[Source],
        sample_articles: list[Article],
    ) -> None:
        sources_by_id = {s.id: s for s in sample_sources if s.id is not None}
        cluster_map = {
            0: [0, 1, 2],  # Earthquake cluster (articles 1, 2, 3)
            1: [3, 4],  # Markets cluster (articles 4, 5)
        }

        pipeline_cfg = PipelineConfig(stories_per_run=5)
        scoring_cfg = ScoringConfig()

        ranked = rank_and_select(
            cluster_map,
            sample_articles,
            sources_by_id,
            db_conn,
            pipeline_config=pipeline_cfg,
            scoring_config=scoring_cfg,
        )

        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        # Verify cluster labels come from titles
        assert "earthquake" in ranked[0].label.lower() or "markets" in ranked[0].label.lower()
