"""Integration tests for the end-to-end Phase 1 pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from newscheck.config import Settings
from newscheck.ingest.feeds import FeedItem
from newscheck.models import ReliabilityTier, Source
from newscheck.pipeline import run_pipeline


class TestPipelineIntegration:
    @patch("newscheck.cluster.embed.ArticleEmbedder.embed")
    @patch("newscheck.extract.content.ArticleExtractor.extract")
    @patch("newscheck.ingest.feeds.FeedReader.read_all")
    def test_run_pipeline_end_to_end(
        self,
        mock_read_all: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        project_root: Path,
        settings: Settings,
    ) -> None:
        db_file = project_root / "test_pipeline.db"

        # Mock Feed Items
        mock_source = Source(
            id=1,
            name="Test Source A",
            domain="source-a.example.com",
            feed_url="https://source-a.example.com/rss",
            language="en",
            country="IN",
            tier=ReliabilityTier.WIRE_OFFICIAL,
        )

        mock_read_all.return_value = [
            FeedItem(
                source=mock_source,
                url=f"https://source-a.example.com/item-{i}",
                title=f"Breaking Story {i % 2}",
                snippet=f"Snippet for item {i}",
                published_at=None,
                author="PTI" if i == 0 else "Staff",
            )
            for i in range(4)
        ]

        from newscheck.models import Article, ExtractionStatus

        def fake_extract(item: FeedItem, source_id: int, run_id: int) -> Article:
            return Article(
                source_id=source_id,
                url=item.url,
                title=item.title,
                snippet=item.snippet,
                full_text=f"Full text for {item.title}. Details here.",
                author=item.author,
                published_at=item.published_at,
                content_hash=f"hash_{item.url}",
                extraction_status=ExtractionStatus.FULL,
                wire_origin="PTI" if item.author == "PTI" else None,
                run_id=run_id,
            )

        mock_extract.side_effect = fake_extract

        # 4 items: items 0 and 2 cluster together, items 1 and 3 cluster together
        mock_embed.return_value = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [0.05, 0.05],
                [10.05, 10.05],
            ],
            dtype=np.float32,
        )

        result = run_pipeline(
            settings=settings,
            project_root=project_root,
            db_path=db_file,
        )

        assert result.success is True
        assert result.articles_fetched == 4
        assert len(result.clusters) > 0
        assert result.run.id is not None
