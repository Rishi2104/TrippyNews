"""Tests for MinHash and wire deduplication."""

from __future__ import annotations

from datetime import UTC, datetime

from newscheck.config import DedupConfig
from newscheck.dedup.minhash import MinHashDeduplicator, compute_similarity
from newscheck.dedup.wire import WireDetector
from newscheck.models import Article, DedupMethod, ExtractionStatus


class TestMinHashDeduplicator:
    def test_compute_similarity(self) -> None:
        text1 = "A powerful earthquake struck the northern coast early morning."
        text2 = "A powerful earthquake struck the northern coast early morning today."
        text3 = "Stock markets hit all time highs following tech rally."

        sim_high = compute_similarity(text1, text2)
        sim_low = compute_similarity(text1, text3)

        assert sim_high > 0.6
        assert sim_low < 0.2

    def test_find_duplicates(self) -> None:
        config = DedupConfig(minhash_threshold=0.5, minhash_num_perm=128)
        deduper = MinHashDeduplicator(config)

        now = datetime.now(UTC)
        art1 = Article(
            id=10,
            source_id=1,
            url="https://a.com/1",
            title="Earthquake hits coast",
            full_text=("A massive 6.5 earthquake hit the coastal area early morning destroying several buildings."),
            content_hash="h1",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
            published_at=now,
            fetched_at=now,
        )
        art2 = Article(
            id=11,
            source_id=2,
            url="https://b.com/1",
            title="Earthquake hits coast area",
            full_text=(
                "A massive 6.5 earthquake hit the coastal area early morning "
                "destroying several buildings and structures."
            ),
            content_hash="h2",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
            published_at=now,
            fetched_at=now,
        )
        art3 = Article(
            id=12,
            source_id=3,
            url="https://c.com/1",
            title="Sensex gains 1000 points",
            full_text=("BSE Sensex jumped 1000 points in early trade on strong global cues and foreign inflows."),
            content_hash="h3",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
            published_at=now,
            fetched_at=now,
        )

        groups = deduper.find_duplicates([art1, art2, art3])
        assert len(groups) == 1
        assert groups[0].article_id == 11
        assert groups[0].canonical_id == 10
        assert groups[0].method == DedupMethod.MINHASH


class TestWireDetector:
    def test_wire_detection_author(self) -> None:
        detector = WireDetector()
        art = Article(
            id=1,
            source_id=1,
            url="https://example.com/art",
            title="Government introduces bill",
            author="PTI / New Delhi",
            content_hash="h",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
        )
        wire = detector.detect(art)
        assert wire == "PTI"

    def test_wire_detection_body(self) -> None:
        detector = WireDetector()
        art = Article(
            id=1,
            source_id=1,
            url="https://example.com/art2",
            title="Markets open higher",
            full_text="Mumbai: Stock indices opened higher on Friday, reported Reuters.",
            content_hash="h",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
        )
        wire = detector.detect(art)
        assert wire == "Reuters"

    def test_wire_grouping(self, sample_articles: list[Article]) -> None:
        detector = WireDetector()
        detector.annotate_articles(sample_articles)

        groups = detector.build_wire_dedup_groups(sample_articles)
        # Sample articles 1 and 3 are both PTI wire stories on earthquake
        matching = [g for g in groups if g.article_id == 3 and g.canonical_id == 1]
        assert len(matching) == 1
        assert matching[0].method == DedupMethod.WIRE
