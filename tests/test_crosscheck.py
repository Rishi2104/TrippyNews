"""Tests for NLI cross-checking between claims."""

from __future__ import annotations

from newscheck.crosscheck.nli import NLICrossChecker, cross_check_cluster_claims
from newscheck.models import Article, Claim, ClaimLabel, ExtractionStatus


class TestNLICrossChecker:
    def test_agreement_heuristic(self) -> None:
        checker = NLICrossChecker(offline_mode=True)
        claim1 = Claim(
            id=1,
            article_id=101,
            cluster_id=1,
            claim_text="Disaster agency reported 15 injured in coastal tremor",
            source_sentence="Disaster agency reported 15 injured in coastal tremor.",
        )
        claim2 = Claim(
            id=2,
            article_id=102,
            cluster_id=1,
            claim_text="At least 15 injured in coastal tremor according to disaster agency",
            source_sentence="At least 15 injured in coastal tremor according to disaster agency.",
        )

        label, score = checker.compare_claims(claim1, claim2)
        assert label == ClaimLabel.AGREED
        assert score >= 0.6

    def test_numeric_dispute_heuristic(self) -> None:
        checker = NLICrossChecker(offline_mode=True)
        claim1 = Claim(
            id=1,
            article_id=101,
            cluster_id=1,
            claim_text="Police reported 12 casualties following the blast",
            source_sentence="Police reported 12 casualties following the blast.",
        )
        claim2 = Claim(
            id=2,
            article_id=102,
            cluster_id=1,
            claim_text="Police reported 45 casualties following the blast",
            source_sentence="Police reported 45 casualties following the blast.",
        )

        label, score = checker.compare_claims(claim1, claim2)
        assert label == ClaimLabel.DISPUTED
        assert score >= 0.7

    def test_cluster_cross_checking(self) -> None:
        checker = NLICrossChecker(offline_mode=True)
        art1 = Article(
            id=1,
            source_id=10,
            url="http://a.com/1",
            title="A",
            content_hash="1",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
        )
        art2 = Article(
            id=2,
            source_id=20,
            url="http://b.com/1",
            title="B",
            content_hash="2",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
        )
        claims = [
            Claim(
                id=1,
                article_id=1,
                cluster_id=1,
                claim_text="Markets rallied 500 points",
                source_sentence="Markets rallied 500 points",
            ),
            Claim(
                id=2,
                article_id=2,
                cluster_id=1,
                claim_text="Markets surged 500 points on Friday",
                source_sentence="Markets surged 500 points on Friday",
            ),
        ]
        articles_map = {1: art1, 2: art2}

        results = cross_check_cluster_claims(claims, articles_map, checker)
        assert len(results) == 1
        assert results[0].claim_a_id == 1
        assert results[0].claim_b_id == 2
        assert results[0].label == ClaimLabel.AGREED
