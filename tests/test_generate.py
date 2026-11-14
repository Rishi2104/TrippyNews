"""Tests for constrained generation with NLI entailment filtering."""

from __future__ import annotations

from newscheck.config import GenerationConfig, SafetyConfig
from newscheck.crosscheck.nli import NLICrossChecker
from newscheck.generate.writer import ConstrainedWriter, get_agreed_claims
from newscheck.models import Claim, ClaimLabel, CrossCheck


class TestConstrainedWriter:
    def test_entailed_summary_generation(self) -> None:
        nli = NLICrossChecker(offline_mode=True)
        gen_cfg = GenerationConfig(max_summary_sentences=3)
        safety_cfg = SafetyConfig(min_entailment_score=0.6)
        writer = ConstrainedWriter(gen_cfg, safety_cfg, nli)

        agreed_claims = [
            Claim(
                id=1,
                article_id=1,
                cluster_id=1,
                claim_text="Disaster authority reported 15 injured in coastal tremor",
                source_sentence="Disaster authority reported 15 injured in coastal tremor.",
            ),
            Claim(
                id=2,
                article_id=2,
                cluster_id=1,
                claim_text="Rescue teams were immediately deployed across coastal district",
                source_sentence="Rescue teams were immediately deployed across coastal district.",
            ),
        ]
        source_passages = [
            "Disaster authority reported 15 injured in coastal tremor after early morning quake.",
            "Rescue teams were immediately deployed across coastal district by the government.",
        ]

        summary = writer.generate_summary(agreed_claims, source_passages)
        assert summary is not None
        assert "15 injured" in summary
        assert "Rescue teams" in summary

    def test_insufficient_entailed_sentences_returns_none(self) -> None:
        nli = NLICrossChecker(offline_mode=True)
        gen_cfg = GenerationConfig()
        safety_cfg = SafetyConfig(min_entailment_score=0.8)
        writer = ConstrainedWriter(gen_cfg, safety_cfg, nli)

        # Claim text completely un-entailed by source passages
        agreed_claims = [
            Claim(
                id=1,
                article_id=1,
                cluster_id=1,
                claim_text="Aliens landed in downtown Tokyo yesterday",
                source_sentence="Aliens landed in downtown Tokyo yesterday.",
            ),
            Claim(
                id=2,
                article_id=2,
                cluster_id=1,
                claim_text="Stock market reached zero",
                source_sentence="Stock market reached zero.",
            ),
        ]
        source_passages = ["The weather was mild and sunny today."]

        summary = writer.generate_summary(agreed_claims, source_passages)
        assert summary is None

    def test_get_agreed_claims(self) -> None:
        claims = [
            Claim(id=1, article_id=1, cluster_id=1, claim_text="C1", source_sentence="S1"),
            Claim(id=2, article_id=2, cluster_id=1, claim_text="C2", source_sentence="S2"),
            Claim(id=3, article_id=3, cluster_id=1, claim_text="C3", source_sentence="S3"),
        ]
        cross_checks = [
            CrossCheck(claim_a_id=1, claim_b_id=2, label=ClaimLabel.AGREED, nli_score=0.9),
            CrossCheck(claim_a_id=2, claim_b_id=3, label=ClaimLabel.SINGLE_SOURCE, nli_score=0.5),
        ]

        agreed = get_agreed_claims(claims, cross_checks)
        agreed_ids = {c.id for c in agreed}
        assert agreed_ids == {1, 2}
