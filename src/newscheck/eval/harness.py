"""Evaluation harness for benchmark testing of pipeline accuracy and safety.

Tests:
1. Wire deduplication accuracy (correctly grouping wire reprints).
2. Claim anchoring integrity (claims strictly match source passages).
3. NLI agreement and dispute detection accuracy.
4. Safety gate compliance (zero forbidden phrases, zero sensitive narrative leaks).
"""

from __future__ import annotations

from dataclasses import dataclass

from newscheck.claims.extract import RuleBasedClaimExtractor
from newscheck.crosscheck.nli import NLICrossChecker
from newscheck.dedup.wire import WireDetector
from newscheck.models import Article, Claim, ClaimLabel, CorroborationLabel, ExtractionStatus, Story
from newscheck.safety.gates import SafetyAuditor
from newscheck.safety.sensitive import SensitiveTopicDetector


@dataclass
class EvalReport:
    """Summary of benchmark evaluation metrics."""

    wire_dedup_accuracy: float
    claim_anchoring_accuracy: float
    nli_accuracy: float
    safety_gate_compliance: float
    all_passed: bool


class IntegrityEvaluator:
    """Runs automated integrity benchmarks on test datasets."""

    def evaluate_wire_detection(self) -> float:
        detector = WireDetector()
        test_articles = [
            Article(
                id=1,
                source_id=1,
                url="http://1",
                title="Govt news",
                author="PTI",
                content_hash="1",
                extraction_status=ExtractionStatus.FULL,
                run_id=1,
            ),
            Article(
                id=2,
                source_id=2,
                url="http://2",
                title="Govt news",
                full_text="Reuters reports that ...",
                content_hash="2",
                extraction_status=ExtractionStatus.FULL,
                run_id=1,
            ),
            Article(
                id=3,
                source_id=3,
                url="http://3",
                title="Opinion piece",
                full_text="Editorial staff",
                content_hash="3",
                extraction_status=ExtractionStatus.FULL,
                run_id=1,
            ),
        ]
        detector.annotate_articles(test_articles)
        correct = (
            test_articles[0].wire_origin == "PTI"
            and test_articles[1].wire_origin == "Reuters"
            and test_articles[2].wire_origin is None
        )
        return 1.0 if correct else 0.0

    def evaluate_claim_anchoring(self) -> float:
        extractor = RuleBasedClaimExtractor()
        body = "The reserve bank increased interest rates by 25 basis points today."
        art = Article(
            id=1,
            source_id=1,
            url="http://1",
            title="RBI",
            full_text=body,
            content_hash="h",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
        )
        claims = extractor.extract_claims(art, cluster_id=1)
        if not claims:
            return 0.0
        anchored = all(c.source_sentence in body for c in claims)
        return 1.0 if anchored else 0.0

    def evaluate_nli_accuracy(self) -> float:
        nli = NLICrossChecker(offline_mode=True)
        # Agreement pair
        ca1 = Claim(
            article_id=1,
            cluster_id=1,
            claim_text="Tremor measured 6.5 magnitude",
            source_sentence="Tremor measured 6.5 magnitude",
        )
        ca2 = Claim(
            article_id=2,
            cluster_id=1,
            claim_text="Tremor measured 6.5 magnitude",
            source_sentence="Tremor measured 6.5 magnitude",
        )
        v1, _ = nli.compare_claims(ca1, ca2)

        # Contradiction pair
        cd1 = Claim(
            article_id=1,
            cluster_id=1,
            claim_text="10 casualties reported",
            source_sentence="10 casualties reported",
        )
        cd2 = Claim(
            article_id=2,
            cluster_id=1,
            claim_text="90 casualties reported",
            source_sentence="90 casualties reported",
        )
        v2, _ = nli.compare_claims(cd1, cd2)

        correct = (v1 == ClaimLabel.AGREED) and (v2 == ClaimLabel.DISPUTED)
        return 1.0 if correct else 0.0

    def evaluate_safety_compliance(self) -> float:
        auditor = SafetyAuditor()
        detector = SensitiveTopicDetector()

        # Sensitive story must not have summary
        sens_art = Article(
            id=1,
            source_id=1,
            url="http://1",
            title="Massive riot in district",
            snippet="mob violence erupted",
            content_hash="1",
            extraction_status=ExtractionStatus.FULL,
            run_id=1,
        )
        is_sens, _ = detector.is_sensitive([sens_art], [])

        story = Story(
            cluster_id=1,
            headline="Headline with fact-checked claim",
            summary="A summary that should be stripped",
            corroboration_score=5.0,
            corroboration_label=CorroborationLabel.CORROBORATED,
            is_sensitive=is_sens,
            published_at=None,  # type: ignore[arg-type]
        )
        audited = auditor.audit_story(story, independent_source_count=3, claims=[], articles=[sens_art])

        compliant = (audited.summary is None) and ("fact-checked" not in audited.headline.lower())
        return 1.0 if compliant else 0.0

    def run_benchmark(self) -> EvalReport:
        w_score = self.evaluate_wire_detection()
        c_score = self.evaluate_claim_anchoring()
        n_score = self.evaluate_nli_accuracy()
        s_score = self.evaluate_safety_compliance()

        all_ok = all(s >= 1.0 for s in (w_score, c_score, n_score, s_score))
        return EvalReport(
            wire_dedup_accuracy=w_score,
            claim_anchoring_accuracy=c_score,
            nli_accuracy=n_score,
            safety_gate_compliance=s_score,
            all_passed=all_ok,
        )
