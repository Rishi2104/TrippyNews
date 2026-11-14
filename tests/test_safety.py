"""Tests for sensitive topic detection and safety gates."""

from __future__ import annotations

from datetime import UTC, datetime

from newscheck.models import Article, CorroborationLabel, ExtractionStatus, Story
from newscheck.safety.gates import SafetyAuditor
from newscheck.safety.sensitive import SensitiveTopicDetector


def _make_article(title: str, text: str = "") -> Article:
    now = datetime.now(UTC)
    return Article(
        id=1,
        source_id=1,
        url="https://example.com/art",
        title=title,
        snippet=text,
        full_text=text,
        published_at=now,
        fetched_at=now,
        content_hash="h",
        extraction_status=ExtractionStatus.FULL,
        run_id=1,
    )


class TestSensitiveTopicDetector:
    def test_detects_sexual_violence(self) -> None:
        detector = SensitiveTopicDetector()
        art = _make_article("Police investigate reported sexual assault in district")
        is_sens, cat = detector.is_sensitive([art], [])
        assert is_sens is True
        assert cat == "sexual_violence"

    def test_detects_suicide(self) -> None:
        detector = SensitiveTopicDetector()
        art = _make_article("Student tragically ended his life at campus")
        is_sens, cat = detector.is_sensitive([art], [])
        assert is_sens is True
        assert cat == "suicide"

    def test_clean_story_not_sensitive(self) -> None:
        detector = SensitiveTopicDetector()
        art = _make_article("ISRO announces new satellite launch schedule")
        is_sens, cat = detector.is_sensitive([art], [])
        assert is_sens is False
        assert cat is None


class TestSafetyAuditor:
    def test_unconfirmed_story_suppresses_narrative(self) -> None:
        auditor = SafetyAuditor()
        story = Story(
            cluster_id=1,
            headline="Sample Headline",
            corroboration_score=3.0,
            corroboration_label=CorroborationLabel.CORROBORATED,
            published_at=datetime.now(UTC),
            summary="A generated summary that should not be published.",
            is_sensitive=False,
        )

        audited = auditor.audit_story(story, independent_source_count=2, claims=[], articles=[])
        assert audited.corroboration_label == CorroborationLabel.UNCONFIRMED
        assert audited.summary is None

    def test_sensitive_topic_suppresses_narrative(self) -> None:
        auditor = SafetyAuditor()
        story = Story(
            cluster_id=1,
            headline="Sensitive Event",
            corroboration_score=9.0,
            corroboration_label=CorroborationLabel.STRONGLY_CORROBORATED,
            published_at=datetime.now(UTC),
            summary="This summary must be stripped due to sensitive topic policy.",
            is_sensitive=True,
        )

        audited = auditor.audit_story(story, independent_source_count=5, claims=[], articles=[])
        assert audited.summary is None

    def test_forbidden_claim_words_cleared(self) -> None:
        auditor = SafetyAuditor()
        story = Story(
            cluster_id=1,
            headline="Story is fact-checked and verified true",
            corroboration_score=8.0,
            corroboration_label=CorroborationLabel.CORROBORATED,
            published_at=datetime.now(UTC),
            summary="Officials fact-checked the incident.",
            is_sensitive=False,
        )

        audited = auditor.audit_story(story, independent_source_count=4, claims=[], articles=[])
        assert "fact-checked" not in audited.headline.lower()
        assert "verified true" not in audited.headline.lower()

    def test_quote_length_capping(self) -> None:
        auditor = SafetyAuditor()
        long_quote = (
            'The minister stated: "We are actively taking all measures to ensure the complete safety '
            'and security of every single resident in this district without any hesitation whatsoever."'
        )
        story = Story(
            cluster_id=1,
            headline="Minister Statement",
            corroboration_score=8.0,
            corroboration_label=CorroborationLabel.CORROBORATED,
            published_at=datetime.now(UTC),
            summary=long_quote,
            is_sensitive=False,
        )

        audited = auditor.audit_story(story, independent_source_count=4, claims=[], articles=[])
        assert audited.summary is not None
        # Check that quote inside summary was shortened to 15 words + ellipsis
        assert "..." in audited.summary
