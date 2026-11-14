"""Tests for claim extraction (Rule-based and LLM fallback)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from newscheck.claims.extract import (
    GeminiClaimExtractor,
    OllamaClaimExtractor,
    RuleBasedClaimExtractor,
    get_claim_extractor,
)
from newscheck.config import GenerationConfig, Settings
from newscheck.models import Article, ExtractionStatus


def _make_article(text: str) -> Article:
    now = datetime.now(UTC)
    return Article(
        id=42,
        source_id=1,
        url="https://example.com/test-story",
        title="Test Headline",
        snippet="Snippet text",
        full_text=text,
        author="Reporter",
        published_at=now,
        fetched_at=now,
        content_hash="h123",
        extraction_status=ExtractionStatus.FULL,
        run_id=1,
    )


class TestRuleBasedClaimExtractor:
    def test_extract_factual_claims(self) -> None:
        extractor = RuleBasedClaimExtractor()
        body = (
            "The disaster management agency confirmed that 15 people were injured. "
            "Rescue teams deployed immediately after the tremor. "
            "Local weather reports indicated that rain would follow later in the day."
        )
        art = _make_article(body)
        claims = extractor.extract_claims(art, cluster_id=10)

        assert len(claims) >= 1
        assert any("15 people were injured" in c.source_sentence for c in claims)
        for claim in claims:
            assert claim.article_id == 42
            assert claim.cluster_id == 10
            # Every claim MUST anchor to a source sentence in the article
            assert claim.source_sentence in body

    def test_empty_article_returns_no_claims(self) -> None:
        extractor = RuleBasedClaimExtractor()
        art = _make_article("")
        claims = extractor.extract_claims(art, cluster_id=10)
        assert claims == []


class TestLLMClaimExtractors:
    def test_ollama_fallback_when_offline(self) -> None:
        cfg = GenerationConfig(llm_provider="ollama", ollama_base_url="http://invalid.host:9999")
        extractor = OllamaClaimExtractor(cfg)
        art = _make_article("Officials announced that 500 new schools will be opened.")
        claims = extractor.extract_claims(art, cluster_id=5)
        # Should gracefully fall back to rule-based extraction
        assert len(claims) >= 1
        assert "500 new schools" in claims[0].source_sentence

    def test_gemini_fallback_without_key(self) -> None:
        cfg = GenerationConfig(llm_provider="gemini")
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            extractor = GeminiClaimExtractor(cfg)
            art = _make_article("Police arrested 3 suspects in connection with the heist.")
            claims = extractor.extract_claims(art, cluster_id=5)
            assert len(claims) >= 1
            assert "arrested 3 suspects" in claims[0].source_sentence

    def test_factory(self) -> None:
        s1 = Settings(generation=GenerationConfig(llm_provider="none"))
        assert isinstance(get_claim_extractor(s1), RuleBasedClaimExtractor)

        s2 = Settings(generation=GenerationConfig(llm_provider="ollama"))
        assert isinstance(get_claim_extractor(s2), OllamaClaimExtractor)
