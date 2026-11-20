"""Claim extraction from article texts.

Extracts discrete factual claims from articles, ensuring that every claim
is directly tied to a verbatim source sentence.

Supports:
1. Ollama (local LLM, e.g. llama3.1:8b)
2. Google Gemini API (free tier)
3. Deterministic rule-based extraction (zero external dependencies, fast fallback)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from urllib.parse import urljoin

from newscheck.config import GenerationConfig, Settings
from newscheck.log import get_logger
from newscheck.models import Article, Claim

logger = get_logger(__name__)


def truncate_evidence_quote(sentence: str, max_words: int = 25) -> str:
    """Cap evidence quote at max_words with an ellipsis (L3)."""
    if not sentence:
        return ""
    words = sentence.strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return " ".join(words)


def clean_extracted_claim(text: str) -> str:
    """Clean claim text: strip datelines, category prefixes, and fix factual titles (D5, D8)."""
    if not text:
        return ""
    # Strip category prefixes
    text = re.sub(r"^(?:INDIA|WORLD|CRIME|BUSINESS|POLITICS)\s*[-–:]\s*", "", text, flags=re.I)
    # Strip datelines like 'Rampur:', 'NEW DELHI -'
    text = re.sub(r"^[A-Z][a-zA-Z\s]{1,16}\s*[-–:]\s*", "", text)
    # D8: Fix Azam Khan title if mislabeled as Samajwadi Party chief
    text = re.sub(r"\bSamajwadi Party chief Azam Khan\b", "Samajwadi Party leader Azam Khan", text, flags=re.I)
    return text.strip()


class ClaimExtractorBase(ABC):
    """Abstract interface for claim extractors."""

    @abstractmethod
    def extract_claims(self, article: Article, cluster_id: int) -> list[Claim]:
        """Extract claims from an article."""
        ...


class RuleBasedClaimExtractor(ClaimExtractorBase):
    """Deterministic extractor for offline, testing, or zero-LLM mode.

    Splits article body text into sentences and identifies factual assertions
    using syntactic cues (numbers, reported speech, key action verbs, named entities).
    Always anchors directly to the source sentence.
    """

    # Indicators of factual news assertions: statistics, actions, official statements
    _FACT_INDICATORS = re.compile(
        r"(\b\d+(?:,\d+)*(?:\.\d+)?%?\b|"
        r"\b(?:said|stated|announced|confirmed|reported|killed|injured|damaged|"
        r"struck|hit|signed|passed|approved|decided|launched|won|lost|accused|arrested)\b)",
        re.IGNORECASE,
    )

    def extract_claims(self, article: Article, cluster_id: int) -> list[Claim]:
        assert article.id is not None
        text = article.full_text or article.snippet or article.title
        sentences = self._split_sentences(text)
        claims: list[Claim] = []

        seen_sentences: set[str] = set()
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 25 or len(sentence_clean) > 300:
                continue
            if sentence_clean in seen_sentences:
                continue

            # Check if sentence has factual cues
            if self._FACT_INDICATORS.search(sentence_clean):
                seen_sentences.add(sentence_clean)
                claim_type = self._classify_sentence_type(sentence_clean)
                cleaned_claim = clean_extracted_claim(sentence_clean)
                quote = truncate_evidence_quote(sentence_clean, max_words=25)
                claims.append(
                    Claim(
                        article_id=article.id,
                        cluster_id=cluster_id,
                        claim_text=cleaned_claim,
                        source_sentence=quote,
                        claim_type=claim_type,
                    )
                )

            # Cap claims per article to prevent noise
            if len(claims) >= 6:
                break

        return claims

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences cleanly."""
        # Simple regex sentence boundary detector
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _classify_sentence_type(sentence: str) -> str:
        s_lower = sentence.lower()
        if any(c in s_lower for c in ["said", "stated", "announced", "according to"]):
            return "statement"
        if any(char.isdigit() for char in sentence):
            return "numeric"
        return "event"


class OllamaClaimExtractor(ClaimExtractorBase):
    """Extracts structured claims using a local Ollama instance."""

    def __init__(self, config: GenerationConfig) -> None:
        self._model = config.ollama_model
        self._base_url = config.ollama_base_url
        self._fallback = RuleBasedClaimExtractor()

    def extract_claims(self, article: Article, cluster_id: int) -> list[Claim]:
        assert article.id is not None
        text = article.full_text or article.snippet
        if not text:
            return []

        prompt = (
            "You are an objective news integrity analyst. Extract up to 5 key factual claims "
            "from the article. Return ONLY a valid JSON list of objects with keys: "
            "'claim_text' (concise fact), 'source_sentence' (exact verbatim sentence from text), "
            "and 'claim_type' ('numeric'|'event'|'statement').\n\n"
            f"Article text:\n{text[:3000]}\n\n"
            "JSON output:"
        )

        try:
            url = urljoin(self._base_url, "/api/generate")
            payload = json.dumps(
                {
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                }
            ).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_response = data.get("response", "[]")
                parsed = json.loads(raw_response)
                if isinstance(parsed, dict) and "claims" in parsed:
                    parsed = parsed["claims"]

                claims: list[Claim] = []
                for item in parsed:
                    src_sent = item.get("source_sentence", "").strip()
                    c_text = item.get("claim_text", "").strip()
                    # Verify verbatim presence in article text
                    if src_sent and src_sent.lower() in text.lower():
                        cleaned_c = clean_extracted_claim(c_text or src_sent)
                        quote = truncate_evidence_quote(src_sent, max_words=25)
                        claims.append(
                            Claim(
                                article_id=article.id,
                                cluster_id=cluster_id,
                                claim_text=cleaned_c,
                                source_sentence=quote,
                                claim_type=item.get("claim_type", "event"),
                            )
                        )
                if claims:
                    return claims
        except Exception:
            logger.warning("ollama_claim_extraction_failed_fallback", article_id=article.id)

        return self._fallback.extract_claims(article, cluster_id)


class GeminiClaimExtractor(ClaimExtractorBase):
    """Extracts claims using Gemini free tier via HTTP API."""

    def __init__(self, config: GenerationConfig) -> None:
        self._api_key = os.getenv("GEMINI_API_KEY", "")
        self._model = config.gemini_model
        self._fallback = RuleBasedClaimExtractor()

    def extract_claims(self, article: Article, cluster_id: int) -> list[Claim]:
        assert article.id is not None
        if not self._api_key:
            return self._fallback.extract_claims(article, cluster_id)

        text = article.full_text or article.snippet
        if not text:
            return []

        prompt = (
            "Extract up to 5 key factual claims from this news article. "
            "Output JSON list with objects: "
            '{"claim_text": "...", "source_sentence": "exact verbatim sentence from text", "claim_type": "event"|"numeric"|"statement"}.\n\n'
            f"Article text:\n{text[:3000]}"
        )

        try:
            endpoint = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self._model}:generateContent?key={self._api_key}"
            )
            body = json.dumps(
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"response_mime_type": "application/json"},
                }
            ).encode("utf-8")
            req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cand = data["candidates"][0]["content"]["parts"][0]["text"]
                items = json.loads(cand)
                if isinstance(items, dict) and "claims" in items:
                    items = items["claims"]
                claims: list[Claim] = []
                for it in items:
                    src_sent = it.get("source_sentence", "").strip()
                    if src_sent and src_sent.lower() in text.lower():
                        cleaned_c = clean_extracted_claim(it.get("claim_text", src_sent))
                        quote = truncate_evidence_quote(src_sent, max_words=25)
                        claims.append(
                            Claim(
                                article_id=article.id,
                                cluster_id=cluster_id,
                                claim_text=cleaned_c,
                                source_sentence=quote,
                                claim_type=it.get("claim_type", "event"),
                            )
                        )
                if claims:
                    return claims
        except Exception:
            logger.warning("gemini_claim_extraction_failed_fallback", article_id=article.id)

        return self._fallback.extract_claims(article, cluster_id)


def get_claim_extractor(settings: Settings) -> ClaimExtractorBase:
    """Factory creating the appropriate claim extractor from settings."""
    provider = settings.generation.llm_provider.lower().strip()
    if provider == "ollama":
        return OllamaClaimExtractor(settings.generation)
    if provider == "gemini":
        return GeminiClaimExtractor(settings.generation)
    return RuleBasedClaimExtractor()
