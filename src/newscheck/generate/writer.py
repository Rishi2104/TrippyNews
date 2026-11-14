"""Constrained news narrative generation with NLI entailment filtering.

Rules:
1. Summaries are composed ONLY from AGREED claims (corroborated by 2+ sources).
2. Every generated sentence must pass an NLI entailment check against stored source passages.
3. Any candidate sentence failing entailment is dropped immediately.
4. If fewer than 2 sentences survive entailment, publish headline + claim table only.
5. In offline/zero-LLM mode, synthesizes an objective claim synthesis directly from agreed claims.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from newscheck.config import GenerationConfig, SafetyConfig
from newscheck.crosscheck.nli import NLICrossChecker
from newscheck.log import get_logger
from newscheck.models import Claim, ClaimLabel, CrossCheck

logger = get_logger(__name__)


class ConstrainedWriter:
    """Generates conservative, entailment-checked news summaries."""

    def __init__(
        self,
        generation_cfg: GenerationConfig,
        safety_cfg: SafetyConfig,
        nli_checker: NLICrossChecker,
    ) -> None:
        self._gen_cfg = generation_cfg
        self._safety_cfg = safety_cfg
        self._nli = nli_checker

    def generate_summary(
        self,
        agreed_claims: Sequence[Claim],
        source_passages: Sequence[str],
    ) -> str | None:
        """Synthesize a summary strictly from agreed claims, gated by NLI entailment.

        Returns summary text, or None if fewer than 2 sentences survive entailment.
        """
        if len(agreed_claims) < 2:
            return None

        # Formulate candidate sentences from agreed claims
        candidate_sentences = self._build_candidate_sentences(agreed_claims)

        # Entailment filter: Every candidate sentence MUST be entailed by a source passage
        verified_sentences: list[str] = []
        for sentence in candidate_sentences:
            if self._is_entailed_by_sources(sentence, source_passages):
                verified_sentences.append(sentence)
            else:
                logger.info("sentence_dropped_failed_entailment", sentence=sentence)

        # Minimum 2 surviving sentences required to publish narrative
        if len(verified_sentences) < 2:
            logger.info("insufficient_entailed_sentences_falling_back_to_table_only")
            return None

        # Cap max sentences from config
        final_sentences = verified_sentences[: self._gen_cfg.max_summary_sentences]
        return " ".join(final_sentences)

    def _build_candidate_sentences(self, agreed_claims: Sequence[Claim]) -> list[str]:
        """Convert agreed claims into clean, distinct summary sentences."""
        sentences: list[str] = []
        seen_texts: set[str] = set()

        for claim in agreed_claims:
            text = claim.claim_text.strip()
            # Ensure sentence formatting
            if not text.endswith((".", "!", "?")):
                text += "."
            text = text[0].upper() + text[1:]

            norm = re.sub(r"[^\w\s]", "", text.lower())
            if norm not in seen_texts:
                seen_texts.add(norm)
                sentences.append(text)

        return sentences

    def _is_entailed_by_sources(self, sentence: str, source_passages: Sequence[str]) -> bool:
        """Check if candidate sentence is entailed by at least one source passage."""
        dummy_claim = Claim(
            article_id=0,
            cluster_id=0,
            claim_text=sentence,
            source_sentence=sentence,
        )

        for passage in source_passages:
            passage_claim = Claim(
                article_id=0,
                cluster_id=0,
                claim_text=passage,
                source_sentence=passage,
            )
            verdict, conf = self._nli.compare_claims(dummy_claim, passage_claim)
            if verdict == ClaimLabel.AGREED and conf >= self._safety_cfg.min_entailment_score:
                return True

        return False


def get_agreed_claims(claims: Sequence[Claim], cross_checks: Sequence[CrossCheck]) -> list[Claim]:
    """Find all claims that participate in an AGREED cross-check."""
    agreed_claim_ids: set[int] = set()
    for cc in cross_checks:
        if cc.label == ClaimLabel.AGREED:
            agreed_claim_ids.add(cc.claim_a_id)
            agreed_claim_ids.add(cc.claim_b_id)

    return [c for c in claims if c.id in agreed_claim_ids]
