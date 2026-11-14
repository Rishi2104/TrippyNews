"""Pairwise Natural Language Inference (NLI) cross-checking for claims.

Compares claims across independent sources in a cluster to identify:
- AGREED: Both independent sources corroborate the claim (entailment).
- DISPUTED: Sources contradict each other (contradiction).
- SINGLE_SOURCE: Claim only appears in one source or is neutral.
"""

from __future__ import annotations

import re
from typing import Any

from newscheck.log import get_logger
from newscheck.models import Article, Claim, ClaimLabel, CrossCheck

logger = get_logger(__name__)


class NLICrossChecker:
    """NLI cross-checker for comparing claim pairs across independent sources."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        *,
        offline_mode: bool = False,
    ) -> None:
        self._model_name = model_name
        self._offline_mode = offline_mode
        self._classifier: Any = None

    def _load_model(self) -> Any:
        if self._classifier is None and not self._offline_mode:
            try:
                from transformers import pipeline

                logger.info("loading_nli_model", model=self._model_name)
                # Zero-shot / NLI classification pipeline
                self._classifier = pipeline(
                    "text-classification",
                    model=self._model_name,
                    top_k=None,
                    device=-1,  # CPU
                )
            except Exception:
                logger.warning("nli_model_load_failed_falling_back_to_heuristic")
                self._offline_mode = True
        return self._classifier

    def compare_claims(self, claim_a: Claim, claim_b: Claim) -> tuple[ClaimLabel, float]:
        """Compare two claims and return (verdict, confidence_score)."""
        text_a = claim_a.source_sentence or claim_a.claim_text
        text_b = claim_b.source_sentence or claim_b.claim_text

        # In offline/fallback mode or if text is identical
        if self._offline_mode or self._classifier is None:
            return self._heuristic_compare(text_a, text_b)

        try:
            # Pairwise cross-encoder classification
            # cross-encoder/nli-deberta-v3-base outputs CONTRADICTION, ENTAILMENT, NEUTRAL
            res = self._classifier(f"{text_a} [SEP] {text_b}")
            scores = {item["label"].upper(): item["score"] for item in res[0]}
            entail = scores.get("ENTAILMENT", 0.0)
            contra = scores.get("CONTRADICTION", 0.0)

            if entail >= 0.65:
                return ClaimLabel.AGREED, float(entail)
            if contra >= 0.50:
                return ClaimLabel.DISPUTED, float(contra)
            return ClaimLabel.SINGLE_SOURCE, float(scores.get("NEUTRAL", 0.5))
        except Exception:
            return self._heuristic_compare(text_a, text_b)

    @staticmethod
    def _heuristic_compare(text_a: str, text_b: str) -> tuple[ClaimLabel, float]:
        """Syntactic heuristic for comparing claims without neural weights."""
        norm_a = re.sub(r"[^\w\s]", "", text_a.lower())
        norm_b = re.sub(r"[^\w\s]", "", text_b.lower())

        words_a = set(norm_a.split())
        words_b = set(norm_b.split())

        if not words_a or not words_b:
            return ClaimLabel.SINGLE_SOURCE, 0.5

        # Check for numeric contradiction (e.g. 50 dead vs 12 dead)
        nums_a = set(re.findall(r"\b\d+\b", text_a))
        nums_b = set(re.findall(r"\b\d+\b", text_b))
        has_common_topic = len(words_a & words_b) / min(len(words_a), len(words_b)) >= 0.4
        if has_common_topic and nums_a and nums_b and not (nums_a & nums_b):
            return ClaimLabel.DISPUTED, 0.85

        # Check for negation contradiction
        negations = {"not", "never", "no", "denied", "denies", "refused", "rejected"}
        neg_a = bool(words_a & negations)
        neg_b = bool(words_b & negations)
        if has_common_topic and (neg_a != neg_b):
            return ClaimLabel.DISPUTED, 0.80

        # Jaccard overlap on significant words
        common = words_a & words_b
        overlap = len(common) / min(len(words_a), len(words_b))
        if overlap >= 0.60:
            return ClaimLabel.AGREED, round(min(0.99, overlap), 2)

        return ClaimLabel.SINGLE_SOURCE, 0.50


def cross_check_cluster_claims(
    claims: list[Claim],
    articles_by_id: dict[int, Article],
    checker: NLICrossChecker,
) -> list[CrossCheck]:
    """Perform all pairwise cross-checks among claims from independent articles."""
    results: list[CrossCheck] = []
    n = len(claims)

    for i in range(n):
        claim_a = claims[i]
        art_a = articles_by_id.get(claim_a.article_id)
        if art_a is None or claim_a.id is None:
            continue

        for j in range(i + 1, n):
            claim_b = claims[j]
            art_b = articles_by_id.get(claim_b.article_id)
            if art_b is None or claim_b.id is None:
                continue

            # Compare only claims from different sources
            if art_a.source_id == art_b.source_id:
                continue

            label, score = checker.compare_claims(claim_a, claim_b)
            results.append(
                CrossCheck(
                    claim_a_id=claim_a.id,
                    claim_b_id=claim_b.id,
                    label=label,
                    nli_score=score,
                )
            )

    return results
