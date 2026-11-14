"""Sensitive topic detection.

Identifies topics that require high integrity caution:
- Crime naming individuals
- Health or medical claims
- Communal / religious / political violence
- Minors / children in distress or crime
- Suicide / self-harm
- Sexual violence
- Active armed conflicts
- Contested election results

For any sensitive topic, NewsCheck PUBLISHES ONLY:
neutral headline, source links, and verified claim table.
NO GENERATED NARRATIVE IS ALLOWED.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from newscheck.models import Article, Claim

# Keyword and phrase patterns for sensitive categories
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "sexual_violence",
        re.compile(r"\b(rape|raped|sexual assault|molestation|posco|pocso)\b", re.IGNORECASE),
    ),
    (
        "suicide",
        re.compile(
            r"\b(suicide|suicidal|ended (?:his|her|their) life|hung (?:himself|herself))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "minors_distress",
        re.compile(
            r"\b(minor child|juvenile delinquent|underage victim|schoolchild abducted)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "communal_violence",
        re.compile(
            r"\b(communal clash|lynching|mob violence|religious riot|sectarian clash)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "active_conflict",
        re.compile(
            r"\b(missile strike|airstrike|artillery barrage|incursion|cross-border firing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "health_medical",
        re.compile(
            r"\b(clinical trial fails|cure for cancer|vaccine death|pandemic outbreak|miracle remedy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "election_dispute",
        re.compile(
            r"\b(election rigging|ballot tampering|booth capturing|evm hacked|stolen election)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "crime_named_individual",
        re.compile(r"\b(?:arrested|booked|accused|charged with murder|detained)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
    ),
]


class SensitiveTopicDetector:
    """Classifies whether a story cluster covers sensitive topics."""

    def is_sensitive(self, articles: Sequence[Article], claims: Sequence[Claim]) -> tuple[bool, str | None]:
        """Check articles and claims for sensitive topic triggers.

        Returns:
            (is_sensitive, category_or_none)
        """
        # Aggregate text from titles, snippets, and claims
        texts: list[str] = []
        for art in articles:
            texts.append(art.title)
            if art.snippet:
                texts.append(art.snippet)
        for claim in claims:
            texts.append(claim.claim_text)
            texts.append(claim.source_sentence)

        combined = " ".join(texts)

        for category, pattern in _SENSITIVE_PATTERNS:
            if pattern.search(combined):
                return True, category

        return False, None
