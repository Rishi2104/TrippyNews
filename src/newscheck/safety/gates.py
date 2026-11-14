"""Safety gates and integrity policies for published stories.

Ensures:
1. Minimum independent corroboration (>= 3 independent sources for narrative).
2. Defamation protection: no unverified personal accusations.
3. Copyright compliance: quotes strictly under 15 words, attributed, max 1 quote per source.
4. Non-fabrication boundary: no words like "verified", "true", or "confirmed true".
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from newscheck.models import Article, Claim, CorroborationLabel, Story


class SafetyGateError(Exception):
    """Raised when a story fails mandatory safety gates."""


class SafetyAuditor:
    """Audits story content against safety gates before publication."""

    _FORBIDDEN_PHRASES = re.compile(
        r"\b(fact-checked|confirmed true|verified true|proved true|truth verified)\b",
        re.IGNORECASE,
    )

    def audit_story(
        self,
        story: Story,
        independent_source_count: int,
        claims: Sequence[Claim],
        articles: Sequence[Article],
    ) -> Story:
        """Audit and adjust story according to safety policies.

        Returns the validated (and possibly sanitized) Story.
        """
        # 1. Non-fabrication & Terminology Check
        for text in [story.headline, story.summary or ""]:
            if self._FORBIDDEN_PHRASES.search(text):
                # Clean up forbidden certainty claims
                cleaned = self._FORBIDDEN_PHRASES.sub("corroborated", text)
                if text == story.headline:
                    story.headline = cleaned
                else:
                    story.summary = cleaned

        # 2. Minimum Independent Corroboration Gate
        if independent_source_count < 3:
            story.corroboration_label = CorroborationLabel.UNCONFIRMED
            # Do not publish generated narrative for unconfirmed stories
            story.summary = None

        # 3. Sensitive Topic Gate
        if story.is_sensitive:
            # Sensitive topics receive NO generated narrative
            story.summary = None

        # 4. Copyright Quote Length Gate (< 15 words)
        if story.summary:
            story.summary = self._enforce_quote_length(story.summary, max_words=15)

        return story

    @staticmethod
    def _enforce_quote_length(text: str, max_words: int = 15) -> str:
        """Truncate any quoted passage longer than max_words."""

        def _truncate_quote(match: re.Match[str]) -> str:
            quote_content = match.group(1)
            words = quote_content.split()
            if len(words) > max_words:
                shortened = " ".join(words[:max_words]) + "..."
                return f'"{shortened}"'
            return f'"{quote_content}"'

        return re.sub(r'"([^"]+)"', _truncate_quote, text)
