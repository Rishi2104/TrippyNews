"""Wire-service credit detection.

Detects when an article is a reprint of a wire story (PTI, ANI, Reuters,
AP, AFP, etc.) by inspecting bylines, article text, and metadata.

Wire reprints from different outlets count as ONE independent source,
not multiple.
"""

from __future__ import annotations

import re

from newscheck.log import get_logger
from newscheck.models import Article, DedupGroup, DedupMethod

logger = get_logger(__name__)

# ── Wire-service patterns ──────────────────────────────────────────────────────
#
# Each tuple: (canonical_name, list_of_regex_patterns).
# Patterns are case-insensitive and tested against byline, first 500 chars
# of body text, and the author field.

_WIRE_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "PTI",
        [
            re.compile(r"\bPTI\b"),
            re.compile(r"\bPress\s+Trust\s+of\s+India\b", re.IGNORECASE),
        ],
    ),
    (
        "ANI",
        [
            re.compile(r"\bANI\b"),
            re.compile(r"\bAsian\s+News\s+International\b", re.IGNORECASE),
        ],
    ),
    (
        "Reuters",
        [
            re.compile(r"\bReuters\b", re.IGNORECASE),
        ],
    ),
    (
        "AP",
        [
            re.compile(r"\bAssociated\s+Press\b", re.IGNORECASE),
            re.compile(r"\b(?<!\w)AP(?!\w)"),  # standalone "AP"
        ],
    ),
    (
        "AFP",
        [
            re.compile(r"\bAFP\b"),
            re.compile(r"\bAgence\s+France[- ]Presse\b", re.IGNORECASE),
        ],
    ),
    (
        "IANS",
        [
            re.compile(r"\bIANS\b"),
            re.compile(r"\bIndo[- ]Asian\s+News\s+Service\b", re.IGNORECASE),
        ],
    ),
    (
        "UNI",
        [
            re.compile(r"\bUNI\b"),
            re.compile(r"\bUnited\s+News\s+of\s+India\b", re.IGNORECASE),
        ],
    ),
    (
        "PA",
        [
            re.compile(r"\bPA\s+Media\b", re.IGNORECASE),
            re.compile(r"\bPress\s+Association\b", re.IGNORECASE),
        ],
    ),
    (
        "DPA",
        [
            re.compile(r"\bdpa\b", re.IGNORECASE),
        ],
    ),
    (
        "Xinhua",
        [
            re.compile(r"\bXinhua\b", re.IGNORECASE),
        ],
    ),
]


class WireDetector:
    """Detects wire-service credits in articles.

    Usage::

        detector = WireDetector()
        wire = detector.detect(article)
        # wire is e.g. "PTI" or None
    """

    def detect(self, article: Article) -> str | None:
        """Return the canonical wire name if the article is a wire reprint.

        Checks the author field, the first 500 characters of body text,
        and the title.

        Returns:
            The canonical wire name (e.g. ``"PTI"``, ``"Reuters"``),
            or ``None`` if no wire credit was found.
        """
        # Assemble text to search.
        search_parts: list[str] = []
        if article.author:
            search_parts.append(article.author)
        if article.full_text:
            search_parts.append(article.full_text[:500])
        if article.title:
            search_parts.append(article.title)

        search_text = " ".join(search_parts)
        if not search_text:
            return None

        for wire_name, patterns in _WIRE_PATTERNS:
            for pattern in patterns:
                if pattern.search(search_text):
                    logger.debug(
                        "wire_detected",
                        article_url=article.url,
                        wire=wire_name,
                    )
                    return wire_name

        return None

    def annotate_articles(self, articles: list[Article]) -> list[Article]:
        """Set ``wire_origin`` on each article in-place and return the list."""
        for article in articles:
            if article.wire_origin is None:
                article.wire_origin = self.detect(article)
        return articles

    def build_wire_dedup_groups(self, articles: list[Article]) -> list[DedupGroup]:
        """Group wire-credited articles that share the same wire origin.

        Within each wire group, articles are further checked for topic
        similarity via title overlap before grouping — two PTI articles
        on different topics should NOT be grouped.

        Returns:
            DedupGroup entries mapping wire copies to a canonical article.
        """
        # Group by wire origin.
        wire_buckets: dict[str, list[Article]] = {}
        for art in articles:
            if art.wire_origin and art.id is not None:
                wire_buckets.setdefault(art.wire_origin, []).append(art)

        groups: list[DedupGroup] = []
        for _wire_name, bucket in wire_buckets.items():
            if len(bucket) < 2:
                continue

            # Within a wire bucket, sub-group by title similarity.
            sub_groups = self._cluster_by_title(bucket)
            for sg in sub_groups:
                if len(sg) < 2:
                    continue
                # Canonical = the one with the lowest article id (deterministic).
                sg.sort(key=lambda a: a.id or 0)
                canonical = sg[0]
                for dup in sg[1:]:
                    groups.append(
                        DedupGroup(
                            article_id=dup.id or 0,
                            canonical_id=canonical.id or 0,
                            similarity=1.0,
                            method=DedupMethod.WIRE,
                        )
                    )

        logger.info(
            "wire_dedup_complete",
            wire_groups=len(wire_buckets),
            duplicates=len(groups),
        )
        return groups

    @staticmethod
    def _cluster_by_title(articles: list[Article]) -> list[list[Article]]:
        """Sub-group articles by title word overlap.

        Articles from the same wire service sharing key topical words or stems
        (or with overlap >= 30%) belong to the same syndicated wire story.
        Uses simple Union-Find.
        """
        # Extract stems (4-char prefix of words > 3 chars) per article
        title_stems: dict[int, set[str]] = {}
        for art in articles:
            stems = {w.lower()[:4] for w in re.findall(r"\w+", art.title) if len(w) > 3}
            title_stems[art.id or 0] = stems

        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra > rb:
                    ra, rb = rb, ra
                parent[rb] = ra

        ids = list(title_stems.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                s_a = title_stems[ids[i]]
                s_b = title_stems[ids[j]]
                if not s_a or not s_b:
                    continue
                common = s_a & s_b
                overlap = len(common) / min(len(s_a), len(s_b))
                # 2 or more shared topical stems or >= 30% overlap
                if len(common) >= 2 or overlap >= 0.30:
                    union(ids[i], ids[j])

        # Collect groups.
        group_map: dict[int, list[Article]] = {}
        art_by_id = {(a.id or 0): a for a in articles}
        for aid in ids:
            root = find(aid)
            group_map.setdefault(root, []).append(art_by_id[aid])

        return list(group_map.values())
