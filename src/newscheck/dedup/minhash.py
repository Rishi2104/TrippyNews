"""MinHash-based near-duplicate article detection.

Uses datasketch's MinHash LSH to find article pairs with high
Jaccard similarity on character-level shingles, then groups them
so copies of the same underlying text count as ONE independent source.
"""

from __future__ import annotations

import contextlib

from datasketch import MinHash, MinHashLSH

from newscheck.config import DedupConfig
from newscheck.log import get_logger
from newscheck.models import Article, DedupGroup, DedupMethod

logger = get_logger(__name__)


class MinHashDeduplicator:
    """Find near-duplicate articles using MinHash LSH.

    Articles with Jaccard similarity above the configured threshold
    are grouped together.  The first article encountered in each group
    becomes the "canonical" representative.

    Usage::

        deduper = MinHashDeduplicator(dedup_config)
        groups = deduper.find_duplicates(articles)
    """

    def __init__(self, config: DedupConfig) -> None:
        self._threshold = config.minhash_threshold
        self._num_perm = config.minhash_num_perm
        self._shingle_size = config.minhash_shingle_size

    def find_duplicates(self, articles: list[Article]) -> list[DedupGroup]:
        """Identify near-duplicate articles and return dedup groupings.

        Args:
            articles: Articles to compare.  Each must have a non-None ``id``
                      and either ``full_text`` or ``title``.

        Returns:
            A list of :class:`DedupGroup` entries mapping duplicate article ids
            to their canonical (first-seen) article id.  Articles that are
            not duplicates of anything are **not** included.
        """
        if len(articles) < 2:
            return []

        # Build MinHash signatures.
        signatures: dict[int, MinHash] = {}
        for art in articles:
            if art.id is None:
                continue
            text = art.full_text or art.title
            sig = self._build_minhash(text)
            signatures[art.id] = sig

        if not signatures:
            return []

        # Build LSH index.
        lsh = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
        for art_id, sig in signatures.items():
            with contextlib.suppress(ValueError):
                lsh.insert(str(art_id), sig)

        # Query each article to find its neighbours.
        # Union-Find to group transitive duplicates.
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # Smaller id becomes the canonical (deterministic).
                if ra > rb:
                    ra, rb = rb, ra
                parent[rb] = ra

        seen_pairs: set[tuple[int, int]] = set()
        for art_id, sig in signatures.items():
            neighbours = lsh.query(sig)
            for nbr_key in neighbours:
                nbr_id = int(nbr_key)
                if nbr_id == art_id:
                    continue
                pair = (min(art_id, nbr_id), max(art_id, nbr_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                union(art_id, nbr_id)

        # Build DedupGroup entries.
        groups: list[DedupGroup] = []
        # Compute actual similarity for each pair for transparency.
        sim_cache: dict[tuple[int, int], float] = {}
        for a_id, b_id in seen_pairs:
            sim = signatures[a_id].jaccard(signatures[b_id])
            sim_cache[(a_id, b_id)] = sim

        for art_id in signatures:
            canonical = find(art_id)
            if canonical != art_id:
                pair_key = (min(art_id, canonical), max(art_id, canonical))
                sim = sim_cache.get(pair_key, self._threshold)
                groups.append(
                    DedupGroup(
                        article_id=art_id,
                        canonical_id=canonical,
                        similarity=sim,
                        method=DedupMethod.MINHASH,
                    )
                )

        logger.info(
            "minhash_dedup_complete",
            articles=len(articles),
            duplicates=len(groups),
        )
        return groups

    def _build_minhash(self, text: str) -> MinHash:
        """Build a MinHash signature from character-level shingles."""
        m = MinHash(num_perm=self._num_perm)
        text_lower = text.lower()
        for i in range(len(text_lower) - self._shingle_size + 1):
            shingle = text_lower[i : i + self._shingle_size]
            m.update(shingle.encode("utf-8"))
        # Handle very short texts (shorter than shingle size).
        if len(text_lower) < self._shingle_size:
            m.update(text_lower.encode("utf-8"))
        return m


def compute_similarity(text_a: str, text_b: str, *, num_perm: int = 128, shingle_size: int = 5) -> float:
    """Compute the Jaccard similarity between two texts using MinHash.

    Utility function for testing and debugging.
    """

    def _build(text: str) -> MinHash:
        m = MinHash(num_perm=num_perm)
        t = text.lower()
        for i in range(len(t) - shingle_size + 1):
            m.update(t[i : i + shingle_size].encode("utf-8"))
        if len(t) < shingle_size:
            m.update(t.encode("utf-8"))
        return m

    return float(_build(text_a).jaccard(_build(text_b)))
