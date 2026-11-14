"""Unit tests for the corroboration scoring formula."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from newscheck.config import ScoringConfig
from newscheck.models import CorroborationLabel, ReliabilityTier, Source
from newscheck.scoring.formula import compute_corroboration_score


def _make_source(id_: int, tier: ReliabilityTier) -> Source:
    return Source(
        id=id_,
        name=f"Source {id_}",
        domain=f"src{id_}.com",
        feed_url=f"https://src{id_}.com/rss",
        language="en",
        country="IN",
        tier=tier,
    )


class TestScoringFormula:
    def test_minimum_sources_rule_enforced(self) -> None:
        """Fewer than 3 sources MUST be labeled Unconfirmed."""
        cfg = ScoringConfig()
        sources = [
            _make_source(1, ReliabilityTier.WIRE_OFFICIAL),
            _make_source(2, ReliabilityTier.MAJOR_OUTLET),
        ]
        res = compute_corroboration_score(sources, config=cfg)
        assert res.independent_sources == 2
        assert res.label == CorroborationLabel.UNCONFIRMED

    def test_strongly_corroborated(self) -> None:
        """4+ sources with a Tier 1 source and high score receives Strongly Corroborated."""
        cfg = ScoringConfig(tier1_weight=3.0, tier2_weight=2.0, tier3_weight=1.0, primary_source_bonus=2.0)
        sources = [
            _make_source(1, ReliabilityTier.WIRE_OFFICIAL),
            _make_source(2, ReliabilityTier.MAJOR_OUTLET),
            _make_source(3, ReliabilityTier.MAJOR_OUTLET),
            _make_source(4, ReliabilityTier.OTHER_WHITELISTED),
        ]
        # Base: 3 + 2 + 2 + 1 = 8.0, primary bonus: +2.0 = 10.0
        res = compute_corroboration_score(sources, config=cfg)
        assert res.independent_sources == 4
        assert res.final_score >= 8.0
        assert res.label == CorroborationLabel.STRONGLY_CORROBORATED

    def test_dispute_penalty_applied(self) -> None:
        """Disputed claims apply a conflict penalty."""
        cfg = ScoringConfig(conflict_penalty=2.0)
        sources = [
            _make_source(1, ReliabilityTier.MAJOR_OUTLET),
            _make_source(2, ReliabilityTier.MAJOR_OUTLET),
            _make_source(3, ReliabilityTier.MAJOR_OUTLET),
        ]
        res_clean = compute_corroboration_score(sources, disputed_claim_count=0, config=cfg)
        res_disputed = compute_corroboration_score(sources, disputed_claim_count=2, config=cfg)

        assert res_disputed.final_score < res_clean.final_score
        assert res_disputed.dispute_penalty == 4.0

    def test_developing_badge(self) -> None:
        """Stories under 2 hours old receive the Developing badge."""
        cfg = ScoringConfig()
        now = datetime.now(UTC)
        recent_pub = now - timedelta(minutes=45)
        old_pub = now - timedelta(hours=5)
        sources = [
            _make_source(1, ReliabilityTier.MAJOR_OUTLET),
            _make_source(2, ReliabilityTier.MAJOR_OUTLET),
            _make_source(3, ReliabilityTier.MAJOR_OUTLET),
        ]

        res_recent = compute_corroboration_score(sources, newest_publication=recent_pub, config=cfg, now=now)
        assert res_recent.is_developing is True

        res_old = compute_corroboration_score(sources, newest_publication=old_pub, config=cfg, now=now)
        assert res_old.is_developing is False
