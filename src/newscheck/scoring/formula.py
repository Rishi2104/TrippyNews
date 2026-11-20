"""Corroboration scoring formula and label mapping.

All scoring rules and label thresholds are centralized in this single module.
The site never claims a story is "verified" or "true"; it reports
"Corroborated by N independent sources" with a calibrated label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from newscheck.config import ScoringConfig
from newscheck.models import CorroborationLabel, ReliabilityTier, Source

# ── Single Source of Truth for Scoring & Methodology (L12) ───────────────────
SCORING_CONSTANTS: dict[str, float | int | bool] = {
    "tier1_weight": 3.0,
    "tier2_weight": 2.0,
    "tier3_weight": 1.0,
    "primary_source_bonus": 2.0,
    "conflict_penalty": 0.3,
    "recency_decay_hours": 6.0,
    "strongly_min_sources": 4,
    "strongly_min_score": 8.0,
    "strongly_requires_tier1": True,
    "corroborated_min_sources": 3,
    "corroborated_min_score": 4.5,
    "min_sources_for_narrative": 3,
    "min_entailed_sentences_for_narrative": 2,
    "max_quote_words": 25,
}


@dataclass(frozen=True)
class StoryScoreBreakdown:
    """Detailed components explaining how the corroboration score was calculated."""

    base_score: float
    primary_bonus: float
    dispute_penalty: float
    recency_factor: float
    final_score: float
    independent_sources: int
    label: CorroborationLabel
    is_developing: bool


def compute_corroboration_score(
    independent_sources: Sequence[Source],
    *,
    disputed_claim_count: int = 0,
    oldest_publication: datetime | None = None,
    newest_publication: datetime | None = None,
    config: ScoringConfig,
    now: datetime | None = None,
) -> StoryScoreBreakdown:
    """Compute transparent corroboration score and label for a story cluster.

    Formula:
      1. Base score = sum(tier_weight for each independent source)
         - Tier 1 (Wire/Official): 3.0
         - Tier 2 (Major Reputable): 2.0
         - Tier 3 (Other Whitelisted): 1.0
      2. Primary / Official source bonus:
         - +2.0 if at least one Tier 1 source is present.
      3. Conflict / dispute penalty:
         - Reduces score by (disputed_claim_count * conflict_penalty)
      4. Recency factor:
         - Slowly decays older stories beyond recency_decay_hours.
      5. Mandatory Safety Gate:
         - Stories with fewer than 3 independent sources are ALWAYS labelled "Unconfirmed".
      6. Developing Badge:
         - Stories first reported under 2 hours ago are flagged is_developing = True.

    Args:
        independent_sources: Unique independent sources covering the story.
        disputed_claim_count: Number of contradictory claim pairs detected.
        oldest_publication: Timestamp of the earliest article in the cluster.
        newest_publication: Timestamp of the latest article in the cluster.
        config: Weights and penalty configuration.
        now: Current UTC timestamp (defaults to datetime.now(timezone.utc)).

    Returns:
        A StoryScoreBreakdown with numerical components and final label.
    """
    if now is None:
        now = datetime.now(UTC)

    num_sources = len(independent_sources)

    # 1. Base tier score
    tier_weights = {
        ReliabilityTier.WIRE_OFFICIAL: config.tier1_weight,
        ReliabilityTier.MAJOR_OUTLET: config.tier2_weight,
        ReliabilityTier.OTHER_WHITELISTED: config.tier3_weight,
    }
    base_score = sum(tier_weights.get(s.tier, 1.0) for s in independent_sources)

    # 2. Primary / Official bonus
    has_tier1 = any(s.tier == ReliabilityTier.WIRE_OFFICIAL for s in independent_sources)
    primary_bonus = config.primary_source_bonus if has_tier1 else 0.0

    # 3. Dispute penalty
    dispute_penalty = disputed_claim_count * config.conflict_penalty

    # 4. Recency decay
    recency_factor = 1.0
    if newest_publication is not None:
        age_hours = max(0.0, (now - newest_publication).total_seconds() / 3600.0)
        recency_factor = max(0.2, 1.0 - (age_hours / (config.recency_decay_hours * 10.0)))

    # Raw score calculation
    raw_score = (base_score + primary_bonus - dispute_penalty) * recency_factor
    final_score = round(max(0.0, raw_score), 2)

    # 5. Label mapping with safety constraints
    if num_sources < 3:
        label = CorroborationLabel.UNCONFIRMED
    elif disputed_claim_count > 0 and final_score < 5.0:
        label = CorroborationLabel.LIMITED
    elif final_score >= 8.0 and num_sources >= 4 and has_tier1:
        label = CorroborationLabel.STRONGLY_CORROBORATED
    elif final_score >= 4.5 and num_sources >= 3:
        label = CorroborationLabel.CORROBORATED
    else:
        label = CorroborationLabel.LIMITED

    # 6. Developing flag (under 2 hours old)
    is_developing = False
    ref_time = oldest_publication or newest_publication
    if ref_time is not None:
        age_hours = max(0.0, (now - ref_time).total_seconds() / 3600.0)
        is_developing = age_hours <= 2.0

    return StoryScoreBreakdown(
        base_score=round(base_score, 2),
        primary_bonus=primary_bonus,
        dispute_penalty=round(dispute_penalty, 2),
        recency_factor=round(recency_factor, 2),
        final_score=final_score,
        independent_sources=num_sources,
        label=label,
        is_developing=is_developing,
    )
