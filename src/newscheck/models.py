"""Domain models for the NewsCheck pipeline.

All data structures that flow between pipeline stages are defined here.
Uses standard-library dataclasses to avoid coupling domain models to any ORM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum, StrEnum

# ── Enums ──────────────────────────────────────────────────────────────────────


class ReliabilityTier(IntEnum):
    """Source reliability tier. Lower = more authoritative."""

    WIRE_OFFICIAL = 1
    MAJOR_OUTLET = 2
    OTHER_WHITELISTED = 3


class ExtractionStatus(StrEnum):
    """Outcome of full-text extraction for an article."""

    FULL = "full"
    LIMITED = "limited"
    FAILED = "failed"


class CorroborationLabel(StrEnum):
    """Human-readable corroboration level for a published story."""

    STRONGLY_CORROBORATED = "Strongly corroborated"
    CORROBORATED = "Corroborated"
    LIMITED = "Limited corroboration"
    UNCONFIRMED = "Unconfirmed"


class ClaimLabel(StrEnum):
    """Cross-check verdict for a claim across sources."""

    AGREED = "agreed"
    DISPUTED = "disputed"
    SINGLE_SOURCE = "single_source"


class RunStatus(StrEnum):
    """Pipeline run lifecycle status."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class DedupMethod(StrEnum):
    """How two articles were identified as near-duplicates."""

    MINHASH = "minhash"
    WIRE = "wire"
    BOTH = "both"


# ── Data models ────────────────────────────────────────────────────────────────


@dataclass
class Source:
    """A curated news source from sources.yaml."""

    name: str
    domain: str
    feed_url: str
    language: str
    country: str
    tier: ReliabilityTier
    enabled: bool = True
    id: int | None = None


@dataclass
class Article:
    """An ingested and extracted news article."""

    source_id: int
    url: str
    title: str
    content_hash: str
    extraction_status: ExtractionStatus
    run_id: int
    snippet: str = ""
    full_text: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    wire_origin: str | None = None
    id: int | None = None


@dataclass
class DedupGroup:
    """Maps a duplicate article to its canonical (first-seen) article."""

    article_id: int
    canonical_id: int
    similarity: float
    method: DedupMethod


@dataclass
class Cluster:
    """A group of articles about the same news event."""

    run_id: int
    rank: int
    label: str
    article_count: int = 0
    independent_source_count: int = 0
    tier_weighted_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: int | None = None


@dataclass
class ClusterArticle:
    """Association between a cluster and an article."""

    cluster_id: int
    article_id: int


@dataclass
class Claim:
    """A discrete factual claim extracted from an article (Phase 2)."""

    article_id: int
    cluster_id: int
    claim_text: str
    source_sentence: str
    claim_type: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: int | None = None


@dataclass
class CrossCheck:
    """NLI comparison result between two claims (Phase 2)."""

    claim_a_id: int
    claim_b_id: int
    label: ClaimLabel
    nli_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: int | None = None


@dataclass
class Story:
    """A published story derived from a cluster (Phase 3+)."""

    cluster_id: int
    headline: str
    corroboration_score: float
    corroboration_label: CorroborationLabel
    published_at: datetime
    summary: str | None = None
    is_sensitive: bool = False
    is_developing: bool = False
    updated_at: datetime | None = None
    content_hash: str | None = None
    id: int | None = None


@dataclass
class StoryChangelog:
    """An entry in a story's visible change log."""

    story_id: int
    field_changed: str
    changed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    old_value: str | None = None
    new_value: str | None = None
    reason: str | None = None
    id: int | None = None


@dataclass
class Run:
    """Metadata for a single pipeline execution."""

    started_at: datetime
    status: RunStatus = RunStatus.RUNNING
    finished_at: datetime | None = None
    articles_fetched: int = 0
    clusters_formed: int = 0
    stories_published: int = 0
    error_message: str | None = None
    id: int | None = None
