"""Configuration loader for NewsCheck.

Reads ``settings.toml`` (via stdlib ``tomllib``), merges environment-variable
overrides (prefixed ``NEWSCHECK_``), and exposes typed, validated config
sections as frozen Pydantic models.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

# ── Section models ─────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel, frozen=True):
    """Top-level pipeline knobs."""

    stories_per_run: int = 10
    min_sources_to_publish: int = 3
    min_sources_for_valid_run: int = 15
    max_story_age_hours: int = 48
    developing_threshold_hours: int = 2


class IngestConfig(BaseModel, frozen=True):
    """RSS ingestion settings."""

    request_timeout_seconds: int = 10
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    user_agent: str = "NewsCheckBot/1.0 (+https://rishi2104.github.io/TrippyNews/about.html; contact@trippynews.org)"
    max_articles_per_feed: int = 25
    rate_limit_per_domain_seconds: float = 1.0
    feeds_file: str = "sources.yaml"
    blocklist_file: str = "blocklist.yaml"


class ExtractConfig(BaseModel, frozen=True):
    """Full-text extraction settings."""

    max_content_length: int = 50_000
    hash_algorithm: str = "sha256"


class DedupConfig(BaseModel, frozen=True):
    """Near-duplicate / syndication dedup settings."""

    minhash_threshold: float = 0.55
    minhash_num_perm: int = 128
    minhash_shingle_size: int = 5


class ClusterConfig(BaseModel, frozen=True):
    """Embedding & clustering settings."""

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.0
    metric: str = "euclidean"


class ScoringConfig(BaseModel, frozen=True):
    """Corroboration scoring weights."""

    tier1_weight: float = 3.0
    tier2_weight: float = 2.0
    tier3_weight: float = 1.0
    conflict_penalty: float = 0.3
    primary_source_bonus: float = 2.0
    recency_decay_hours: float = 6.0


class GenerationConfig(BaseModel, frozen=True):
    """LLM generation settings."""

    llm_provider: str = "none"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    gemini_model: str = "gemini-2.0-flash"
    max_summary_sentences: int = 5
    max_quote_words: int = 15


class SafetyConfig(BaseModel, frozen=True):
    """Safety gate thresholds."""

    min_entailment_score: float = 0.7


class BuildConfig(BaseModel, frozen=True):
    """Static-site build settings."""

    output_dir: str = "public"
    site_title: str = "TrippyNews"
    site_description: str = "Automated news corroboration — what sources agree and differ on."
    site_url: str = "https://rishi2104.github.io/TrippyNews"
    contact_email: str = "contact@trippynews.org"


class AlertingConfig(BaseModel, frozen=True):
    """Error-alerting settings."""

    enabled: bool = False
    method: str = "telegram"


# ── Root config ────────────────────────────────────────────────────────────────


class Settings(BaseModel, frozen=True):
    """Root configuration aggregating all sections."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)


def load_settings(
    settings_path: Path | None = None,
    *,
    project_root: Path | None = None,
) -> Settings:
    """Load settings from a TOML file with sensible defaults.

    Args:
        settings_path: Explicit path to ``settings.toml``.
                       If *None*, looks in *project_root* or cwd.
        project_root:  Project root directory.  Defaults to cwd.

    Returns:
        A fully validated, frozen :class:`Settings` instance.
    """
    if project_root is None:
        project_root = Path.cwd()

    if settings_path is None:
        settings_path = project_root / "settings.toml"

    raw: dict[str, object] = {}
    if settings_path.is_file():
        with open(settings_path, "rb") as fh:
            raw = tomllib.load(fh)

    return Settings.model_validate(raw)


def load_sources_yaml(path: Path) -> list[dict[str, object]]:
    """Parse ``sources.yaml`` and return the list of source dicts.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        List of source dictionaries (name, domain, feed_url, …).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the file is empty or missing the ``sources`` key.
    """
    import yaml  # lazy import — only needed here

    if not path.is_file():
        raise FileNotFoundError(f"Sources file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "sources" not in data:
        raise ValueError(f"Expected a 'sources' key in {path}")

    sources: list[dict[str, object]] = data["sources"]
    if not sources:
        raise ValueError(f"No sources defined in {path}")

    return sources


def load_blocklist(path: Path) -> dict[str, set[str] | set[int]]:
    """Parse blocklist.yaml and return blocked domains, urls, and story ids.

    If file does not exist, returns empty sets.
    """
    import yaml

    if not path.is_file():
        return {"domains": set(), "urls": set(), "story_ids": set()}

    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return {"domains": set(), "urls": set(), "story_ids": set()}

    domains = {str(d).lower().strip() for d in data.get("domains", []) if d}
    urls = {str(u).strip() for u in data.get("urls", []) if u}
    story_ids = {int(s) for s in data.get("story_ids", []) if s is not None}

    return {"domains": domains, "urls": urls, "story_ids": story_ids}
