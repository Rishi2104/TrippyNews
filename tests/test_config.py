"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from newscheck.config import load_settings, load_sources_yaml


class TestLoadSettings:
    """Tests for load_settings()."""

    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        """If settings.toml is absent, all defaults are used."""
        settings = load_settings(project_root=tmp_path)
        assert settings.pipeline.stories_per_run == 10
        assert settings.ingest.request_timeout_seconds == 10
        assert settings.cluster.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_loads_from_file(self, project_root: Path) -> None:
        """Test settings loaded from the test project root."""
        settings = load_settings(project_root=project_root)
        assert settings.pipeline.stories_per_run == 5
        assert settings.pipeline.min_sources_to_publish == 2

    def test_settings_are_frozen(self, project_root: Path) -> None:
        """Settings should be immutable."""
        settings = load_settings(project_root=project_root)
        with pytest.raises((TypeError, ValueError)):
            settings.pipeline.stories_per_run = 99  # type: ignore[misc]


class TestLoadSourcesYaml:
    """Tests for load_sources_yaml()."""

    def test_loads_sources(self, project_root: Path) -> None:
        """Test loading sources from the test YAML file."""
        sources = load_sources_yaml(project_root / "sources.yaml")
        assert len(sources) == 3
        assert sources[0]["name"] == "Test Source A"
        assert sources[0]["tier"] == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_sources_yaml(tmp_path / "nonexistent.yaml")

    def test_empty_sources_raises(self, tmp_path: Path) -> None:
        """An empty sources list should raise ValueError."""
        (tmp_path / "empty.yaml").write_text("sources: []", encoding="utf-8")
        with pytest.raises(ValueError, match="No sources defined"):
            load_sources_yaml(tmp_path / "empty.yaml")

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        """YAML without 'sources' key should raise ValueError."""
        (tmp_path / "bad.yaml").write_text("feeds:\n  - name: X", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a 'sources' key"):
            load_sources_yaml(tmp_path / "bad.yaml")
