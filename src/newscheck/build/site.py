"""Static site generator for NewsCheck using Jinja2.

Implements atomic site publishing and last-good-site rollback:
- Renders site pages into a staging directory.
- On complete success, atomically swaps staging to the target public/ directory.
- If any rendering error occurs, the previous live site is left untouched.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from newscheck.config import BuildConfig, Settings
from newscheck.db import get_articles_for_cluster
from newscheck.log import get_logger
from newscheck.models import (
    Claim,
    ClaimLabel,
    CorroborationLabel,
    Story,
    StoryChangelog,
)

logger = get_logger(__name__)


class SiteGenerator:
    """Renders the static website from SQLite database records."""

    def __init__(self, settings: Settings, project_root: Path) -> None:
        self._cfg: BuildConfig = settings.build
        self._root = project_root
        self._templates_dir = project_root / "templates"
        self._static_dir = project_root / "static"
        self._env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            autoescape=True,
        )
        # Register IST timezone filter for templates
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        self._env.filters["to_ist"] = lambda dt: (
            dt.replace(tzinfo=UTC).astimezone(ist_tz)
            if dt and dt.tzinfo is None
            else dt.astimezone(ist_tz)
            if dt
            else dt
        )

    def build_site(self, conn: sqlite3.Connection, *, staging_dir: Path | None = None) -> Path:
        """Render the complete static site atomically.

        Returns path to the live output directory.
        """
        output_dir = self._root / self._cfg.output_dir
        if staging_dir is None:
            staging_dir = self._root / f"{self._cfg.output_dir}_staging"

        # Clean staging dir
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Copy static assets (CSS, etc.)
            if self._static_dir.exists():
                for item in self._static_dir.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, staging_dir / item.name)

            # 2. Fetch published stories (latest on top, up to 100)
            stories_data = self._load_stories_data(conn, limit=100)

            # 3. Render paginated index pages (10 stories per page)
            page_size = 10
            total_stories = len(stories_data)
            total_pages = max(1, (total_stories + page_size - 1) // page_size)

            for page_num in range(1, total_pages + 1):
                start = (page_num - 1) * page_size
                end = min(start + page_size, total_stories)
                page_stories = stories_data[start:end]

                has_prev = page_num > 1
                prev_url = "index.html" if page_num == 2 else f"page_{page_num - 1}.html"
                has_next = page_num < total_pages
                next_url = f"page_{page_num + 1}.html"

                page_numbers = [
                    {
                        "number": i,
                        "url": "index.html" if i == 1 else f"page_{i}.html",
                        "is_current": (i == page_num),
                    }
                    for i in range(1, total_pages + 1)
                ]

                # Render index.html (and page_1.html alias) for page 1, or page_X.html
                dest_files = (
                    [staging_dir / "index.html", staging_dir / "page_1.html"]
                    if page_num == 1
                    else [staging_dir / f"page_{page_num}.html"]
                )
                for dest_path in dest_files:
                    self._render_page(
                        "index.html",
                        dest_path,
                        stories=page_stories,
                        current_page=page_num,
                        total_pages=total_pages,
                        total_stories=total_stories,
                        start_index=start + 1 if total_stories > 0 else 0,
                        end_index=end,
                        has_prev=has_prev,
                        has_next=has_next,
                        prev_url=prev_url,
                        next_url=next_url,
                        page_numbers=page_numbers,
                        site_title=self._cfg.site_title,
                        site_description=self._cfg.site_description,
                    )

            # 4. Render story details pages
            for s_data in stories_data:
                story = s_data["story"]
                self._render_page(
                    "story.html",
                    staging_dir / f"story_{story.id}.html",
                    story=story,
                    articles=s_data["articles"],
                    agreed_claims=s_data["agreed_claims"],
                    disputed_claims=s_data["disputed_claims"],
                    single_source_claims=s_data["single_source_claims"],
                    independent_source_count=s_data["independent_source_count"],
                    changelog=s_data["changelog"],
                    site_title=self._cfg.site_title,
                    site_description=self._cfg.site_description,
                )

            # 5. Render methodology.html
            self._render_page(
                "methodology.html",
                staging_dir / "methodology.html",
                site_title=self._cfg.site_title,
                site_description=self._cfg.site_description,
            )

            # 6. Render corrections.html
            changelogs = self._load_changelogs(conn)
            self._render_page(
                "corrections.html",
                staging_dir / "corrections.html",
                changelogs=changelogs,
                site_title=self._cfg.site_title,
                site_description=self._cfg.site_description,
            )

            # 7. Atomic deployment swap
            if output_dir.exists():
                backup_dir = self._root / f"{self._cfg.output_dir}_backup"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                output_dir.rename(backup_dir)
                staging_dir.rename(output_dir)
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
            else:
                staging_dir.rename(output_dir)

            logger.info("static_site_published", output_dir=str(output_dir))
            return output_dir

        except Exception:
            logger.exception("site_build_failed_preserving_last_good_site")
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _render_page(self, template_name: str, dest_path: Path, **context: Any) -> None:
        template = self._env.get_template(template_name)
        html = template.render(**context)
        dest_path.write_text(html, encoding="utf-8")

    def _load_stories_data(self, conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
        """Load stories with associated claims, articles, and changelogs.

        Ordered by published_at DESC (latest on top), then by corroboration_score DESC.
        """
        rows = conn.execute(
            """
            SELECT * FROM stories
            ORDER BY published_at DESC, corroboration_score DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            story = Story(
                id=row["id"],
                cluster_id=row["cluster_id"],
                headline=row["headline"],
                summary=row["summary"],
                corroboration_score=row["corroboration_score"],
                corroboration_label=CorroborationLabel(row["corroboration_label"]),
                is_sensitive=bool(row["is_sensitive"]),
                is_developing=bool(row["is_developing"]),
                published_at=datetime.fromisoformat(row["published_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            )

            articles = get_articles_for_cluster(conn, story.cluster_id, with_text=False)
            unique_sources = {a.source_id for a in articles}

            # Fetch claims for this cluster
            claims_rows = conn.execute(
                "SELECT * FROM claims WHERE cluster_id = ?",
                (story.cluster_id,),
            ).fetchall()
            claims = [
                Claim(
                    id=cr["id"],
                    article_id=cr["article_id"],
                    cluster_id=cr["cluster_id"],
                    claim_text=cr["claim_text"],
                    source_sentence=cr["source_sentence"],
                    claim_type=cr["claim_type"],
                )
                for cr in claims_rows
            ]

            # Fetch cross checks
            cross_rows = conn.execute(
                """
                SELECT * FROM cross_checks
                WHERE claim_a_id IN (SELECT id FROM claims WHERE cluster_id = ?)
                """,
                (story.cluster_id,),
            ).fetchall()

            agreed_ids: set[int] = set()
            disputed_ids: set[int] = set()
            for cx in cross_rows:
                lbl = cx["label"]
                if lbl == ClaimLabel.AGREED.value:
                    agreed_ids.add(cx["claim_a_id"])
                    agreed_ids.add(cx["claim_b_id"])
                elif lbl == ClaimLabel.DISPUTED.value:
                    disputed_ids.add(cx["claim_a_id"])
                    disputed_ids.add(cx["claim_b_id"])

            agreed_claims = [c for c in claims if c.id in agreed_ids]
            disputed_claims = [c for c in claims if c.id in disputed_ids]
            single_source_claims = [c for c in claims if c.id not in agreed_ids and c.id not in disputed_ids]

            # Changelogs
            changelogs = self._load_story_changelogs(conn, story.id or 0)

            # Sources info
            src_objs: list[dict[str, str]] = []
            for art in articles:
                src_row = conn.execute("SELECT name, domain FROM sources WHERE id = ?", (art.source_id,)).fetchone()
                if src_row:
                    src_objs.append(
                        {
                            "name": src_row["name"],
                            "domain": src_row["domain"],
                            "url": art.url,
                        }
                    )

            results.append(
                {
                    "story": story,
                    "articles": articles,
                    "sources": src_objs,
                    "independent_source_count": len(unique_sources),
                    "agreed_claims": agreed_claims,
                    "disputed_claims": disputed_claims,
                    "single_source_claims": single_source_claims,
                    "changelog": changelogs,
                }
            )

        return results

    @staticmethod
    def _load_changelogs(conn: sqlite3.Connection) -> list[StoryChangelog]:
        rows = conn.execute("SELECT * FROM story_changelog ORDER BY changed_at DESC LIMIT 50").fetchall()
        return [
            StoryChangelog(
                id=r["id"],
                story_id=r["story_id"],
                field_changed=r["field_changed"],
                old_value=r["old_value"],
                new_value=r["new_value"],
                reason=r["reason"],
                changed_at=datetime.fromisoformat(r["changed_at"]),
            )
            for r in rows
        ]

    @staticmethod
    def _load_story_changelogs(conn: sqlite3.Connection, story_id: int) -> list[StoryChangelog]:
        rows = conn.execute(
            "SELECT * FROM story_changelog WHERE story_id = ? ORDER BY changed_at DESC",
            (story_id,),
        ).fetchall()
        return [
            StoryChangelog(
                id=r["id"],
                story_id=r["story_id"],
                field_changed=r["field_changed"],
                old_value=r["old_value"],
                new_value=r["new_value"],
                reason=r["reason"],
                changed_at=datetime.fromisoformat(r["changed_at"]),
            )
            for r in rows
        ]
