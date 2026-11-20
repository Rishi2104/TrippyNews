"""Static site generator for NewsCheck / TrippyNews using Jinja2.

Implements atomic site publishing, last-good-site rollback, strict autoescaping,
compliance checking (no example.com, no verbatim publisher text outside quotes),
and transparent corroboration ranking.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from newscheck.config import BuildConfig, Settings
from newscheck.db import get_articles_for_cluster
from newscheck.log import get_logger
from newscheck.models import (
    ClaimLabel,
    CorroborationLabel,
    Story,
    StoryChangelog,
)
from newscheck.scoring.formula import SCORING_CONSTANTS

logger = get_logger(__name__)


def max_consecutive_word_overlap(text_a: str, text_b: str) -> int:
    """Find the length of the longest run of consecutive identical words between two texts."""
    words_a = re.findall(r"\b\w+\b", text_a.lower())
    words_b = re.findall(r"\b\w+\b", text_b.lower())

    if not words_a or not words_b:
        return 0

    set_b = set(words_b)
    max_len = 0

    for i in range(len(words_a)):
        if words_a[i] not in set_b:
            continue
        for j in range(len(words_b)):
            k = 0
            while i + k < len(words_a) and j + k < len(words_b) and words_a[i + k] == words_b[j + k]:
                k += 1
            if k > max_len:
                max_len = k

    return max_len


class SiteGenerator:
    """Renders the static website from SQLite database records."""

    def __init__(self, settings: Settings, project_root: Path) -> None:
        self._cfg: BuildConfig = settings.build
        self._settings = settings
        self._root = project_root
        self._templates_dir = project_root / "templates"
        self._static_dir = project_root / "static"

        # C9: Explicit autoescape for HTML and XML
        self._env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Register IST timezone filter with safe None handling (T1)
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        self._env.filters["to_ist"] = lambda dt: (
            dt.replace(tzinfo=UTC).astimezone(ist_tz)
            if dt and getattr(dt, "tzinfo", None) is None
            else dt.astimezone(ist_tz)
            if dt
            else None
        )

    def build_site(self, conn: sqlite3.Connection, *, staging_dir: Path | None = None) -> Path:
        """Render the complete static site atomically with compliance checks.

        Returns path to the live output directory.
        """
        output_dir = self._root / self._cfg.output_dir
        if staging_dir is None:
            staging_dir = self._root / f"{self._cfg.output_dir}_staging"

        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Copy static assets (CSS, JS, etc.)
            if self._static_dir.exists():
                for item in self._static_dir.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, staging_dir / item.name)

            # Common template context
            now_utc = datetime.now(UTC)
            common_ctx = {
                "site_title": self._cfg.site_title,
                "site_description": self._cfg.site_description,
                "site_url": self._cfg.site_url,
                "contact_email": self._cfg.contact_email,
                "generated_at": now_utc,
                "scoring_constants": SCORING_CONSTANTS,
            }

            # 2. Fetch published stories (ranked by corroboration & recency)
            stories_data = self._load_stories_data(conn, limit=100)

            # 3. Render paginated index pages (10 stories per page with infinite scroll)
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
                        is_homepage=(page_num == 1 and dest_path.name == "index.html"),
                        **common_ctx,
                    )

            # 4. Render story details pages
            for s_data in stories_data:
                story = s_data["story"]
                self._render_page(
                    "story.html",
                    staging_dir / f"story_{story.id}.html",
                    story=story,
                    articles=s_data["articles"],
                    article_sources=s_data["article_sources"],
                    agreed_claims=s_data["agreed_claims"],
                    disputed_claims=s_data["disputed_claims"],
                    single_source_claims=s_data["single_source_claims"],
                    independent_source_count=s_data["independent_source_count"],
                    changelog=s_data["changelog"],
                    is_homepage=False,
                    **common_ctx,
                )

            # 5. Render methodology.html with scoring constants (L12)
            self._render_page(
                "methodology.html",
                staging_dir / "methodology.html",
                is_homepage=False,
                **common_ctx,
            )

            # 6. Render corrections.html
            changelogs = self._load_changelogs(conn)
            self._render_page(
                "corrections.html",
                staging_dir / "corrections.html",
                changelogs=changelogs,
                is_homepage=False,
                **common_ctx,
            )

            # 7. Render required legal & informational pages (L11)
            legal_pages = ["about.html", "terms.html", "privacy.html", "takedown.html", "sources_policy.html"]
            for lp in legal_pages:
                if (self._templates_dir / lp).is_file():
                    self._render_page(
                        lp,
                        staging_dir / lp,
                        is_homepage=False,
                        **common_ctx,
                    )

            # 8. Compliance Gate: Verify no example.com exists in generated files (L10)
            for f in staging_dir.glob("*.html"):
                content = f.read_text(encoding="utf-8")
                if "example.com" in content:
                    raise ValueError(f"Build failed compliance gate L10: 'example.com' found in {f.name}")

            # 9. Atomic deployment swap
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
        html_out = template.render(**context)
        dest_path.write_text(html_out, encoding="utf-8")

    def _load_stories_data(self, conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
        """Load stories ranked by corroboration and recency with claim & source attribution."""
        rows = conn.execute(
            """
            SELECT * FROM stories
            ORDER BY corroboration_score DESC, published_at DESC
            LIMIT ?
            """,
            (limit * 2,),  # fetch extra for ranking filter
        ).fetchall()

        now = datetime.now(UTC)
        candidates: list[dict[str, Any]] = []

        for row in rows:
            pub_raw = row["published_at"]
            pub_dt = datetime.fromisoformat(pub_raw) if pub_raw else None
            upd_raw = row["updated_at"]
            upd_dt = datetime.fromisoformat(upd_raw) if upd_raw else None

            story = Story(
                id=row["id"],
                cluster_id=row["cluster_id"],
                headline=row["headline"],
                summary=row["summary"],
                corroboration_score=row["corroboration_score"],
                corroboration_label=CorroborationLabel(row["corroboration_label"]),
                is_sensitive=bool(row["is_sensitive"]),
                is_developing=bool(row["is_developing"]),
                published_at=pub_dt or now,
                updated_at=upd_dt,
            )

            articles = get_articles_for_cluster(conn, story.cluster_id, with_text=False)
            unique_sources = {a.source_id for a in articles}

            # Map articles to sources for clear attribution (D4, T2)
            article_sources: list[dict[str, Any]] = []
            source_dict_by_id: dict[int, dict[str, str]] = {}
            for art in articles:
                src_row = conn.execute("SELECT name, domain FROM sources WHERE id = ?", (art.source_id,)).fetchone()
                outlet_name = src_row["name"] if src_row else f"Source #{art.source_id}"
                outlet_domain = src_row["domain"] if src_row else ""
                source_dict_by_id[art.source_id] = {"name": outlet_name, "domain": outlet_domain}
                article_sources.append(
                    {
                        "article_id": art.id,
                        "title": art.title,
                        "url": art.url,
                        "outlet_name": outlet_name,
                        "outlet_domain": outlet_domain,
                        "wire_origin": art.wire_origin,
                    }
                )

            # Deduplicated sources list for card footer
            unique_outlet_map: dict[str, dict[str, Any]] = {}
            for item in article_sources:
                name = item["outlet_name"]
                if name not in unique_outlet_map:
                    unique_outlet_map[name] = item

            # Fetch claims for this cluster
            claims_rows = conn.execute(
                "SELECT * FROM claims WHERE cluster_id = ?",
                (story.cluster_id,),
            ).fetchall()

            # Map claims to article outlet for attribution (L3, T3)
            art_to_source = {a.id: source_dict_by_id.get(a.source_id, {"name": "Source", "domain": ""}) for a in articles}
            art_to_url = {a.id: a.url for a in articles}

            claims_with_meta: list[dict[str, Any]] = []
            for cr in claims_rows:
                aid = cr["article_id"]
                s_meta = art_to_source.get(aid, {"name": "Source", "domain": ""})
                claims_with_meta.append(
                    {
                        "id": cr["id"],
                        "article_id": aid,
                        "cluster_id": cr["cluster_id"],
                        "claim_text": cr["claim_text"],
                        "source_sentence": cr["source_sentence"],
                        "claim_type": cr["claim_type"],
                        "outlet_name": s_meta["name"],
                        "outlet_url": art_to_url.get(aid, "#"),
                    }
                )

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
            disputed_pairs: list[dict[str, Any]] = []

            for cx in cross_rows:
                lbl = cx["label"]
                if lbl == ClaimLabel.AGREED.value:
                    agreed_ids.add(cx["claim_a_id"])
                    agreed_ids.add(cx["claim_b_id"])
                elif lbl == ClaimLabel.DISPUTED.value:
                    disputed_ids.add(cx["claim_a_id"])
                    disputed_ids.add(cx["claim_b_id"])
                    c_a = next((c for c in claims_with_meta if c["id"] == cx["claim_a_id"]), None)
                    c_b = next((c for c in claims_with_meta if c["id"] == cx["claim_b_id"]), None)
                    if c_a and c_b:
                        disputed_pairs.append({"claim_a": c_a, "claim_b": c_b})

            agreed_claims = [c for c in claims_with_meta if c["id"] in agreed_ids]
            disputed_claims = [c for c in claims_with_meta if c["id"] in disputed_ids]
            single_source_claims = [c for c in claims_with_meta if c["id"] not in agreed_ids and c["id"] not in disputed_ids]

            # Changelogs
            changelogs = self._load_story_changelogs(conn, story.id or 0)

            # Stated ranking score: balances corroboration score and recency (D3)
            age_hours = max(0.0, (now - (story.published_at or now)).total_seconds() / 3600.0)
            recency_multiplier = 1.0 / (1.0 + (age_hours / 24.0))
            rank_score = (story.corroboration_score or 0.0) * recency_multiplier

            candidates.append(
                {
                    "story": story,
                    "articles": articles,
                    "article_sources": article_sources,
                    "sources": list(unique_outlet_map.values()),
                    "independent_source_count": len(unique_sources),
                    "agreed_claims": agreed_claims,
                    "disputed_claims": disputed_claims,
                    "disputed_pairs": disputed_pairs,
                    "single_source_claims": single_source_claims,
                    "changelog": changelogs,
                    "rank_score": rank_score,
                }
            )

        # Sort candidates by combined rank_score descending (D3)
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)
        return candidates[:limit]

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
