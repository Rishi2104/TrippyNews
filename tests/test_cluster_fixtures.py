"""Golden cluster fixture tests for D1 data correctness.

Tests that unrelated articles linked by loose keywords are successfully pruned:
- Cauvery SC water dispute does not contain Kerala Waqf HC or Krishna basin board.
- Queen's dressmaker barn does not contain Malibu, Zuckerberg, Tom Holland, Aamir Khan.
- Texas execution does not contain Epstein or Illinois cold case.
- Siddaramaiah politics does not contain Mohammed Siraj cricket.
- Wild cat / leopard story does not contain Brooklyn building collapse cat.
- FIFA / Tashima election does not contain El Salvador Bukele third term.
- CJI hearing with 15 sources is capped at max 8 sources.
"""

from __future__ import annotations

import numpy as np

from newscheck.cluster.group import prune_cluster_articles
from newscheck.models import Article, ExtractionStatus


def _make_article(art_id: int, title: str, snippet: str = "") -> Article:
    return Article(
        id=art_id,
        source_id=art_id,
        url=f"https://source-{art_id}.test/article",
        title=title,
        snippet=snippet or title,
        full_text=snippet or title,
        content_hash=f"hash_{art_id}",
        extraction_status=ExtractionStatus.FULL,
        run_id=1,
    )


class TestGoldenClusterFixtures:
    def test_cauvery_sc_cluster_prunes_unrelated_river_and_waqf_stories(self) -> None:
        """D1: Cauvery SC water dispute must not retain Kerala Waqf HC or Krishna basin stories."""
        articles = [
            _make_article(1, "Cauvery dispute: Supreme Court refuses to stay CWMA order on water release", "Cauvery water Karnataka Tamil Nadu"),
            _make_article(2, "Supreme Court upholds CWMA directive to Karnataka on Cauvery water", "CWMA Karnataka Cauvery Tamil Nadu"),
            _make_article(3, "Cauvery water: Karnataka files compliance petition before Supreme Court", "Cauvery water release Karnataka"),
            _make_article(4, "Kerala High Court issues stay in Waqf land registration plea", "Kerala HC Waqf board property dispute"),
            _make_article(5, "Krishna river basin management board calls emergency meet on Telangana water quota", "Krishna river basin water board"),
        ]

        # Core vectors (1, 2, 3) are tight; outliers (4, 5) point in different directions
        vecs = np.zeros((5, 16), dtype=np.float32)
        vecs[0, 0:4] = [0.8, 0.7, 0.9, 0.6]
        vecs[1, 0:4] = [0.78, 0.72, 0.88, 0.62]
        vecs[2, 0:4] = [0.75, 0.68, 0.85, 0.58]
        # Waqf story
        vecs[3, 8:12] = [0.9, 0.8, 0.7, 0.6]
        # Krishna river
        vecs[4, 4:8] = [0.8, 0.8, 0.7, 0.7]

        surviving = prune_cluster_articles(
            indices=[0, 1, 2, 3, 4],
            articles=articles,
            embeddings=vecs,
            min_similarity=0.58,
            max_sources=8,
        )

        surviving_titles = [articles[i].title for i in surviving]
        assert any("Cauvery" in t for t in surviving_titles)
        assert not any("Kerala High Court" in t or "Waqf" in t for t in surviving_titles)
        assert not any("Krishna" in t for t in surviving_titles)

    def test_queens_dressmaker_barn_prunes_celebrity_real_estate(self) -> None:
        """D1: Queen's dressmaker barn conversion must not retain Malibu, Zuckerberg, Tom Holland, Aamir Khan."""
        articles = [
            _make_article(1, "Queen Elizabeth's former dressmaker secures permission for historic barn home", "Queen Elizabeth dressmaker barn Norfolk"),
            _make_article(2, "Historic Norfolk barn owned by royal dressmaker approved for conversion", "royal dressmaker barn conversion"),
            _make_article(3, "Mark Zuckerberg buys luxury $300M compound in Kauai", "Mark Zuckerberg Hawaii property"),
            _make_article(4, "Tom Holland and Zendaya purchase luxury London mansion", "Tom Holland Zendaya home London"),
            _make_article(5, "Aamir Khan leases Mumbai luxury apartment for Rs 12 lakh a month", "Aamir Khan Mumbai apartment lease"),
            _make_article(6, "Billionaire sells Malibu beach estate for record $200 million", "Malibu beach mansion estate sale"),
        ]

        vecs = np.zeros((6, 16), dtype=np.float32)
        vecs[0, 0:4] = [0.9, 0.85, 0.8, 0.9]
        vecs[1, 0:4] = [0.88, 0.82, 0.79, 0.87]
        vecs[2, 4:8] = [0.9, 0.8, 0.7, 0.6]
        vecs[3, 6:10] = [0.8, 0.7, 0.9, 0.6]
        vecs[4, 8:12] = [0.7, 0.9, 0.6, 0.8]
        vecs[5, 10:14] = [0.6, 0.7, 0.9, 0.8]

        surviving = prune_cluster_articles(
            indices=[0, 1, 2, 3, 4, 5],
            articles=articles,
            embeddings=vecs,
            min_similarity=0.58,
            max_sources=8,
        )

        surviving_titles = [articles[i].title for i in surviving]
        assert any("dressmaker" in t.lower() for t in surviving_titles)
        assert not any("Zuckerberg" in t for t in surviving_titles)
        assert not any("Tom Holland" in t for t in surviving_titles)
        assert not any("Aamir Khan" in t for t in surviving_titles)
        assert not any("Malibu" in t for t in surviving_titles)

    def test_texas_execution_prunes_epstein_and_cold_cases(self) -> None:
        """D1: Texas death row execution must not contain Epstein or Illinois cold case."""
        articles = [
            _make_article(1, "Texas executes inmate by lethal injection for 2001 convenience store murder", "Texas death row inmate lethal injection Huntsville"),
            _make_article(2, "Man put to death in Texas for 2001 robbery murder", "Texas execution death chamber Huntsville"),
            _make_article(3, "Unsealed Jeffrey Epstein flight logs reveal high profile names in court filing", "Jeffrey Epstein court documents New York"),
            _make_article(4, "DNA breakthrough solves 1984 Illinois cold case murder of teenager", "Illinois cold case DNA solved"),
        ]

        vecs = np.zeros((4, 16), dtype=np.float32)
        vecs[0, 0:4] = [0.85, 0.9, 0.8, 0.85]
        vecs[1, 0:4] = [0.82, 0.88, 0.81, 0.83]
        vecs[2, 4:8] = [0.9, 0.8, 0.7, 0.6]
        vecs[3, 8:12] = [0.7, 0.8, 0.9, 0.7]

        surviving = prune_cluster_articles(
            indices=[0, 1, 2, 3],
            articles=articles,
            embeddings=vecs,
            min_similarity=0.58,
            max_sources=8,
        )

        surviving_titles = [articles[i].title for i in surviving]
        assert any("Texas executes" in t or "death in Texas" in t for t in surviving_titles)
        assert not any("Epstein" in t for t in surviving_titles)
        assert not any("Illinois" in t for t in surviving_titles)

    def test_siddaramaiah_politics_prunes_siraj_cricket(self) -> None:
        """D1: Siddaramaiah political cluster must not contain Mohammed Siraj cricket stories."""
        articles = [
            _make_article(1, "Karnataka CM Siddaramaiah chairs state cabinet meet on development funds", "Siddaramaiah Bengaluru cabinet meet Karnataka"),
            _make_article(2, "Siddaramaiah defends state budget allocations in legislative assembly", "Siddaramaiah legislative assembly Karnataka"),
            _make_article(3, "Mohammed Siraj takes six wickets in sensational Asia Cup bowling spell", "Mohammed Siraj cricket wickets bowling spell India"),
        ]

        vecs = np.zeros((3, 16), dtype=np.float32)
        vecs[0, 0:4] = [0.9, 0.85, 0.9, 0.8]
        vecs[1, 0:4] = [0.87, 0.82, 0.88, 0.79]
        vecs[2, 6:10] = [0.9, 0.8, 0.85, 0.7]

        surviving = prune_cluster_articles(
            indices=[0, 1, 2],
            articles=articles,
            embeddings=vecs,
            min_similarity=0.58,
            max_sources=8,
        )

        surviving_titles = [articles[i].title for i in surviving]
        assert any("Siddaramaiah" in t for t in surviving_titles)
        assert not any("Siraj" in t for t in surviving_titles)

    def test_wild_cat_prunes_brooklyn_building_collapse_cat(self) -> None:
        """D1: Leopard / wild cat sighting story must not contain Brooklyn apartment collapse cat rescue."""
        articles = [
            _make_article(1, "Forest department tracks elusive wild leopard sighted in residential suburb", "wild leopard forest department suburb"),
            _make_article(2, "Wildlife rangers set humane traps after wild cat spotted near city outskirts", "wild cat rangers wildlife"),
            _make_article(3, "Firefighters in Brooklyn rescue pet cat trapped under building collapse rubble", "Brooklyn building collapse pet cat rescue rubble"),
        ]

        vecs = np.zeros((3, 16), dtype=np.float32)
        vecs[0, 0:4] = [0.88, 0.85, 0.9, 0.82]
        vecs[1, 0:4] = [0.85, 0.82, 0.88, 0.8]
        vecs[2, 6:10] = [0.8, 0.9, 0.7, 0.85]

        surviving = prune_cluster_articles(
            indices=[0, 1, 2],
            articles=articles,
            embeddings=vecs,
            min_similarity=0.58,
            max_sources=8,
        )

        surviving_titles = [articles[i].title for i in surviving]
        assert any("leopard" in t.lower() or "wild cat" in t.lower() for t in surviving_titles)
        assert not any("Brooklyn" in t for t in surviving_titles)

    def test_fifa_tashima_prunes_bukele_el_salvador(self) -> None:
        """D1: FIFA football executive election must not contain El Salvador Bukele political story."""
        articles = [
            _make_article(1, "Japan FA chief Kohzo Tashima re-elected to FIFA Council", "Kohzo Tashima FIFA Council executive football"),
            _make_article(2, "Kohzo Tashima retains prestigious seat on FIFA ruling committee", "Tashima FIFA football governing body"),
            _make_article(3, "El Salvador President Nayib Bukele registers bid for controversial third term", "El Salvador Nayib Bukele president reelection term"),
        ]

        vecs = np.zeros((3, 16), dtype=np.float32)
        vecs[0, 0:4] = [0.92, 0.88, 0.9, 0.85]
        vecs[1, 0:4] = [0.9, 0.86, 0.88, 0.83]
        vecs[2, 6:10] = [0.9, 0.8, 0.7, 0.8]

        surviving = prune_cluster_articles(
            indices=[0, 1, 2],
            articles=articles,
            embeddings=vecs,
            min_similarity=0.58,
            max_sources=8,
        )

        surviving_titles = [articles[i].title for i in surviving]
        assert any("Tashima" in t or "FIFA" in t for t in surviving_titles)
        assert not any("Bukele" in t for t in surviving_titles)

    def test_cji_story_capped_at_max_sources(self) -> None:
        """D1: Massive cluster with 15 sources covering procedural hearing is capped at 8 sources."""
        articles = [
            _make_article(i, f"Supreme Court bench headed by CJI hears landmark electoral plea - Outlet {i}", "CJI Supreme Court electoral hearing")
            for i in range(1, 16)
        ]

        # 15 highly similar articles
        vecs = np.ones((15, 16), dtype=np.float32) * 0.5
        for i in range(15):
            vecs[i, 0] += 0.01 * i

        surviving = prune_cluster_articles(
            indices=list(range(15)),
            articles=articles,
            embeddings=vecs,
            min_similarity=0.58,
            max_sources=8,
        )

        assert len(surviving) == 8, f"Expected cluster to be capped at 8 sources, got {len(surviving)}"
