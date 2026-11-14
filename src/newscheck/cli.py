"""Command-line interface for NewsCheck.

Provides the ``newscheck`` command with subcommands for running the
pipeline, inspecting results, and managing the database.
"""

from __future__ import annotations

from pathlib import Path

import typer

from newscheck.log import setup_logging

app = typer.Typer(
    name="newscheck",
    help="NewsCheck - automated news corroboration pipeline.",
    no_args_is_help=True,
)


@app.command()
def run(
    project_root: Path | None = typer.Option(None, "--root", "-r", help="Project root directory (defaults to cwd)."),
    db_path: Path | None = typer.Option(None, "--db", help="SQLite database path."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-formatted log lines."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR)."),
) -> None:
    """Run the full pipeline: ingest -> extract -> dedup -> cluster -> rank -> verify -> publish."""
    setup_logging(json_output=json_logs, level=log_level)

    from newscheck.pipeline import run_pipeline

    result = run_pipeline(project_root=project_root, db_path=db_path)

    if result.success:
        typer.echo(f"\n[OK] Pipeline complete. {result.articles_fetched} articles, {len(result.clusters)} clusters.")
        for cluster in result.clusters:
            typer.echo(
                f"  #{cluster.rank}: {cluster.label} "
                f"({cluster.independent_source_count} sources, "
                f"score={cluster.tier_weighted_score})"
            )
    else:
        typer.echo(f"\n[ERROR] Pipeline failed: {result.error}", err=True)
        raise typer.Exit(code=1)


@app.command()
def init_db(
    db_path: Path | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Initialise (or re-initialise) the database schema."""
    setup_logging(level="INFO")

    from newscheck.db import get_connection
    from newscheck.db import init_db as _init_db

    conn = get_connection(db_path)
    _init_db(conn)
    conn.close()
    typer.echo("[OK] Database initialised.")


@app.command()
def show_sources(
    project_root: Path | None = typer.Option(None, "--root", "-r", help="Project root directory."),
) -> None:
    """List configured sources from sources.yaml."""
    setup_logging(level="WARNING")

    from newscheck.config import load_settings, load_sources_yaml

    if project_root is None:
        project_root = Path.cwd()
    settings = load_settings(project_root=project_root)
    feeds_path = project_root / settings.ingest.feeds_file
    sources = load_sources_yaml(feeds_path)

    typer.echo(f"{'Name':<30} {'Tier':<6} {'Domain':<30}")
    typer.echo("-" * 70)
    for src in sources:
        typer.echo(f"{src['name']:<30} {src.get('tier', '?'):<6} {src['domain']:<30}")
    typer.echo(f"\nTotal: {len(sources)} sources")


@app.command()
def build(
    project_root: Path | None = typer.Option(None, "--root", "-r", help="Project root directory."),
    db_path: Path | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Build the static HTML site from current database records."""
    setup_logging(level="INFO")

    from newscheck.build.site import SiteGenerator
    from newscheck.config import load_settings
    from newscheck.db import get_connection

    if project_root is None:
        project_root = Path.cwd()
    settings = load_settings(project_root=project_root)
    conn = get_connection(db_path)
    from newscheck.db import init_db as _init_db

    _init_db(conn)
    site_gen = SiteGenerator(settings, project_root)
    out = site_gen.build_site(conn)
    conn.close()
    typer.echo(f"[OK] Static site built at: {out}")


if __name__ == "__main__":
    app()
