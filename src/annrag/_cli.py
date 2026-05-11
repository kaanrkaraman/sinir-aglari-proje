"""Typer CLI entry point — every operational subcommand for M2.

The CLI is the only public surface the report and any human reviewer hits.
Subcommands group under `gt` (ground-truth) since M2 is exclusively about
producing the evaluation dataset; later milestones will add `index`, `eval`,
and `chat` groups under the same root app.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated

import typer

from annrag.config import LLMProvider, Settings
from annrag.groundtruth.curate import export_for_review, import_curated
from annrag.groundtruth.fetch import ArticleExtract, WikivoyageFetcher
from annrag.groundtruth.generate import QAGenerator
from annrag.groundtruth.stats import compute_stats
from annrag.groundtruth.storage import (
    append_qa_pairs,
    load_qa_pairs,
    load_seed_titles,
)
from annrag.llm.base import LLMError
from annrag.llm.factory import ConfigurationError, build_llm_client
from annrag.llm.ollama_client import list_available_models
from annrag.logging_setup import configure_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="annrag",
    help="Wikivoyage RAG chatbot — coursework CLI.",
    no_args_is_help=True,
    add_completion=False,
)
gt_app = typer.Typer(
    name="gt",
    help="Ground-truth dataset commands (fetch, generate, curate, stats).",
    no_args_is_help=True,
)
app.add_typer(gt_app, name="gt")


_VALID_PROVIDERS: tuple[LLMProvider, ...] = ("anthropic", "ollama")


def _validate_provider(value: str | None) -> LLMProvider | None:
    if value is None:
        return None
    if value not in _VALID_PROVIDERS:
        raise typer.BadParameter(
            f"unknown provider {value!r}; expected one of {list(_VALID_PROVIDERS)}"
        )
    return value  # type: ignore[return-value]


def _slug(value: str) -> str:
    """Filesystem-safe slug for a model tag (e.g. 'llama3:8b' → 'llama3_8b')."""
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "model"


# ---------------------------------------------------------------------------
# Root-level commands
# ---------------------------------------------------------------------------
@app.command()
def bootstrap() -> None:
    """Create data/artifacts directories and print the loaded configuration."""
    settings = Settings()
    configure_logging(settings.log_level)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"data_dir       = {settings.data_dir}")
    typer.echo(f"artifacts_dir  = {settings.artifacts_dir}")
    typer.echo(f"log_level      = {settings.log_level}")
    typer.echo(f"llm_provider   = {settings.llm_provider}")
    typer.echo(f"ollama_model   = {settings.ollama_model}")
    typer.echo(f"wikivoyage_lang= {settings.wikivoyage_lang}")


# ---------------------------------------------------------------------------
# gt fetch
# ---------------------------------------------------------------------------
@gt_app.command("fetch")
def gt_fetch(
    seed: Annotated[
        Path | None,
        typer.Option("--seed", help="Path to seed_articles.json (default: <data>/groundtruth/seed_articles.json)."),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Where to write per-article JSON (default: <data>/groundtruth/raw)."),
    ] = None,
) -> None:
    """Fetch Wikivoyage extracts for every title in the seed list."""
    settings = Settings()
    configure_logging(settings.log_level)
    seed_path = seed or (settings.data_dir / "groundtruth" / "seed_articles.json")
    target_dir = out_dir or (settings.data_dir / "groundtruth" / "raw")
    if not seed_path.exists():
        typer.echo(f"seed file not found: {seed_path}", err=True)
        raise typer.Exit(code=1)

    titles = load_seed_titles(seed_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    skipped = 0
    with WikivoyageFetcher(
        lang=settings.wikivoyage_lang,
        user_agent=settings.wikivoyage_user_agent,
        delay_s=settings.wikivoyage_request_delay_s,
    ) as fetcher:
        for title in titles:
            try:
                article = fetcher.fetch_extract(title)
            except Exception as e:  # noqa: BLE001 — fetcher exposes httpx + custom errors
                logger.error("fetch failed for %r: %s", title, e)
                skipped += 1
                continue
            if article is None:
                skipped += 1
                continue
            out_path = target_dir / f"{_slug(article.resolved_title)}.json"
            out_path.write_text(article.model_dump_json(indent=2), encoding="utf-8")
            fetched += 1
    typer.echo(f"fetched {fetched} articles (skipped {skipped}) → {target_dir}")


# ---------------------------------------------------------------------------
# gt models
# ---------------------------------------------------------------------------
@gt_app.command("models")
def gt_models() -> None:
    """List models installed in the local Ollama daemon."""
    settings = Settings()
    configure_logging(settings.log_level)
    try:
        tags = list_available_models(settings.ollama_base_url)
    except Exception as e:  # noqa: BLE001 — ollama SDK can raise httpx errors
        typer.echo(f"failed to query Ollama at {settings.ollama_base_url}: {e}", err=True)
        raise typer.Exit(code=1) from e
    if not tags:
        typer.echo("(no models installed)")
        return
    for tag in tags:
        typer.echo(tag)


# ---------------------------------------------------------------------------
# gt generate
# ---------------------------------------------------------------------------
@gt_app.command("generate")
def gt_generate(
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Override Settings.llm_provider ('anthropic' or 'ollama')."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override the provider's default model tag."),
    ] = None,
    per_article: Annotated[
        int,
        typer.Option("--per-article", help="Candidates per source article.", min=1, max=20),
    ] = 4,
    raw_dir: Annotated[
        Path | None,
        typer.Option("--raw-dir", help="Directory of fetched ArticleExtract JSONs."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output JSONL; default candidates.<provider>.<model>.jsonl."),
    ] = None,
) -> None:
    """Run LLM-driven Q&A generation against every article in `raw_dir`."""
    settings = Settings()
    configure_logging(settings.log_level)
    validated_provider = _validate_provider(provider)
    try:
        client, model_id = build_llm_client(
            settings, provider=validated_provider, model=model
        )
    except ConfigurationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e

    raw_root = raw_dir or (settings.data_dir / "groundtruth" / "raw")
    if not raw_root.exists():
        typer.echo(f"raw dir not found: {raw_root}", err=True)
        raise typer.Exit(code=1)

    chosen_provider = validated_provider or settings.llm_provider
    out_path = out or (
        settings.data_dir
        / "groundtruth"
        / f"candidates.{chosen_provider}.{_slug(model_id)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator = QAGenerator(client, model_name=model_id)
    json_files = sorted(raw_root.glob("*.json"))
    if not json_files:
        typer.echo(f"no article JSONs in {raw_root}", err=True)
        raise typer.Exit(code=1)

    total_written = 0
    for path in json_files:
        try:
            article = ArticleExtract.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError as e:
            logger.error("skip malformed article JSON %s: %s", path, e)
            continue
        try:
            records, _batch = generator.generate(article, n=per_article)
        except LLMError as e:
            logger.error("generation failed for %r: %s", article.resolved_title, e)
            continue
        total_written += append_qa_pairs(out_path, records)
    typer.echo(f"wrote {total_written} candidates to {out_path}")


# ---------------------------------------------------------------------------
# gt export / import (curation seam)
# ---------------------------------------------------------------------------
@gt_app.command("export")
def gt_export(
    in_path: Annotated[
        Path, typer.Option("--in", help="Candidates JSONL to export.")
    ],
    out_path: Annotated[
        Path | None,
        typer.Option("--out", help="Output CSV (default: <data>/groundtruth/review.csv)."),
    ] = None,
) -> None:
    """Export a JSONL of candidates as a CSV for human curation."""
    settings = Settings()
    configure_logging(settings.log_level)
    target = out_path or (settings.data_dir / "groundtruth" / "review.csv")
    records = load_qa_pairs(in_path)
    written = export_for_review(records, target)
    typer.echo(f"exported {written} rows → {target}")


@gt_app.command("import")
def gt_import(
    in_path: Annotated[
        Path, typer.Option("--in", help="Reviewed CSV to import.")
    ],
    out_path: Annotated[
        Path | None,
        typer.Option("--out", help="Output JSONL (default: <data>/groundtruth/final.jsonl)."),
    ] = None,
) -> None:
    """Import a reviewer-edited CSV: keep accepted rows, re-derive IDs, write JSONL."""
    settings = Settings()
    configure_logging(settings.log_level)
    target = out_path or (settings.data_dir / "groundtruth" / "final.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    n = import_curated(in_path, target)
    typer.echo(f"imported {n} accepted rows → {target}")


# ---------------------------------------------------------------------------
# gt stats
# ---------------------------------------------------------------------------
@gt_app.command("stats")
def gt_stats(
    path: Annotated[Path, typer.Argument(help="Path to a QA JSONL file.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit a single JSON object instead of the human report.")
    ] = False,
) -> None:
    """Print distribution stats over a Q&A JSONL dataset."""
    settings = Settings()
    configure_logging(settings.log_level)
    records = load_qa_pairs(path)
    stats = compute_stats(records)
    if as_json:
        payload = {
            "total": stats.total,
            "by_category": stats.by_category,
            "by_difficulty": stats.by_difficulty,
            "by_source": stats.by_source,
            "by_language": stats.by_language,
            "by_page": stats.by_page,
            "pages_count": stats.pages_count,
            "avg_question_chars": stats.avg_question_chars,
            "avg_answer_chars": stats.avg_answer_chars,
        }
        typer.echo(json.dumps(payload))
    else:
        typer.echo(stats.render())


def main() -> None:
    """Entry point — invoked by the `annrag` console script and `python -m annrag`."""
    app()


if __name__ == "__main__":
    main()
