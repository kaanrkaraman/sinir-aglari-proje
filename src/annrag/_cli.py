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


# ── RAG subcommand group (M3+) ────────────────────────────────────────────────

rag_app = typer.Typer(name="rag", help="RAG pipeline commands (M3+).")
app.add_typer(rag_app)


@rag_app.command("chunk")
def rag_chunk(
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Chunking strategy: fixed_256 | fixed_512 | fixed_1024 | sentence",
        ),
    ] = "fixed_512",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output JSONL path (default: artifacts/chunks.<strategy>.jsonl)"),
    ] = None,
) -> None:
    """Chunk all raw articles and write chunks to JSONL."""
    import json as _json

    from annrag.rag.chunking import get_strategy
    from annrag.rag.loader import load_all_articles

    settings = Settings()
    raw_dir = settings.data_dir / "groundtruth" / "raw"
    artifacts_dir = settings.data_dir.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_path = out or (artifacts_dir / f"chunks.{strategy}.jsonl")

    chunker = get_strategy(strategy)
    articles = load_all_articles(raw_dir)

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for page_title, text in articles:
            chunks = chunker.chunk(page_title, text)
            for chunk in chunks:
                f.write(_json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
            total += len(chunks)
            typer.echo(f"  {page_title}: {len(chunks)} chunks")

    typer.echo(f"\nTotal: {total} chunks → {out_path}")


@rag_app.command("stats")
def rag_stats(
    path: Annotated[Path, typer.Argument(help="Chunks JSONL file.")],
) -> None:
    """Show statistics for a chunks JSONL file."""
    import json as _json

    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(_json.loads(line))

    if not chunks:
        typer.echo("No chunks found.")
        raise typer.Exit(1)

    token_counts = [c["token_count"] for c in chunks]
    pages = {c["source_page"] for c in chunks}
    strategy = chunks[0]["strategy"]

    typer.echo(f"strategy:      {strategy}")
    typer.echo(f"total chunks:  {len(chunks)}")
    typer.echo(f"unique pages:  {len(pages)}")
    typer.echo(f"avg tokens:    {sum(token_counts) / len(token_counts):.1f}")
    typer.echo(f"min tokens:    {min(token_counts)}")
    typer.echo(f"max tokens:    {max(token_counts)}")


if __name__ == "__main__":
    main()


@rag_app.command("embed")
def rag_embed(
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Chunking strategy to embed."),
    ] = "fixed_512",
    model: Annotated[
        str,
        typer.Option("--model", help="Ollama embedding model."),
    ] = "nomic-embed-text",
) -> None:
    """Embed all chunks and save FAISS index."""
    import json as _json

    from annrag.rag.embedder import OllamaEmbedder
    from annrag.rag.index import VectorIndex
    from annrag.rag.models import Chunk

    settings = Settings()
    artifacts_dir = settings.data_dir.parent / "artifacts"
    chunks_path = artifacts_dir / f"chunks.{strategy}.jsonl"
    slug = model.replace(":", "_").replace("-", "_")
    index_dir = artifacts_dir / f"index.{strategy}.{slug}"

    if not chunks_path.exists():
        typer.echo(f"Chunks file not found: {chunks_path}")
        typer.echo(f"Run: annrag rag chunk --strategy {strategy}")
        raise typer.Exit(1)

    typer.echo(f"Loading chunks from {chunks_path}...")
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**_json.loads(line)))

    typer.echo(f"{len(chunks)} chunks loaded.")
    typer.echo(f"Embedding with {model}...")

    embedder = OllamaEmbedder(model=model)
    index = VectorIndex(dim=embedder.dim)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vectors = embedder.embed_batch(texts)
        index.add_batch(batch, vectors)
        typer.echo(f"  {min(i + batch_size, len(chunks))} / {len(chunks)}")

    index.save(index_dir)
    typer.echo(f"\nDone! Index saved to {index_dir}")


@rag_app.command("search")
def rag_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    strategy: Annotated[str, typer.Option("--strategy", "-s")] = "fixed_512",
    model: Annotated[str, typer.Option("--model")] = "nomic-embed-text",
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
) -> None:
    """Search the vector index with a query."""
    from annrag.rag.embedder import OllamaEmbedder
    from annrag.rag.index import VectorIndex

    settings = Settings()
    artifacts_dir = settings.data_dir.parent / "artifacts"
    slug = model.replace(":", "_").replace("-", "_")
    index_dir = artifacts_dir / f"index.{strategy}.{slug}"

    if not index_dir.exists():
        typer.echo(f"Index not found: {index_dir}")
        typer.echo(f"Run: annrag rag embed --strategy {strategy}")
        raise typer.Exit(1)

    index = VectorIndex.load(index_dir)
    embedder = OllamaEmbedder(model=model)

    query_vec = embedder.embed(query)
    results = index.search(query_vec, top_k=top_k)

    typer.echo(f"\nQuery: {query!r}")
    typer.echo(f"Top {top_k} results:\n")
    for i, (chunk, score) in enumerate(results, 1):
        typer.echo(f"[{i}] Score: {score:.4f} | Page: {chunk.source_page}")
        typer.echo(f"    {chunk.text[:200]}...")
        typer.echo()


@rag_app.command("eval")
def rag_eval(
    strategy: Annotated[str, typer.Option("--strategy", "-s")] = "fixed_512",
    model: Annotated[str, typer.Option("--model")] = "nomic-embed-text",
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Evaluate retrieval quality: MRR, Recall@k, NDCG@k."""
    from annrag.rag.metrics import evaluate_retrieval

    settings = Settings()
    final_jsonl = settings.data_dir / "groundtruth" / "final.jsonl"
    artifacts_dir = settings.data_dir.parent / "artifacts"
    slug = model.replace(":", "_").replace("-", "_")
    index_dir = artifacts_dir / f"index.{strategy}.{slug}"

    if not final_jsonl.exists():
        typer.echo(f"final.jsonl not found: {final_jsonl}")
        raise typer.Exit(1)

    if not index_dir.exists():
        typer.echo(f"Index not found: {index_dir}")
        typer.echo(f"Run: annrag rag embed --strategy {strategy}")
        raise typer.Exit(1)

    typer.echo(f"Evaluating retrieval (strategy={strategy}, top_k={top_k})...")
    metrics = evaluate_retrieval(
        final_jsonl=final_jsonl,
        index_dir=index_dir,
        embedder_model=model,
        top_k=top_k,
    )

    if json_out:
        typer.echo(json.dumps(metrics.to_dict(), indent=2))
    else:
        typer.echo("\n" + metrics.render())

    # Save results
    out_path = artifacts_dir / f"eval.{strategy}.{slug}.json"
    out_path.write_text(
        json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    typer.echo(f"\nSaved to {out_path}")
