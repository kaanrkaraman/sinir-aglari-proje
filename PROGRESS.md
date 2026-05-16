## M2 DONE — 2025-05-13
56 Q&A pairs curated, advancing to M3.

---

# annrag — Progress & Roadmap

Resumable session log for the Wikivoyage RAG chatbot (Neural Networks course project).
Pick up here in a fresh Claude Code session.

---

## Quick start (sit-down checklist)

```bash
# 1. Make sure the editable install is visible to Python (macOS quirk — see Gotchas)
chflags -R nohidden .venv

# 2. Toolchain green-light
uv sync
uv run ruff format && uv run ruff check && uv run ty check && uv run pytest

# 3. If Ollama isn't running:
open -a Ollama          # macOS desktop app launches the daemon at :11434
uv run annrag gt models # confirm models are visible

# 4. The next operational step is generating Q&A candidates (see "Where we left off")
```

---

## Project goal (one paragraph)

| 5  | Retrieval-only metrics (MRR, NDCG, Recall@k)    | ✅ done       | MRR=0.26, Recall@5=0.43, NDCG@5=0.35                |
Working principles (mandatory, see `pyproject.toml` for enforcement):
- `uv` only — never pip; always `uv add` (let resolver pick versions).
- `ruff format` + `ruff check` + `ty check` + `pytest` must all pass before any file is "done".
- Pydantic v2 for every config & inter-module data shape — no raw dicts crossing modules.
- `pathlib.Path`, structured `logging` (no `print`), `python-dotenv` + `BaseSettings` for config.
- Verify current docs (RAGAS / LangChain / LlamaIndex / MTEB) before implementing — APIs drift.
- **Milestones are gated** — confirm with the user before starting the next one.

---

## Milestone roadmap

| #  | Milestone                                       | Status        | Notes                                                                          |
| -- | ----------------------------------------------- | ------------- | ------------------------------------------------------------------------------ |
| 1  | Environment setup (uv, ruff, ty, pyproject)     | ✅ done       | hatchling backend, src layout                            
| 2  | Ground truth dataset (50+ Q&A pairs)            | ✅ done       | 56 Q&A pairs curated (20 cities, Claude API). final.jsonl ready.              | |
| 3  | Wikivoyage XML ingestion + chunking             | ✅ done       | fixed_256/512/1024 + sentence, 1055 chunks (API)                                                        |
| 4  | Embedding + vector index                        | ✅ done       | nomic-embed-text, FAISS index, annrag rag embed/search                                                                                |
| 5  | Retrieval-only metrics (MRR, NDCG, Recall@k)    | ⬜            | Page-level overlap (gold pages ∩ retrieved chunks' source page) — design baked |
| 6  | End-to-end RAG pipeline                         | ⬜            |                                                                                |
| 7  | RAGAS integration                               | ⬜            |                                                                                |
| 8  | Experiment grid sweep                           | ⬜            | Chunking × retrieval × embedding × top-k matrix                                |
| 9  | Multi-turn / context-awareness ablation         | ⬜            |                                                                                |
| 10 | Polish + reproducible runs + result tables      | ⬜            |                                                                                |

---

## Where we left off (M2 detailed state)

**All M2 infrastructure is built, lint-clean, type-clean, 66 tests pass, 84% coverage.**

What's in the repo:
- `src/annrag/groundtruth/` — `models.py` (QAPair, Pydantic v2, frozen, `relevant_doc_ids` are normalized Wikivoyage page titles), `storage.py` (JSONL streaming, dedup, malformed-line skip), `fetch.py` (MediaWiki API client, polite UA + 0.5s throttle), `generate.py` (per-item validation, graceful skip on bad rows), `curate.py` (CSV round-trip with editable fields), `stats.py` (distribution by category/difficulty/source/language/page).
- `src/annrag/llm/` — `base.py` (Protocol), `anthropic_client.py`, `ollama_client.py` (uses `format=<json_schema>` for strict structured output), `factory.py` (provider dispatch).
- `src/annrag/_cli.py` — Typer app with `bootstrap`, `gt fetch`, `gt models`, `gt generate`, `gt export`, `gt import`, `gt stats`.
- `data/groundtruth/raw/` — 22 fetched Wikivoyage articles (~1.9 MB plain text).
- `data/groundtruth/seed_articles.json` — curated seed list of 22 cities/countries/regions.
- `tests/` — 66 tests, including httpx MockTransport for fetch, mocked SDKs for LLM clients, Typer CliRunner for CLI.

What's NOT done in M2:
1. **Run the actual generation** against the live Ollama. `llama3:8b` and `mistral:7b` are pulled and ready. **Do this next** (see "Resume actions" below).
2. **Human curation step.** This is unavoidably manual: open `review.csv` in a spreadsheet, mark `accepted=y` on the rows to keep, target ≥50 across the diversity axes (factual / multi_hop / ambiguous).
3. **Run `gt import` then `gt stats`** to land `final.jsonl` and confirm distribution.

---

## Resume actions (in order)

Assume Ollama is running (`open -a Ollama`).

```bash
# 1. Generate with the best local model — full 22 articles, 4 candidates each.
#    Each article is ~30-90s on an M-series Mac; budget ~25 min total.
uv run annrag gt generate --model llama3:8b --per-article 4
#    → data/groundtruth/candidates.ollama.llama3_8b.jsonl

# 2. (Optional but recommended) Same with a different model for cross-model diversity.
uv run annrag gt generate --model mistral:7b --per-article 4
#    → data/groundtruth/candidates.ollama.mistral_7b.jsonl

# 3. Pick which candidates file to curate (or `cat` them together first).
uv run annrag gt export --in data/groundtruth/candidates.ollama.llama3_8b.jsonl
#    → data/groundtruth/review.csv

# 4. Open review.csv in any spreadsheet, set `accepted` to y/yes/true on ≥50 rows.
#    Aim for a mix of factual / multi_hop / ambiguous (small models bias to factual —
#    you may need both candidate files merged to hit a balanced 50).

# 5. Import accepted rows.
uv run annrag gt import
#    → data/groundtruth/final.jsonl

# 6. Stats — for the report.
uv run annrag gt stats
```

After M2 finishes (≥50 records in `final.jsonl`), confirm with the user before starting M3.

---

## M3 design — open questions to answer at the next gate

When the user says "go for M3", confirm these two before writing code:

1. **XML dump scope.** Full `enwikivoyage-latest-pages-articles.xml.bz2` (~600MB) gives a realistic retrieval challenge for the report. Subset (only seed pages + their cross-links) is faster to iterate. Recommendation: full dump — it's a one-time download and makes the experiment grid more meaningful.
2. **Initial chunkers in M3.** Brief lists fixed 256 / 512 / 1024 + sentence-boundary as the *experiment axis*. Implementing one (default 512 with 64-token overlap) in M3 and the others in M8 is the lean path. Or land all four upfront. Recommendation: all four in M3 — they're cheap to implement and it lets M5's retrieval metrics start producing the comparison axis immediately.

Other M3 design choices already baked:
- Use `mwparserfromhell` for wikitext cleaning (templates, tables, links).
- `xml.etree.iterparse` (or `mwxml`) for streaming the XML — never load the whole dump.
- Each chunk gets `source_page` (page title) in metadata so M5 can do page-level retrieval scoring.
- Token-counting via `tiktoken` (cl100k_base) for fixed-size chunkers — purely for chunk sizing, not because we use OpenAI models.

---

## Architecture decisions worth re-reading before extending

- **`relevant_doc_ids` are page titles, not chunk IDs.** Survives every chunking strategy in the M3/M8 sweep. Retrieval scoring in M5 = `gold_pages ∩ {chunk.source_page for chunk in retrieved_top_k}` — chunks count by their parent page.
- **Multi-hop in M2 is intra-article only.** True cross-document multi-hop is deferred to M9 (alongside the multi-turn ablation), where we can co-fetch article pairs and prompt accordingly.
- **LLM provider is pluggable.** `Settings.llm_provider` is `Literal["anthropic", "ollama"]`; `build_llm_client()` is the only construction site. Default is `ollama`. CLI `--provider` and `--model` override per-run, and `gt generate` writes to `candidates.<provider>.<model_slug>.jsonl` so multi-model runs don't clobber each other.
- **`generator_model` is preserved on every QAPair through curation.** The CSV round-trip keeps it, so the final dataset retains provenance for cross-model analysis in the report.
- **Small-model failure mode (open finding).** llama3:8b and llama3.2:3b both collapse the category enum — every Q&A in a batch ends up tagged with one category despite the prompt asking for diversity. Two unimplemented workarounds we may want for the report: (a) per-category passes (3× requests/article, forces diversity), (b) split-array schema with one array per category. For now we lean on multi-model + curation.
- **Evidence is paraphrased, not verbatim, on small models.** Affects RAGAS faithfulness leniency in M7. If problematic we can either re-prompt for verbatim or strip evidence from the gold set entirely.

---

## Gotchas (don't relearn these the hard way)

1. **macOS hidden flag on `.venv/` breaks editable installs.** uv tags the venv with `UF_HIDDEN`; CPython's `site.addpackage` silently skips hidden `.pth` files. Symptom: `ModuleNotFoundError: No module named 'annrag'` in `uv run python -c "import annrag"` despite the `.pth` existing. Fix: `chflags -R nohidden .venv`. Persistent across `uv sync --reinstall-package` runs once cleared. Diagnostic: `ls -lO .venv/lib/python3.12/site-packages/*.pth` — `hidden` in the flags column = this issue.
2. **Wikimedia API requires URL/email contact in User-Agent.** A bare descriptive UA gets HTTP 403 with body referencing `https://w.wiki/4wJS`. The current default UA in `Settings.wikivoyage_user_agent` includes a URL contact and works. `curl` and `httpx` behave differently for the *same* UA string (TLS fingerprint heuristics), so curl-passes ≠ httpx-passes — always test through the actual fetcher.
3. **Ollama `format=<schema>` doesn't enforce `minLength`.** Even with `required` and a JSON schema, small models emit empty strings for fields they're stuck on. The Pydantic inner `_RawQAItem` is now lenient for `evidence` and the per-item validator skips bad rows rather than failing the whole batch.
4. **Build backend.** Switched from `uv_build` (beta) to `hatchling` early. The hidden-flag issue is venv-level, not backend-level, so the switch was incidental — but hatchling stays since it's mature.

---

## Where things live

```
src/annrag/
  config.py            BaseSettings (paths, log level, LLM provider, Ollama/Anthropic config, Wikivoyage UA)
  logging_setup.py     Idempotent stderr handler, structured format
  _cli.py              Typer app — every subcommand
  groundtruth/
    models.py          QAPair, normalize_page_title, derive_qa_id, enums
    storage.py         JSONL load/save/append, seed loader
    fetch.py           WikivoyageFetcher (httpx, MediaWiki API)
    generate.py        QAGenerator + tool schema + system prompt
    curate.py          CSV export/import for human-in-the-loop curation
    stats.py           Distribution reporter
  llm/
    base.py            LLMClient Protocol + LLMError
    anthropic_client.py
    ollama_client.py   format=<schema> for structured output; list_available_models
    factory.py         build_llm_client(settings, provider?, model?) → (client, model_id)

tests/                  66 tests (mirrors src layout)
data/
  groundtruth/
    seed_articles.json  curated seed list (22 articles)
    raw/                fetched ArticleExtract JSONs (gitignored, but already on disk)
    candidates.<provider>.<model>.jsonl   per-run output (gitignored)
    review.csv          human-curated CSV
    final.jsonl         accepted Q&A → drives all later metrics
artifacts/             empty; lands indexes, embeddings, eval CSVs in M4+
```

Auto-loaded session memory (per-project) lives at:
`~/.claude/projects/-Users-kaan-Desktop-Files-Kod-Python-ANNRAGProject/memory/`
— see `MEMORY.md` there for the index. Notable entries: project goal & evaluation strategy, toolchain rules, milestone gating, Ollama setup notes, Wikimedia UA gotcha, hidden-flag gotcha.

---

## Toolchain quick reference

```bash
uv add <pkg>                # add runtime dep (let resolver pick version)
uv add --dev <pkg>          # dev dep
uv sync                     # resync .venv
uv run <cmd>                # run inside the venv

uv run ruff format          # auto-format
uv run ruff check --fix     # lint + auto-fix
uv run ty check             # static types
uv run pytest               # tests + coverage

uv run annrag <subcmd>      # the project CLI
```
