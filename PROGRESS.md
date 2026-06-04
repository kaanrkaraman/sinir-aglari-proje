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
```

---

## Project goal (one paragraph)

Build a RAG travel-guide chatbot over the Wikivoyage XML dump. Multi-turn, context-aware. **Evaluation rigor is the deliverable** — RAGAS (Faithfulness / Answer Relevance / Context Precision / Context Recall) + retrieval-only metrics (MRR, NDCG, Recall@k) over a manually curated 50+ Q&A ground-truth set. The report compares chunking sizes, retrieval strategies (dense / sparse / hybrid / hybrid+rerank), embedding models, top-k, single-turn vs multi-turn, and a no-retrieval baseline.

Working principles (mandatory, see `pyproject.toml` for enforcement):
- `uv` only — never pip; always `uv add` (let resolver pick versions).
- `ruff format` + `ruff check` + `ty check` + `pytest` must all pass before any file is "done".
- Pydantic v2 for every config & inter-module data shape — no raw dicts crossing modules.
- `pathlib.Path`, structured `logging` (no `print`), `python-dotenv` + `BaseSettings` for config.
- Verify current docs (RAGAS / LangChain / LlamaIndex / MTEB) before implementing — APIs drift.
- **Milestones are gated** — confirm with the user before starting the next one.

---

## Milestone roadmap

| #  | Milestone                                       | Status        | Notes                                                              |
| -- | ----------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| 1  | Environment setup (uv, ruff, ty, pyproject)     | ✅ done       | hatchling backend, src layout                                      |
| 2  | Ground truth dataset (50+ Q&A pairs)            | ✅ done       | 56 Q&A pairs, 20 cities, Claude API, final.jsonl ready             |
| 3  | Wikivoyage XML ingestion + chunking             | ✅ done       | fixed_256/512/1024 + sentence, 1055 chunks (Wikivoyage API)        |
| 4  | Embedding + vector index                        | ✅ done       | nomic-embed-text via Ollama, FAISS index                           |
| 5  | Retrieval-only metrics (MRR, NDCG, Recall@k)    | ✅ done       | MRR=0.26, Recall@5=0.43, NDCG@5=0.35 (fixed_512)                 |
| 6  | End-to-end RAG pipeline                         | ✅ done       | retrieve top-5 + llama3:8b, annrag rag ask                         |
| 7  | RAGAS integration                               | ✅ done       | faithfulness=0.40, context_recall=0.25 (Ollama, 5 queries)         |
| 8  | Experiment grid sweep                           | ⬜            | Chunking × retrieval × embedding × top-k matrix                    |
| 9  | Multi-turn / context-awareness ablation         | ⬜            | Compare single-turn vs multi-turn; cross-document multi-hop         |
| 10 | Polish + reproducible runs + result tables      | ⬜            | Reproducible scripts, final result tables, report                   |

---

## Current state (M7 done, M8 next)

### M1 — Environment setup ✅
- uv, ruff, ty, pytest, hatchling backend, src layout
- 66 tests, 84% coverage

### M2 — Ground truth dataset ✅
- 20 Wikivoyage articles fetched via API (clean text)
- 56 Q&A pairs generated via Claude API
- Distribution: factual=34, multi_hop=7, ambiguous=15
- CLI: `annrag gt fetch`, `annrag gt stats`
- Output: `data/groundtruth/final.jsonl`

### M3 — Chunking ✅
- 4 strategies: fixed_256, fixed_512, fixed_1024, sentence
- 1055 chunks total (20 cities, Wikivoyage API source)
- CLI: `annrag rag chunk --strategy fixed_512`
- Output: `artifacts/chunks.<strategy>.jsonl`

### M4 — Embedding + Vector Index ✅
- Embedder: nomic-embed-text via Ollama (local, free)
- Index: FAISS (IndexFlatIP, cosine similarity)
- CLI: `annrag rag embed --strategy fixed_512`
- CLI: `annrag rag search "query"`
- Output: `artifacts/index.<strategy>.<model>/`

### M5 — Retrieval Metrics ✅
- Evaluated all 56 ground truth queries
- Results (fixed_512, nomic-embed-text, top-10):

| Metric     | Score |
| ---------- | ----- |
| MRR        | 0.26  |
| Recall@1   | 0.13  |
| Recall@3   | 0.34  |
| Recall@5   | 0.43  |
| Recall@10  | 0.55  |
| NDCG@5     | 0.35  |
| NDCG@10    | 0.46  |

- CLI: `annrag rag eval --strategy fixed_512`
- Output: `artifacts/eval.<strategy>.<model>.json`

### M6 — End-to-end RAG pipeline ✅
- Retrieve top-5 chunks → feed to llama3:8b → generate answer
- CLI: `annrag rag ask "What is Singapore known for?"`

### M7 — RAGAS Integration ✅
- Evaluated 5 queries with RAGAS (Ollama backend)
- Results:

| Metric             | Score |
| ------------------ | ----- |
| Faithfulness       | 0.40  |
| Answer Relevancy   | nan*  |
| Context Precision  | nan*  |
| Context Recall     | 0.25  |

*nan = timeout (llama3:8b too slow for RAGAS async jobs on Mac)
- CLI: `annrag rag ragas --limit 5`
- Output: `artifacts/ragas.<strategy>.<model>.json`

### M8 — Experiment Grid Sweep ⬜ (next)
- Compare chunking strategies × top-k × embedding models

---

## Resume actions (M8)

```bash
# Ollama running?
open -a Ollama
ollama list   # llama3:8b and nomic-embed-text must be present

# Run grid sweep
uv run annrag rag grid
```

---

## Architecture decisions

- **`relevant_doc_ids` are page titles, not chunk IDs.** Survives every chunking strategy in the M3/M8 sweep. Retrieval scoring in M5 = `gold_pages ∩ {chunk.source_page for chunk in retrieved_top_k}` — chunks count by their parent page.
- **Multi-hop in M2 is intra-article only.** True cross-document multi-hop is deferred to M9 (alongside the multi-turn ablation).
- **LLM provider is pluggable.** `Settings.llm_provider` is `Literal["anthropic", "ollama"]`; `build_llm_client()` is the only construction site.
- **`generator_model` is preserved on every QAPair through curation.** The CSV round-trip keeps it, so the final dataset retains provenance for cross-model analysis in the report.

---

## Gotchas (don't relearn these the hard way)

1. **macOS hidden flag on `.venv/` breaks editable installs.** Fix: `chflags -R nohidden .venv`
2. **Wikimedia API requires URL/email contact in User-Agent.** A bare descriptive UA gets HTTP 403.
3. **Ollama `format=<schema>` doesn't enforce `minLength`.** Small models emit empty strings — per-item validator skips bad rows.
4. **nomic-embed-text has a token limit.** Truncate input to 2000 chars before embedding to avoid HTTP 500.

---

## Where things live

```
src/annrag/
  config.py            BaseSettings (paths, log level, LLM provider, Wikivoyage UA)
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
  rag/
    models.py          Chunk (Pydantic v2, frozen)
    chunking.py        FixedSizeChunker, SentenceBoundaryChunker, get_strategy()
    loader.py          JSON article loader
    embedder.py        OllamaEmbedder (nomic-embed-text)
    index.py           VectorIndex (FAISS, save/load/search)
    metrics.py         QueryResult, RetrievalMetrics (MRR, Recall@k, NDCG@k)

tests/
  rag/
    test_chunking.py   10 tests
    test_index.py       7 tests
    test_metrics.py    15 tests

data/
  groundtruth/
    seed_articles.json  curated seed list (20 cities)
    raw/                fetched ArticleExtract JSONs (gitignored)
    candidates.*.jsonl  per-run output (gitignored)
    review.csv          human-curated CSV (gitignored)
    final.jsonl         56 accepted Q&A pairs (gitignored)

artifacts/
  chunks.<strategy>.jsonl          chunked articles (gitignored)
  index.<strategy>.<model>/        FAISS index + metadata (gitignored)
  eval.<strategy>.<model>.json     retrieval metrics (gitignored)
```

---

## Toolchain quick reference

```bash
uv add <pkg>                # add runtime dep
uv add --dev <pkg>          # dev dep
uv sync                     # resync .venv
uv run <cmd>                # run inside the venv

uv run ruff format          # auto-format
uv run ruff check --fix     # lint + auto-fix
uv run ty check             # static types
uv run pytest               # tests + coverage

uv run annrag <subcmd>      # the project CLI
```
