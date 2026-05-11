# annrag

RAG-based travel guide chatbot built over the Wikivoyage XML dump.
Course project for Neural Networks — pipeline quality, evaluation rigor,
and comparative analysis are the deliverable.

## Stack

- **Package management:** `uv` (never `pip`)
- **Lint + format:** `ruff`
- **Static types:** `ty`
- **Tests:** `pytest`
- **Config + data models:** Pydantic v2 / `pydantic-settings`

## Quick start

```bash
uv sync                 # install runtime + dev deps from uv.lock
cp .env.example .env    # fill in values as needed (M1 needs none)
uv run annrag           # bootstrap smoke check
```

## Toolchain loop

Before any file is "done":

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check
uv run pytest
```

## Layout

```
src/annrag/        package source (config, logging, future RAG modules)
tests/             pytest suite
data/              raw + processed Wikivoyage data (gitignored)
artifacts/         indexes, embeddings, eval CSVs (gitignored)
```

## Ground-truth pipeline (M2)

The default LLM provider is **Ollama** (local). Anthropic stays available as
an alternative — set `ANTHROPIC_API_KEY` and pass `--provider anthropic`.

```bash
# 0. Start Ollama (macOS: `open -a Ollama`; Linux: `ollama serve &`)
uv run annrag gt models                          # confirm models are pulled

# 1. Fetch seed Wikivoyage articles (read-only API; ~12s for 22 articles)
uv run annrag gt fetch

# 2. Generate Q&A candidates per (provider, model). Output filename auto-includes
#    the model slug so multi-model runs don't clobber each other.
uv run annrag gt generate --model llama3:8b
uv run annrag gt generate --model mistral:7b
#   → data/groundtruth/candidates.ollama.llama3_8b.jsonl
#   → data/groundtruth/candidates.ollama.mistral_7b.jsonl

# 3. Export the candidates you want to curate as a CSV → open in spreadsheet
#    → set `accepted=y` on the rows to keep. `relevant_doc_ids` is preserved.
uv run annrag gt export --in data/groundtruth/candidates.ollama.llama3_8b.jsonl

# 4. Import accepted rows → final dataset
uv run annrag gt import

# 5. Stats over the final dataset (or any candidates file)
uv run annrag gt stats
```

Each `QAPair` carries `generator_model` through curation so the final
dataset retains provenance for cross-model analysis in the report.

## Milestones

See the project brief for the full 10-step roadmap. Each milestone is
gated — confirm before advancing.

| # | Milestone                                       | Status |
| - | ----------------------------------------------- | ------ |
| 1 | Environment setup (uv, ruff, ty, pyproject)     |   ✓    |
| 2 | Ground truth dataset (50+ Q&A pairs)            |  WIP   |
| 3 | Wikivoyage XML ingestion + chunking             |        |
| 4 | Embedding + vector index                        |        |
| 5 | Retrieval-only metrics (MRR, NDCG, Recall@k)    |        |
| 6 | End-to-end RAG pipeline                         |        |
| 7 | RAGAS integration                               |        |
| 8 | Experiment grid sweep                           |        |
| 9 | Multi-turn / context-awareness ablation         |        |
| 10| Polish + reproducible runs + result tables      |        |
