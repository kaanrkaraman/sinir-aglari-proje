"""RAGAS integration — M7.

Evaluates RAG quality using RAGAS metrics:
  - Faithfulness
  - Answer Relevance
  - Context Precision
  - Context Recall

Uses Ollama (llama3:8b + nomic-embed-text) as LLM/embedding backend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_ragas_dataset(
    final_jsonl: Path,
    index_dir: str,
    embedder_model: str = "nomic-embed-text",
    llm_model: str = "llama3:8b",
    top_k: int = 5,
    limit: int = 10,
) -> list[dict]:
    """Run RAG pipeline on ground truth and collect inputs for RAGAS.

    Returns list of dicts with:
        question, answer, contexts, ground_truth
    """
    from annrag.rag.embedder import OllamaEmbedder
    from annrag.rag.index import VectorIndex
    from annrag.rag.pipeline import generate_answer, build_context

    index = VectorIndex.load(Path(index_dir))
    embedder = OllamaEmbedder(model=embedder_model)

    qa_pairs = []
    with open(final_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))

    # Limit for speed
    qa_pairs = qa_pairs[:limit]
    logger.info("Building RAGAS dataset for %d queries", len(qa_pairs))

    dataset = []
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        ground_truth = qa["ground_truth_answer"]

        # Embed + retrieve
        query_vec = embedder.embed(question)
        results = index.search(query_vec, top_k=top_k)
        contexts = [chunk.text[:500] for chunk, _ in results]

        # Generate answer
        context_str = build_context(results)
        answer = generate_answer(question, context_str, model=llm_model)

        dataset.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        logger.info("Processed %d / %d", i + 1, len(qa_pairs))
        print(f"  {i + 1}/{len(qa_pairs)}: {question[:50]}...")

    return dataset


def run_ragas(
    dataset: list[dict],
    llm_model: str = "llama3:8b",
    embedder_model: str = "nomic-embed-text",
) -> dict:
    """Run RAGAS evaluation on the dataset using Ollama."""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    from datasets import Dataset

    # Setup Ollama LLM + embeddings for RAGAS
    ollama_llm = LangchainLLMWrapper(Ollama(model=llm_model))
    ollama_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=embedder_model)
    )

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset)

    # Run evaluation
    result = evaluate(
        dataset=hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ollama_llm,
        embeddings=ollama_embeddings,
    )

    return {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
    }
