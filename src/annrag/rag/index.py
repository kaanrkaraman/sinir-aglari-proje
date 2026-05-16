"""Vector index — M4.

Stores chunk embeddings in a FAISS index for fast similarity search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from annrag.rag.models import Chunk

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS-backed vector index for chunk retrieval.

    Stores:
        - FAISS flat L2 index (vectors)
        - Parallel list of Chunk metadata

    Args:
        dim: Embedding dimension (e.g. 768 for nomic-embed-text).
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vecs
        self._chunks: list[Chunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        """Add a single chunk and its vector to the index."""
        vec = np.array([vector], dtype=np.float32)
        # Normalize for cosine similarity
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._chunks.append(chunk)

    def add_batch(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Add multiple chunks and vectors at once."""
        mat = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(mat)
        self._index.add(mat)
        self._chunks.extend(chunks)
        logger.info("Index now has %d vectors", self.size)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Find top-k most similar chunks.

        Returns:
            List of (Chunk, score) tuples, sorted by score descending.
        """
        vec = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, directory: Path) -> None:
        """Save index and metadata to directory."""
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        meta = [c.model_dump() for c in self._chunks]
        (directory / "chunks_meta.jsonl").write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in meta),
            encoding="utf-8",
        )
        logger.info("Saved index (%d vectors) to %s", self.size, directory)

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        """Load index and metadata from directory."""
        index = faiss.read_index(str(directory / "index.faiss"))
        dim = index.d

        obj = cls(dim)
        obj._index = index

        lines = (directory / "chunks_meta.jsonl").read_text(encoding="utf-8").splitlines()
        obj._chunks = [Chunk(**json.loads(line)) for line in lines if line.strip()]

        logger.info("Loaded index (%d vectors) from %s", obj.size, directory)
        return obj
