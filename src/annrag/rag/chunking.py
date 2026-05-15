"""Chunking strategies — M3.

Four strategies as specified in PROGRESS.md:
  - fixed_256   : fixed token window, 32-token overlap
  - fixed_512   : fixed token window, 64-token overlap
  - fixed_1024  : fixed token window, 128-token overlap
  - sentence    : sentence-boundary aware, target ~512 tokens
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from annrag.rag.models import Chunk

# Simple whitespace tokenizer (~= cl100k_base word count).
# Replace with tiktoken on a machine with internet access.
def _tokenize(text: str) -> list[str]:
    return text.split()

def _count_tokens(text: str) -> int:
    return len(_tokenize(text))

def _make_chunk_id(page_title: str, index: int) -> str:
    return f"{page_title}::{index}"


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class ChunkingStrategy(Protocol):
    @property
    def name(self) -> str: ...
    def chunk(self, page_title: str, text: str) -> list[Chunk]: ...


# ── Fixed-size chunker ────────────────────────────────────────────────────────

class FixedSizeChunker:
    """Overlapping fixed-token windows."""

    def __init__(self, chunk_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens

    @property
    def name(self) -> str:
        return f"fixed_{self._chunk_tokens}"

    def chunk(self, page_title: str, text: str) -> list[Chunk]:
        words = _tokenize(text)
        stride = self._chunk_tokens - self._overlap_tokens
        chunks: list[Chunk] = []
        idx = 0
        i = 0
        char_cursor = 0

        while i < len(words):
            window = words[i : i + self._chunk_tokens]
            chunk_text = " ".join(window)

            start_char = text.find(chunk_text[:40], char_cursor)
            if start_char == -1:
                start_char = char_cursor
            end_char = min(start_char + len(chunk_text), len(text))
            char_cursor = start_char

            chunks.append(Chunk(
                chunk_id=_make_chunk_id(page_title, idx),
                source_page=page_title,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                token_count=len(window),
                strategy=self.name,
            ))
            idx += 1
            i += stride

        return chunks


# ── Sentence-boundary chunker ─────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


class SentenceBoundaryChunker:
    """Accumulates sentences until token budget, then starts a new chunk."""

    def __init__(self, target_tokens: int = 512, overlap_sentences: int = 2) -> None:
        self._target = target_tokens
        self._overlap = overlap_sentences

    @property
    def name(self) -> str:
        return "sentence"

    def chunk(self, page_title: str, text: str) -> list[Chunk]:
        sentences = _split_sentences(text)
        chunks: list[Chunk] = []
        idx = 0
        i = 0

        while i < len(sentences):
            bucket: list[str] = []
            token_count = 0

            while i < len(sentences):
                s = sentences[i]
                s_tokens = _count_tokens(s)
                if bucket and token_count + s_tokens > self._target:
                    break
                bucket.append(s)
                token_count += s_tokens
                i += 1

            if not bucket:
                bucket = [sentences[i]]
                token_count = _count_tokens(bucket[0])
                i += 1

            chunk_text = " ".join(bucket)
            start_char = text.find(bucket[0][:40])
            if start_char == -1:
                start_char = 0
            end_char = min(start_char + len(chunk_text), len(text))

            chunks.append(Chunk(
                chunk_id=_make_chunk_id(page_title, idx),
                source_page=page_title,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                strategy=self.name,
            ))
            idx += 1
            i = max(i - self._overlap, i - len(bucket) + 1, i)

        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────

ALL_STRATEGIES: dict[str, ChunkingStrategy] = {
    "fixed_256":  FixedSizeChunker(chunk_tokens=256,  overlap_tokens=32),
    "fixed_512":  FixedSizeChunker(chunk_tokens=512,  overlap_tokens=64),
    "fixed_1024": FixedSizeChunker(chunk_tokens=1024, overlap_tokens=128),
    "sentence":   SentenceBoundaryChunker(target_tokens=512, overlap_sentences=2),
}

def get_strategy(name: str) -> ChunkingStrategy:
    try:
        return ALL_STRATEGIES[name]
    except KeyError:
        valid = ", ".join(ALL_STRATEGIES)
        raise ValueError(f"Unknown strategy {name!r}. Valid: {valid}") from None
