"""Chunk data model — M3."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A single text chunk produced by a chunking strategy."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique ID: '<page_title>::<index>'")
    source_page: str = Field(description="Wikivoyage page title (normalized).")
    text: str = Field(description="Chunk text content.")
    start_char: int = Field(description="Start character offset in original article.")
    end_char: int = Field(description="End character offset in original article.")
    token_count: int = Field(description="Approximate token count (cl100k_base).")
    strategy: str = Field(description="Chunking strategy name, e.g. 'fixed_512'.")
