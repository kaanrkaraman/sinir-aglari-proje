"""Tests for M3 chunking strategies."""

from annrag.rag.chunking import FixedSizeChunker, SentenceBoundaryChunker, get_strategy


SAMPLE_TEXT = " ".join([f"Word{i}" for i in range(1000)])
PAGE = "TestPage"


def test_fixed_chunk_count():
    chunker = FixedSizeChunker(chunk_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk(PAGE, SAMPLE_TEXT)
    assert len(chunks) > 0


def test_fixed_chunk_max_tokens():
    chunker = FixedSizeChunker(chunk_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk(PAGE, SAMPLE_TEXT)
    for c in chunks:
        assert c.token_count <= 100


def test_fixed_source_page():
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(PAGE, SAMPLE_TEXT)
    for c in chunks:
        assert c.source_page == PAGE


def test_fixed_chunk_ids_unique():
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(PAGE, SAMPLE_TEXT)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_strategy_name():
    assert FixedSizeChunker(512).name == "fixed_512"
    assert FixedSizeChunker(256).name == "fixed_256"


def test_sentence_chunker_basic():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunker = SentenceBoundaryChunker(target_tokens=10, overlap_sentences=0)
    chunks = chunker.chunk(PAGE, text)
    assert len(chunks) > 0


def test_sentence_source_page():
    text = "Hello world. Foo bar baz."
    chunker = SentenceBoundaryChunker()
    chunks = chunker.chunk(PAGE, text)
    for c in chunks:
        assert c.source_page == PAGE


def test_get_strategy_valid():
    for name in ["fixed_256", "fixed_512", "fixed_1024", "sentence"]:
        s = get_strategy(name)
        assert s.name == name


def test_get_strategy_invalid():
    import pytest
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("nonexistent")


def test_chunk_text_nonempty():
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(PAGE, SAMPLE_TEXT)
    for c in chunks:
        assert c.text.strip() != ""
