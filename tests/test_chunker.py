import pytest

from src.ingestion.chunker import chunk_document


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_windows_overlap_and_tail():
    chunks = chunk_document(words(95), "d", 40, 10)
    assert [(c.start_token, c.end_token) for c in chunks] == [(0, 40), (30, 70), (60, 95)]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_neighbouring_chunks_share_overlap_tokens():
    a, b = chunk_document(words(60), "d", 40, 10)[:2]
    assert a.text.split()[-10:] == b.text.split()[:10]


@pytest.mark.parametrize("n,expected", [(3, 1), (40, 1), (41, 2), (70, 2), (71, 3)])
def test_no_redundant_tail_chunk(n, expected):
    assert len(chunk_document(words(n), "d", 40, 10)) == expected


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_text_yields_no_chunks(text):
    assert chunk_document(text, "d", 40, 10) == []


def test_whitespace_is_normalised():
    assert chunk_document("a\n\nb\tc", "d", 40, 10)[0].text == "a b c"


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 11)])
def test_overlap_must_be_smaller_than_size(size, overlap):
    with pytest.raises(ValueError):
        chunk_document("a b c", "d", size, overlap)


def test_chunk_ids_are_deterministic_unique_and_doc_scoped():
    a1 = chunk_document(words(95), "doc-a", 40, 10)
    a2 = chunk_document(words(95), "doc-a", 40, 10)
    b = chunk_document(words(95), "doc-b", 40, 10)
    assert [c.chunk_id for c in a1] == [c.chunk_id for c in a2]
    assert len({c.chunk_id for c in a1 + b}) == len(a1) + len(b)


def test_metadata_and_doc_id_propagate():
    chunks = chunk_document(words(95), "d", 40, 10, {"source": "unit"})
    assert all(c.doc_id == "d" and c.metadata == {"source": "unit"} for c in chunks)
