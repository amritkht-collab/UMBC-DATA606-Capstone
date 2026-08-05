import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import ingest


def test_chunk_text_returns_chunks_for_non_empty_text():
    text = " ".join(["token"] * 400)
    chunks = ingest.chunk_text(text, chunk_size=120, overlap=20)
    assert chunks
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_chunk_text_validates_window_arguments():
    with pytest.raises(ValueError):
        ingest.chunk_text("abc", chunk_size=50, overlap=50)

    with pytest.raises(ValueError):
        ingest.chunk_text("abc", chunk_size=0, overlap=1)
