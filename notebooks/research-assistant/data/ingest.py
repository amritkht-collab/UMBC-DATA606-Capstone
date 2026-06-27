"""PDF ingestion: parse -> chunk -> embed -> upsert into ChromaDB.

Phase 2 Week 3. Each chunk stores ``{text, source, page_num, doc_title}`` as
metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def parse_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """TODO Phase 2 Day 1-2: extract raw text per page using PyMuPDF.

    Return a list of ``{page_num, text}`` dicts.
    """
    raise NotImplementedError


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """TODO Phase 2 Day 1-2: use LangChain's ``RecursiveCharacterTextSplitter``."""
    raise NotImplementedError


def ingest_pdf(file_path: str | Path, doc_title: str | None = None) -> int:
    """Full pipeline: parse the PDF, chunk it, embed, and upsert.

    Returns the number of chunks written to the vector store.
    """
    raise NotImplementedError
