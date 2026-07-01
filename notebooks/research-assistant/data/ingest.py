"""PDF ingestion: parse -> chunk -> embed -> upsert into ChromaDB.

Phase 2 Week 3. Each chunk stores ``{text, source, page_num, doc_title}`` as
metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nlp.embeddings import embed
from data.vectorstore import upsert_chunks

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def parse_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract raw text per page using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("pymupdf is not installed. Install requirements first.") from exc

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[dict[str, Any]] = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                pages.append({"page_num": idx, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using LangChain when available."""
    if not text.strip():
        return []
    if overlap < 0 or chunk_size <= 0:
        raise ValueError("chunk_size must be > 0 and overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        return [c for c in chunks if c.strip()]
    except Exception:
        # Fallback keeps ingestion functional if LangChain splitter APIs change.
        chunks: list[str] = []
        step = chunk_size - overlap
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks


def ingest_pdf(file_path: str | Path, doc_title: str | None = None) -> int:
    """Full pipeline: parse the PDF, chunk it, embed, and upsert.

    Returns the number of chunks written to the vector store.
    """
    path = Path(file_path)
    title = doc_title or path.stem

    pages = parse_pdf(path)
    chunk_rows: list[dict[str, Any]] = []
    for page in pages:
        page_num = int(page["page_num"])
        for chunk in chunk_text(str(page.get("text", ""))):
            chunk_rows.append(
                {
                    "text": chunk,
                    "source": str(path),
                    "page_num": page_num,
                    "doc_title": title,
                }
            )

    if not chunk_rows:
        return 0

    vectors = embed([row["text"] for row in chunk_rows])
    upsert_chunks(chunk_rows, vectors)
    return len(chunk_rows)
