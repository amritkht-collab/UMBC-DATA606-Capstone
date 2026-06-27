"""ChromaDB wrapper with a persistent client.

Phase 2 Week 3 Day 4-5. Persists to ``CHROMA_DB_DIR`` so embeddings survive
restarts.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_store")
DEFAULT_COLLECTION = "research"
DEFAULT_TOP_K = 5


def init_collection(name: str = DEFAULT_COLLECTION) -> Any:
    """TODO Phase 2 Day 4-5: create/load a ``chromadb.PersistentClient`` collection."""
    raise NotImplementedError


def upsert_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    """Store chunks with their embeddings and metadata."""
    raise NotImplementedError


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, collection_name: str = DEFAULT_COLLECTION) -> list[dict[str, Any]]:
    """Embed ``query`` and return the top-K most similar chunks with metadata."""
    raise NotImplementedError
