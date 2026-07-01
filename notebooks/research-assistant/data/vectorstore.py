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
    """Create or load a persistent Chroma collection."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed. Install requirements first.") from exc

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    """Store chunks with their embeddings and metadata."""
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    collection = init_collection(collection_name)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        page_num = chunk.get("page_num", 0)
        chunk_id = str(chunk.get("id") or f"{source}:{page_num}:{idx}")

        ids.append(chunk_id)
        documents.append(str(chunk.get("text", "")))
        metadatas.append(
            {
                "source": str(source),
                "page_num": int(page_num) if isinstance(page_num, (int, float)) else 0,
                "doc_title": str(chunk.get("doc_title", "")),
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, collection_name: str = DEFAULT_COLLECTION) -> list[dict[str, Any]]:
    """Embed ``query`` and return the top-K most similar chunks with metadata."""
    if top_k <= 0:
        return []

    from nlp.embeddings import embed

    collection = init_collection(collection_name)
    query_embedding = embed([query])[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    rows: list[dict[str, Any]] = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        payload = dict(meta or {})
        payload["text"] = doc
        payload["distance"] = float(distance)
        rows.append(payload)
    return rows
