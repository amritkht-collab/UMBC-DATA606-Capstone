"""Short-term (conversation) and long-term (SQLite) memory.

Phase 3 Week 5 deliverable. The short-term store is an in-memory list of
``{role, content}`` dicts capped at the most recent exchanges. The long-term
store persists past sessions to SQLite so the agent can surface similar
past research from the Streamlit sidebar.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nlp.embeddings import embed

MAX_HISTORY_TURNS = 10
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "logs" / "sessions.sqlite3"

# Short-term conversation buffer; reset per session.
_history: list[dict[str, str]] = []


def _db_path() -> Path:
    path = Path(DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            query TEXT NOT NULL,
            report TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            score REAL NOT NULL,
            query_embedding_json TEXT NOT NULL
        )
        """
    )
    return conn


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _fallback_vector(text: str) -> list[float]:
    buckets = [0.0, 0.0, 0.0, 0.0]
    for idx, token in enumerate(text.lower().split()):
        buckets[idx % len(buckets)] += sum(ord(ch) for ch in token)
    norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / norm for value in buckets]


def _embed_query(query: str) -> list[float]:
    try:
        vectors = embed([query])
        if vectors and vectors[0]:
            return [float(value) for value in vectors[0]]
    except Exception:
        pass
    return _fallback_vector(query)


def append_turn(role: str, content: str) -> None:
    """Append a chat turn and trim to the most recent ``MAX_HISTORY_TURNS``."""
    _history.append({"role": role, "content": content})
    overflow = len(_history) - MAX_HISTORY_TURNS
    if overflow > 0:
        del _history[:overflow]


def get_history() -> list[dict[str, str]]:
    return list(_history)


def save_session(query: str, report: str, sources: list[dict[str, Any]], score: float) -> int:
    """Persist a completed session and return its SQLite row id."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not report.strip():
        raise ValueError("report must not be empty")

    payload = json.dumps(sources, default=str)
    vector = json.dumps(_embed_query(query))

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (created_at, query, report, sources_json, score, query_embedding_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                query,
                report,
                payload,
                float(score),
                vector,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_similar_sessions(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return the most similar saved sessions ranked by cosine similarity."""
    if not query.strip():
        raise ValueError("query must not be empty")

    query_vector = _embed_query(query)
    limit = max(1, int(top_k))
    ranked: list[dict[str, Any]] = []

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, query, report, sources_json, score, query_embedding_json
            FROM sessions
            ORDER BY id DESC
            """
        ).fetchall()

    for row in rows:
        stored_vector = json.loads(row["query_embedding_json"])
        similarity = _cosine_similarity(query_vector, [float(value) for value in stored_vector])
        ranked.append(
            {
                "id": int(row["id"]),
                "created_at": row["created_at"],
                "query": row["query"],
                "report": row["report"],
                "sources": json.loads(row["sources_json"]),
                "score": float(row["score"]),
                "similarity": round(similarity, 4),
            }
        )

    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked[:limit]
