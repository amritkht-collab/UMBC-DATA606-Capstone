"""Short-term (conversation) and long-term (SQLite) memory.

Phase 3 Week 5 deliverable. The short-term store is an in-memory list of
``{role, content}`` dicts capped at the most recent exchanges. The long-term
store persists past sessions to SQLite so the agent can surface similar
past research from the Streamlit sidebar.
"""

from __future__ import annotations

from typing import Any

MAX_HISTORY_TURNS = 10

# Short-term conversation buffer; reset per session.
_history: list[dict[str, str]] = []


def append_turn(role: str, content: str) -> None:
    """Append a chat turn and trim to the most recent ``MAX_HISTORY_TURNS``."""
    _history.append({"role": role, "content": content})
    overflow = len(_history) - MAX_HISTORY_TURNS
    if overflow > 0:
        del _history[:overflow]


def get_history() -> list[dict[str, str]]:
    return list(_history)


def save_session(query: str, report: str, sources: list[dict[str, Any]], score: float) -> int:
    """TODO Phase 3 Day 4-5: persist to SQLite (``sessions`` table)."""
    raise NotImplementedError


def get_similar_sessions(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """TODO Phase 3 Day 4-5: embed the query, find top-K past sessions."""
    raise NotImplementedError
