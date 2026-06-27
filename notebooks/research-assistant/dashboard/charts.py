"""Plotly chart helpers consumed by the Streamlit dashboard.

Phase 4 Week 7 Day 4.
"""

from __future__ import annotations

from typing import Any


def relevance_bar_chart(sources_df: Any) -> Any:
    """TODO: ``px.bar`` of source name vs relevance score."""
    raise NotImplementedError


def topic_scatter(embeddings_2d: Any, labels: list[str]) -> Any:
    """TODO: ``px.scatter`` of UMAP-reduced embeddings colored by topic."""
    raise NotImplementedError


def agent_timeline(steps_log: list[dict[str, Any]]) -> Any:
    """TODO: ``px.timeline`` of agent steps with duration per stage."""
    raise NotImplementedError


def confidence_gauge(score: float) -> Any:
    """TODO: ``go.Indicator(mode='gauge+number')`` showing critic score."""
    raise NotImplementedError
