"""Plotly chart helpers consumed by the Streamlit dashboard.

Phase 4 Week 7 Day 4.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _fallback_figure(kind: str, traces: list[dict[str, Any]], title: str) -> Any:
    return SimpleNamespace(data=traces, layout={"title": {"text": title}, "kind": kind})


def relevance_bar_chart(sources_df: Any) -> Any:
    """Return a bar chart of sources ranked by relevance proxy."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None

    if sources_df is None:
        rows: list[dict[str, Any]] = []
    elif hasattr(sources_df, "to_dict"):
        rows = list(sources_df.to_dict("records"))
    else:
        rows = list(sources_df)

    if not rows:
        return _fallback_figure("bar", [], "Source Relevance") if go is None else go.Figure()

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        distance = row.get("distance")
        relevance = 1.0 / (1.0 + float(distance)) if distance is not None else max(0.2, 0.55 - ((idx - 1) * 0.05))
        normalized.append(
            {
                "label": row.get("title") or row.get("doc_title") or row.get("url") or f"Source {idx}",
                "relevance": round(relevance, 4),
                "source": row.get("source", "unknown"),
            }
        )

    if go is None:
        return _fallback_figure("bar", normalized, "Source Relevance")

    fig = go.Figure()
    sources = sorted({row["source"] for row in normalized})
    for source_name in sources:
        subset = [row for row in normalized if row["source"] == source_name]
        fig.add_bar(name=source_name, x=[row["label"] for row in subset], y=[row["relevance"] for row in subset])
    fig.update_layout(title="Source Relevance", xaxis_title="Source", yaxis_title="Relevance", barmode="group")
    return fig


def topic_scatter(embeddings_2d: Any, labels: list[str]) -> Any:
    """Return a scatter chart for topic clusters or source groupings."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None

    points = list(embeddings_2d or [])
    if not points:
        return _fallback_figure("scatter", [], "Topic Clusters") if go is None else go.Figure()

    rows: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        label = labels[idx] if idx < len(labels) else f"Topic {idx + 1}"
        if isinstance(point, dict):
            x_val = float(point.get("x", idx))
            y_val = float(point.get("y", 0.0))
            title = point.get("title", label)
        else:
            x_val = float(point[0]) if len(point) > 0 else float(idx)
            y_val = float(point[1]) if len(point) > 1 else 0.0
            title = label
        rows.append({"x": x_val, "y": y_val, "label": label, "title": title})

    if go is None:
        return _fallback_figure("scatter", rows, "Topic Clusters")

    fig = go.Figure()
    for label in sorted({row["label"] for row in rows}):
        subset = [row for row in rows if row["label"] == label]
        fig.add_scatter(
            x=[row["x"] for row in subset],
            y=[row["y"] for row in subset],
            mode="markers",
            name=label,
            text=[row["title"] for row in subset],
        )
    fig.update_layout(title="Topic Clusters", xaxis_title="Cluster X", yaxis_title="Cluster Y")
    return fig


def agent_timeline(steps_log: list[dict[str, Any]]) -> Any:
    """Return a timeline view of reasoning, tool, and critique stages."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None

    rows = list(steps_log or [])
    if not rows:
        return _fallback_figure("timeline", [], "Agent Timeline") if go is None else go.Figure()

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        normalized.append(
            {
                "stage": str(row.get("stage", "unknown")),
                "label": str(row.get("label", f"step-{idx}")),
                "start": row.get("started_at"),
                "finish": row.get("finished_at") or row.get("started_at"),
                "step": int(row.get("step", idx)),
            }
        )

    if go is None:
        return _fallback_figure("timeline", normalized, "Agent Timeline")

    fig = go.Figure()
    for row in normalized:
        fig.add_bar(
            x=[1],
            y=[row["label"]],
            base=[row["start"]],
            orientation="h",
            name=row["stage"],
            hovertext=f"step={row['step']}",
            showlegend=False,
        )
    fig.update_layout(title="Agent Timeline", xaxis_title="Time", yaxis_title="Stage")
    return fig


def confidence_gauge(score: float) -> Any:
    """Return a gauge chart showing the final critique score."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None

    safe_score = max(0.0, min(10.0, float(score)))
    if go is None:
        return _fallback_figure("gauge", [{"value": safe_score}], "Critic Confidence")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=safe_score,
            number={"suffix": "/10"},
            title={"text": "Critic Confidence"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 4], "color": "#f4cccc"},
                    {"range": [4, 7], "color": "#ffe599"},
                    {"range": [7, 10], "color": "#d9ead3"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin={"l": 24, "r": 24, "t": 56, "b": 24})
    return fig
