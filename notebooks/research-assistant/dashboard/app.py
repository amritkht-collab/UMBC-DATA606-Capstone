"""Streamlit entry point.

Phase 4 Week 7. Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only when UI deps are absent
    class _MissingStreamlit:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("streamlit is required to run the dashboard UI.")

    st = _MissingStreamlit()

from agent.memory import get_history
from agent.react_loop import DEFAULT_MODEL, MAX_STEPS, run_agent_session
from dashboard.charts import agent_timeline, confidence_gauge, relevance_bar_chart, topic_scatter


def init_session_state() -> None:
    defaults = {
        "current_result": None,
        "last_error": "",
        "max_steps": MAX_STEPS,
        "model": DEFAULT_MODEL,
        "query_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def summarize_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, source in enumerate(sources, start=1):
        distance = source.get("distance")
        rows.append(
            {
                "rank": idx,
                "source": source.get("source", "unknown"),
                "title": source.get("title") or source.get("doc_title") or f"Source {idx}",
                "url": source.get("url", ""),
                "page_num": source.get("page_num"),
                "distance": distance,
                "relevance": round(1.0 / (1.0 + float(distance)), 4) if distance is not None else round(max(0.2, 0.55 - ((idx - 1) * 0.05)), 4),
            }
        )
    return rows


def build_topic_points(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    points: list[dict[str, Any]] = []
    labels: list[str] = []
    source_offsets: dict[str, float] = {}
    for idx, source in enumerate(sources, start=1):
        label = str(source.get("source", "unknown"))
        labels.append(label)
        source_offsets.setdefault(label, float(len(source_offsets)))
        points.append(
            {
                "x": float(idx),
                "y": float(source_offsets[label]) + (0.1 if source.get("distance") is None else float(source.get("distance"))),
                "title": source.get("title") or source.get("doc_title") or label,
            }
        )
    return points, labels


def build_report_markdown(result: dict[str, Any]) -> str:
    critique = result.get("critique", {})
    issues = critique.get("issues", [])
    suggestions = critique.get("suggestions", [])
    lines = [
        f"# Research Assistant Report\n",
        f"Question: {result.get('question', '')}\n",
        f"Model: {result.get('model', '')}\n",
        f"Score: {result.get('score', 0):.1f}/10\n",
        "## Answer\n",
        result.get("answer", ""),
        "\n## Sub-queries\n",
        *(f"- {item}" for item in result.get("subqueries", [])),
        "\n## Critique Issues\n",
        *(f"- {item}" for item in issues or ["None"]),
        "\n## Critique Suggestions\n",
        *(f"- {item}" for item in suggestions or ["None"]),
    ]
    return "\n".join(lines)


def build_export_payload(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)


def run_query(query: str) -> None:
    try:
        result = run_agent_session(
            query,
            model=st.session_state.model,
            max_steps=int(st.session_state.max_steps),
        )
        st.session_state.current_result = result
        st.session_state.last_error = ""
        st.session_state.query_count += 1
    except Exception as exc:
        st.session_state.last_error = str(exc)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Research Assistant")
        st.caption("UMBC DATA606 Capstone")
        if st.button("New Query", use_container_width=True):
            st.session_state.current_result = None
            st.session_state.last_error = ""
        st.divider()
        st.subheader("Settings")
        st.session_state.model = st.text_input("Claude model", value=st.session_state.model)
        st.session_state.max_steps = st.slider("Max reasoning steps", min_value=1, max_value=10, value=int(st.session_state.max_steps))
        st.caption(f"Queries this session: {st.session_state.query_count}")
        st.divider()
        st.subheader("History")
        history = get_history()
        if history:
            for turn in history[-6:]:
                st.write(f"{turn.get('role', 'unknown').title()}: {turn.get('content', '')[:120]}")
        else:
            st.caption("No saved conversation turns yet.")
        current = st.session_state.current_result or {}
        similar_sessions = current.get("similar_sessions", [])
        if similar_sessions:
            st.divider()
            st.subheader("Related Sessions")
            for session in similar_sessions:
                st.caption(
                    f"{session.get('similarity', 0):.2f} similarity | {session.get('query', '')}"
                )


def render_report_tab(result: dict[str, Any]) -> None:
    critique = result.get("critique", {})
    st.subheader("Answer")
    st.write(result.get("answer", ""))
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Critic Score", f"{result.get('score', 0):.1f}/10")
        st.caption(f"Review model: {critique.get('model', 'n/a')}")
    with col2:
        st.metric("Sources Used", len(result.get("sources", [])))
        st.metric("Sub-queries", len(result.get("subqueries", [])))
    st.subheader("Critique")
    issues = critique.get("issues", [])
    suggestions = critique.get("suggestions", [])
    st.write("Issues")
    if issues:
        for item in issues:
            st.write(f"- {item}")
    else:
        st.write("- None")
    st.write("Suggestions")
    if suggestions:
        for item in suggestions:
            st.write(f"- {item}")
    else:
        st.write("- None")
    st.download_button("Download report (Markdown)", data=build_report_markdown(result), file_name="research-report.md", mime="text/markdown")
    st.download_button("Download session (JSON)", data=build_export_payload(result), file_name="research-session.json", mime="application/json")


def render_sources_tab(result: dict[str, Any]) -> None:
    rows = summarize_sources(result.get("sources", []))
    if not rows:
        st.info("No source rows were captured for this run.")
        return
    st.dataframe(rows, use_container_width=True)
    for row in rows:
        with st.expander(f"{row['rank']}. {row['title']}"):
            st.write(f"Source: {row['source']}")
            st.write(f"Relevance: {row['relevance']:.2f}")
            if row.get("page_num") is not None:
                st.write(f"Page: {row['page_num']}")
            if row.get("url"):
                st.write(row["url"])


def render_charts_tab(result: dict[str, Any]) -> None:
    sources = summarize_sources(result.get("sources", []))
    points, labels = build_topic_points(sources)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(confidence_gauge(result.get("score", 0.0)), use_container_width=True)
        st.plotly_chart(relevance_bar_chart(sources), use_container_width=True)
    with chart_col2:
        st.plotly_chart(topic_scatter(points, labels), use_container_width=True)
        st.plotly_chart(agent_timeline(result.get("steps_log", [])), use_container_width=True)


def render_main() -> None:
    st.title("Autonomous Research Assistant")
    st.caption("Multi-source research, RAG retrieval, self-critique, and session memory in one workflow.")
    query = st.chat_input("Ask a research question...")
    if query:
        st.chat_message("user").write(query)
        with st.spinner("Running the research agent..."):
            run_query(query)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    result = st.session_state.current_result
    if not result:
        st.info("Submit a question to generate a report, source table, and diagnostic charts.")
        return

    tabs = st.tabs(["Report", "Sources", "Charts"])
    with tabs[0]:
        render_report_tab(result)
    with tabs[1]:
        render_sources_tab(result)
    with tabs[2]:
        render_charts_tab(result)

def main() -> None:
    st.set_page_config(page_title="Research Assistant", layout="wide")
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
