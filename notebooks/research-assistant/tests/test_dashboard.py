import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import app, charts
from agent import react_loop


def test_run_agent_session_returns_dashboard_payload(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _FakeMessages:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    stop_reason="tool_use",
                    content=[
                        SimpleNamespace(type="tool_use", id="tool-1", name="fetch_papers", input={"query": "rag", "max_results": 1}),
                    ],
                )
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="Final report (https://example.com/paper).")],
            )

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.Anthropic = _FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    monkeypatch.setattr(react_loop, "get_history", lambda: [])
    monkeypatch.setattr(react_loop, "append_turn", lambda role, content: None)
    monkeypatch.setattr(react_loop, "decompose_query", lambda question: [question, "rag evidence"])
    monkeypatch.setattr(react_loop, "get_similar_sessions", lambda query, top_k=3: [])
    monkeypatch.setattr(
        react_loop,
        "tool_dispatcher",
        lambda _name, _payload: [{"source": "arxiv", "title": "Paper", "url": "https://example.com/paper", "distance": 0.2}],
    )
    monkeypatch.setattr(
        react_loop,
        "critique",
        lambda answer, question: {"scores": {"overall": 8.8}, "issues": [], "suggestions": ["Ship it."], "model": "test"},
    )
    monkeypatch.setattr(react_loop, "save_session", lambda *args, **kwargs: 41)

    result = react_loop.run_agent_session("What is RAG?", max_steps=2)

    assert result["answer"] == "Final report (https://example.com/paper)."
    assert result["score"] == 8.8
    assert result["session_id"] == 41
    assert result["sources"][0]["source"] == "arxiv"
    assert result["subqueries"][1] == "rag evidence"
    assert any(step["stage"] == "critique" for step in result["steps_log"])


def test_dashboard_helpers_build_exportable_views():
    result = {
        "question": "What is RAG?",
        "answer": "RAG combines retrieval with generation.",
        "model": "claude-test",
        "score": 8.4,
        "subqueries": ["What is RAG?", "retrieval augmented generation"],
        "sources": [
            {"source": "arxiv", "title": "Paper A", "url": "https://example.com/a", "distance": 0.25},
            {"source": "web", "title": "Paper B", "url": "https://example.com/b"},
        ],
        "critique": {"issues": ["Needs more citations."], "suggestions": ["Add one citation."], "model": "heuristic"},
        "steps_log": [
            {"stage": "reason", "label": "tool_use", "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:00:01+00:00", "step": 1},
        ],
    }

    source_rows = app.summarize_sources(result["sources"])
    topic_points, labels = app.build_topic_points(source_rows)
    markdown = app.build_report_markdown(result)
    payload = app.build_export_payload(result)

    assert source_rows[0]["relevance"] > source_rows[1]["relevance"]
    assert len(topic_points) == 2
    assert labels == ["arxiv", "web"]
    assert "# Research Assistant Report" in markdown
    assert '"question": "What is RAG?"' in payload


def test_chart_helpers_return_figures():
    sources = [
        {"source": "arxiv", "title": "Paper A", "distance": 0.1},
        {"source": "pubmed", "title": "Paper B", "distance": 0.4},
    ]
    timeline = [
        {"stage": "reason", "label": "tool_use", "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:00:01+00:00", "step": 1},
        {"stage": "critique", "label": "revision-0", "started_at": "2026-01-01T00:00:02+00:00", "finished_at": "2026-01-01T00:00:03+00:00", "step": 2},
    ]

    bar = charts.relevance_bar_chart(sources)
    scatter = charts.topic_scatter([{"x": 1, "y": 0.1, "title": "Paper A"}, {"x": 2, "y": 1.4, "title": "Paper B"}], ["arxiv", "pubmed"])
    gantt = charts.agent_timeline(timeline)
    gauge = charts.confidence_gauge(8.4)

    assert len(bar.data) >= 1
    assert len(scatter.data) >= 1
    assert len(gantt.data) >= 1
    assert len(gauge.data) == 1