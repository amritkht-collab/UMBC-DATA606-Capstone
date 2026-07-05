import logging
import os
import sys
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import react_loop


@pytest.mark.skipif(
    (
        not os.getenv("ANTHROPIC_API_KEY")
        or not os.getenv("SERPAPI_API_KEY")
        or importlib.util.find_spec("anthropic") is None
        or importlib.util.find_spec("serpapi") is None
    ),
    reason="Keys and runtime dependencies (anthropic, serpapi) must be configured",
)
def test_react_loop_integration_smoke(capsys, caplog):
    caplog.set_level(logging.DEBUG)

    exit_code = react_loop.main(["What is RAG in AI?", "--verbose", "--max-steps", "2"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip()
    assert "ACT:" in caplog.text
    assert "OBS:" in caplog.text


def test_run_agent_integrates_phase3_modules(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    create_calls: list[list[dict[str, object]]] = []
    recorded_turns: list[tuple[str, str]] = []
    saved_sessions: list[dict[str, object]] = []
    critique_calls: list[str] = []

    class _FakeMessages:
        def create(self, *, messages, **_kwargs):
            create_calls.append(list(messages))
            if len(create_calls) == 1:
                return SimpleNamespace(
                    stop_reason="tool_use",
                    content=[
                        SimpleNamespace(type="text", text="Need evidence from the vector store."),
                        SimpleNamespace(
                            type="tool_use",
                            id="tool-1",
                            name="retrieve_from_db",
                            input={"query": "medical rag", "top_k": 1},
                        ),
                    ],
                )
            if len(create_calls) == 2:
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[
                        SimpleNamespace(
                            type="text",
                            text="Draft answer with one source (https://example.com/paper).",
                        )
                    ],
                )
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(
                        type="text",
                        text="Revised answer with clearer support (https://example.com/paper).",
                    )
                ],
            )

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.Anthropic = _FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    monkeypatch.setattr(
        react_loop,
        "decompose_query",
        lambda question: [question, "medical AI retrieval", "clinical RAG evidence"],
    )
    monkeypatch.setattr(
        react_loop,
        "get_similar_sessions",
        lambda query, top_k=3: [
            {
                "query": "Earlier medical RAG question",
                "report": "Prior report summary.",
                "score": 7.5,
                "similarity": 0.82,
            }
        ],
    )
    monkeypatch.setattr(react_loop, "get_history", lambda: [{"role": "user", "content": "Previous question"}])
    monkeypatch.setattr(react_loop, "append_turn", lambda role, content: recorded_turns.append((role, content)))
    monkeypatch.setattr(
        react_loop,
        "tool_dispatcher",
        lambda name, payload: [
            {
                "source": "arxiv",
                "title": "Medical RAG Paper",
                "url": "https://example.com/paper",
                "doc_title": "Medical RAG Paper",
            }
        ],
    )

    def _fake_critique(answer, original_query):
        critique_calls.append(answer)
        if len(critique_calls) == 1:
            return {
                "scores": {"overall": 5.5},
                "issues": ["Needs a clearer synthesis."],
                "suggestions": ["Tighten the wording."],
                "model": "test-critic",
            }
        return {
            "scores": {"overall": 8.2},
            "issues": [],
            "suggestions": ["Looks good."],
            "model": "test-critic",
        }

    monkeypatch.setattr(react_loop, "critique", _fake_critique)
    monkeypatch.setattr(
        react_loop,
        "save_session",
        lambda query, report, sources, score: saved_sessions.append(
            {"query": query, "report": report, "sources": sources, "score": score}
        )
        or 1,
    )

    answer = react_loop.run_agent("What are recent advances in RAG for medical AI?", max_steps=3)

    assert answer == "Revised answer with clearer support (https://example.com/paper)."
    assert recorded_turns == [
        ("user", "What are recent advances in RAG for medical AI?"),
        ("assistant", "Revised answer with clearer support (https://example.com/paper)."),
    ]
    assert len(saved_sessions) == 1
    assert saved_sessions[0]["score"] == 8.2
    assert saved_sessions[0]["sources"][0]["url"] == "https://example.com/paper"
    first_prompt = create_calls[0][0]["content"]
    assert "Sub-queries to consider" in first_prompt
    assert "clinical RAG evidence" in first_prompt
    assert "Related prior sessions" in first_prompt
    assert len(critique_calls) == 2
    assert "Critique JSON" in create_calls[2][-1]["content"]
