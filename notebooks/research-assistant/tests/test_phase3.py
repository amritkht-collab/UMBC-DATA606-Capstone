import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import critic, memory
from nlp import decompose


def test_critique_returns_structured_feedback_without_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = critic.critique(
        "RAG can improve clinical NLP by grounding answers in retrieved evidence.",
        "What are recent advances in RAG for medical AI?",
    )

    assert set(result.keys()) == {"scores", "issues", "suggestions", "model"}
    assert result["model"] == "heuristic-fallback"
    assert set(result["scores"].keys()) == {
        "completeness",
        "factual_consistency",
        "citation_coverage",
        "clarity",
        "overall",
    }
    assert result["scores"]["overall"] >= 0.0
    assert result["suggestions"]


def test_memory_persists_and_ranks_similar_sessions(monkeypatch, tmp_path):
    db_path = tmp_path / "sessions.sqlite3"
    monkeypatch.setattr(memory, "DEFAULT_DB_PATH", db_path)

    def _fake_embed(texts):
        vectors = []
        for text in texts:
            low = text.lower()
            if "medical" in low:
                vectors.append([1.0, 0.0, 0.0])
            elif "finance" in low:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    monkeypatch.setattr(memory, "embed", _fake_embed)

    first_id = memory.save_session(
        "medical rag systems",
        "Report about medical RAG.",
        [{"source": "arxiv", "url": "https://example.com/1"}],
        8.5,
    )
    second_id = memory.save_session(
        "finance rag systems",
        "Report about finance RAG.",
        [{"source": "web", "url": "https://example.com/2"}],
        7.0,
    )

    matches = memory.get_similar_sessions("medical evidence retrieval", top_k=2)

    assert first_id > 0
    assert second_id > first_id
    assert len(matches) == 2
    assert matches[0]["query"] == "medical rag systems"
    assert matches[0]["similarity"] >= matches[1]["similarity"]
    assert matches[0]["sources"][0]["source"] == "arxiv"


def test_decompose_query_returns_unique_subqueries():
    subqueries = decompose.decompose_query(
        "What are recent advances in RAG for medical AI?",
        max_subqueries=4,
    )

    assert 1 <= len(subqueries) <= 4
    assert subqueries[0] == "What are recent advances in RAG for medical AI?"
    assert len({item.lower() for item in subqueries}) == len(subqueries)
    assert any("rag" in item.lower() or "medical" in item.lower() for item in subqueries[1:])


def test_decompose_query_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        decompose.decompose_query("   ")