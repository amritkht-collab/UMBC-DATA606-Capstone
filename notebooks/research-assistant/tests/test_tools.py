import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import tools


def test_registry_contains_phase1_and_phase2_tools():
    assert "web_search" in tools.TOOL_REGISTRY
    assert "fetch_papers" in tools.TOOL_REGISTRY
    assert "retrieve_from_db" in tools.TOOL_REGISTRY
    assert "summarize_doc" in tools.TOOL_REGISTRY
    specs = tools.claude_tool_specs()
    assert any(spec["name"] == "summarize_doc" for spec in specs)


def test_fetch_papers_best_effort(monkeypatch):
    monkeypatch.setattr(tools, "fetch_arxiv", lambda query, max_results=5: [{"title": "A", "url": "u1", "source": "arxiv"}])
    monkeypatch.setattr(tools, "fetch_pubmed", lambda query, max_results=5: [{"title": "A", "url": "u1", "source": "pubmed"}])

    def _boom(query, limit=5):
        raise RuntimeError("s2 down")

    monkeypatch.setattr(tools, "fetch_s2", _boom)

    papers = tools.fetch_papers("rag", max_results=3)
    assert len(papers) == 1
    assert papers[0]["url"] == "u1"
    assert set(papers[0]["success_sources"]) == {"arxiv", "pubmed"}
    assert papers[0]["source_errors"]["semantic_scholar"] == "s2 down"


def test_fetch_papers_returns_system_error_row_if_all_sources_fail(monkeypatch):
    def _boom_arxiv(query, max_results=5):
        raise RuntimeError("arxiv down")

    def _boom_pubmed(query, max_results=5):
        raise RuntimeError("pubmed down")

    def _boom_s2(query, limit=5):
        raise RuntimeError("s2 down")

    monkeypatch.setattr(tools, "fetch_arxiv", _boom_arxiv)
    monkeypatch.setattr(tools, "fetch_pubmed", _boom_pubmed)
    monkeypatch.setattr(tools, "fetch_s2", _boom_s2)

    papers = tools.fetch_papers("rag", max_results=2)
    assert len(papers) == 1
    assert papers[0]["source"] == "system"
    assert papers[0]["success_sources"] == []
    assert set(papers[0]["source_errors"].keys()) == {
        "arxiv",
        "pubmed",
        "semantic_scholar",
    }


def test_retrieve_from_db_clamps_top_k(monkeypatch):
    called = {}

    def _fake_retrieve(query, top_k):
        called["query"] = query
        called["top_k"] = top_k
        return []

    monkeypatch.setattr(tools, "retrieve", _fake_retrieve)

    tools.retrieve_from_db("medical rag", top_k=999)
    assert called["query"] == "medical rag"
    assert called["top_k"] == 20


def test_summarize_doc_uses_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = tools.summarize_doc(
        (
            "Retrieval-augmented generation improves answer grounding by attaching external evidence. "
            "In medical AI, this reduces unsupported claims and improves traceability. "
            "Recent work also studies better chunking and reranking for clinical corpora. "
            "The strongest systems combine retrieval quality with robust evaluation."
        ),
        max_sentences=2,
    )

    assert result["method"] == "extractive-fallback"
    assert result["sentences"] == 2
    assert "medical" in result["summary"].lower() or "retrieval" in result["summary"].lower()


def test_summarize_doc_rejects_empty_text():
    try:
        tools.summarize_doc("   ")
        assert False, "Expected ValueError for empty text"
    except ValueError as exc:
        assert "must not be empty" in str(exc)
