import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import react_loop, tools
from data.sources import arxiv as arxiv_source
from data.sources import pubmed as pubmed_source
from data.sources import semantic_scholar as s2_source
from data.sources import serpapi as serpapi_source
from tests.conftest import skip_for_live_api_failure


LIVE_QUERY = "retrieval augmented generation in medical AI"


def _require_package(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not installed in the current environment")


@pytest.mark.live_api
def test_fetch_arxiv_live(require_live_api_tests):
    _require_package("arxiv")
    try:
        rows = arxiv_source.fetch_arxiv(LIVE_QUERY, max_results=2)
    except Exception as exc:
        skip_for_live_api_failure(exc)

    assert rows
    for row in rows:
        assert {"title", "abstract", "url", "authors", "year", "source"}.issubset(row.keys())
        assert row["source"] == "arxiv"


@pytest.mark.live_api
def test_fetch_pubmed_live(require_live_api_tests):
    _require_package("Bio")
    if not os.getenv("ENTREZ_EMAIL"):
        pytest.skip("ENTREZ_EMAIL must be configured for live PubMed tests")
    try:
        rows = pubmed_source.fetch_pubmed(LIVE_QUERY, max_results=2)
    except Exception as exc:
        skip_for_live_api_failure(exc)

    assert rows
    for row in rows:
        assert {"title", "abstract", "url", "authors", "year", "source"}.issubset(row.keys())
        assert row["source"] == "pubmed"


@pytest.mark.live_api
def test_fetch_semantic_scholar_live(require_live_api_tests):
    try:
        rows = s2_source.fetch_s2(LIVE_QUERY, limit=2)
    except Exception as exc:
        skip_for_live_api_failure(exc)

    assert rows
    for row in rows:
        assert {"title", "abstract", "url", "authors", "year", "citationCount", "source"}.issubset(row.keys())
        assert row["source"] == "semantic_scholar"


@pytest.mark.live_api
def test_fetch_serpapi_live(require_live_api_tests):
    _require_package("serpapi")
    if not os.getenv("SERPAPI_API_KEY"):
        pytest.skip("SERPAPI_API_KEY must be configured for live SerpAPI tests")
    try:
        rows = serpapi_source.fetch_serpapi(LIVE_QUERY, num_results=2)
    except Exception as exc:
        skip_for_live_api_failure(exc)

    assert rows
    for row in rows:
        assert {"title", "snippet", "url", "source"}.issubset(row.keys())
        assert row["source"] == "web"


@pytest.mark.live_api
def test_fetch_papers_live_aggregate(require_live_api_tests):
    try:
        papers = tools.fetch_papers(LIVE_QUERY, max_results=2)
    except Exception as exc:
        skip_for_live_api_failure(exc)

    assert papers
    non_system = [paper for paper in papers if paper.get("source") != "system"]
    assert non_system
    assert any(paper.get("source") in {"arxiv", "pubmed", "semantic_scholar"} for paper in non_system)


@pytest.mark.live_api
@pytest.mark.functional
def test_run_agent_session_live(require_live_api_tests):
    _require_package("anthropic")
    _require_package("serpapi")
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY must be configured for live agent tests")
    if not os.getenv("SERPAPI_API_KEY"):
        pytest.skip("SERPAPI_API_KEY must be configured for live agent tests")

    model = os.getenv("CLAUDE_MODEL_FAST") or os.getenv("CLAUDE_MODEL_MAIN") or "claude-haiku-4-5"
    try:
        result = react_loop.run_agent_session("What is retrieval augmented generation in AI?", model=model, max_steps=2)
    except Exception as exc:
        skip_for_live_api_failure(exc)

    assert result["answer"].strip()
    assert isinstance(result["sources"], list)
    assert isinstance(result["subqueries"], list)
    assert isinstance(result["critique"], dict)
    assert isinstance(result["steps_log"], list)
    assert result["model"] == model