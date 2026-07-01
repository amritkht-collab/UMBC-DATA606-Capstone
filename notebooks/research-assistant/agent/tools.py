"""Tool registry and dispatcher for the research agent.

Each tool is a small function plus a JSON-style schema that Claude can call.
Tools are added incrementally across the project:

  Phase 1: web_search       (SerpAPI)
  Phase 2: fetch_papers     (ArXiv + PubMed + Semantic Scholar)
           retrieve_from_db (ChromaDB RAG)
           summarize_doc    (BART or Claude Sonnet)
"""

from __future__ import annotations

from typing import Any, Callable

from data.sources.arxiv import fetch_arxiv
from data.sources.pubmed import fetch_pubmed
from data.sources.semantic_scholar import fetch_s2
from data.sources.serpapi import fetch_serpapi
from data.vectorstore import retrieve

ToolFn = Callable[..., Any]

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(name: str, description: str, schema: dict[str, Any]) -> Callable[[ToolFn], ToolFn]:
    """Decorator that registers a function in :data:`TOOL_REGISTRY`."""

    def decorator(fn: ToolFn) -> ToolFn:
        TOOL_REGISTRY[name] = {"description": description, "schema": schema, "fn": fn}
        return fn

    return decorator


def tool_dispatcher(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Look up a tool by name and call it with the provided input dict."""
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name]["fn"](**tool_input)


def claude_tool_specs() -> list[dict[str, Any]]:
    """Format the registry into the schema Anthropic's API expects."""
    return [
        {
            "name": name,
            "description": entry["description"],
            "input_schema": entry["schema"],
        }
        for name, entry in TOOL_REGISTRY.items()
    ]


@register_tool(
    name="web_search",
    description=(
        "Run a Google web search via SerpAPI and return the top organic results. "
        "Use this for general-purpose questions, current events, definitions, "
        "or whenever academic-paper sources are not specifically required."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string."},
            "num_results": {
                "type": "integer",
                "description": "Number of organic results to return (1-10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
def web_search(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Web search tool backed by SerpAPI."""
    return fetch_serpapi(query=query, num_results=num_results)


@register_tool(
    name="fetch_papers",
    description=(
        "Search academic papers across ArXiv, PubMed, and Semantic Scholar. "
        "Returns normalized paper metadata with title, abstract, url, authors, year, source."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Paper search query string."},
            "max_results": {
                "type": "integer",
                "description": "Max papers per source (1-10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
def fetch_papers(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Aggregate papers from multiple sources with best-effort behavior."""
    per_source = max(1, min(int(max_results), 10))
    out: list[dict[str, Any]] = []
    source_errors: dict[str, str] = {}
    success_sources: list[str] = []

    try:
        arxiv_rows = fetch_arxiv(query=query, max_results=per_source)
        out.extend(arxiv_rows)
        success_sources.append("arxiv")
    except Exception as exc:
        source_errors["arxiv"] = str(exc)

    try:
        pubmed_rows = fetch_pubmed(query=query, max_results=per_source)
        out.extend(pubmed_rows)
        success_sources.append("pubmed")
    except Exception as exc:
        source_errors["pubmed"] = str(exc)

    try:
        s2_rows = fetch_s2(query=query, limit=per_source)
        out.extend(s2_rows)
        success_sources.append("semantic_scholar")
    except Exception as exc:
        source_errors["semantic_scholar"] = str(exc)

    # Deduplicate by URL first, then by lowercase title.
    seen_url: set[str] = set()
    seen_title: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in out:
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", "")).strip().lower()
        if url and url in seen_url:
            continue
        if title and title in seen_title:
            continue
        if url:
            seen_url.add(url)
        if title:
            seen_title.add(title)
        deduped.append(row)

    papers = deduped[: per_source * 3]

    if papers:
        for paper in papers:
            paper["success_sources"] = list(success_sources)
            if source_errors:
                paper["source_errors"] = dict(source_errors)
        return papers

    if source_errors and not papers:
        return [
            {
                "title": "",
                "abstract": "",
                "url": "",
                "authors": [],
                "year": None,
                "source": "system",
                "success_sources": list(success_sources),
                "source_errors": dict(source_errors),
            }
        ]

    return papers


@register_tool(
    name="retrieve_from_db",
    description=(
        "Retrieve top-k relevant chunks from the local ChromaDB vector store "
        "for retrieval-augmented generation."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic retrieval query."},
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to return (1-20).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
def retrieve_from_db(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Local vector retrieval tool."""
    return retrieve(query=query, top_k=max(1, min(int(top_k), 20)))
