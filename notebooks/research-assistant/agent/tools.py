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

from data.sources.serpapi import fetch_serpapi

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
