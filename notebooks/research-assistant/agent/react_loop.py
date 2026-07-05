"""Core ReAct controller for the research agent.

Phase 1 (Week 1, Day 3-5): implements the THINK -> ACT -> OBSERVE loop with a
single tool (``web_search``). Subsequent phases add more tools, RAG retrieval,
self-critique, and memory integration.

The loop is driven by Anthropic's native tool-use API: Claude either returns
a final ``text`` block (``stop_reason == "end_turn"``) or one or more
``tool_use`` blocks (``stop_reason == "tool_use"``). When tools are requested
we dispatch them locally, send the ``tool_result`` blocks back, and continue
until the model produces a final answer or ``MAX_STEPS`` is reached.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.critic import MAX_REVISIONS, MIN_ACCEPTABLE_SCORE, critique
from agent.memory import append_turn, get_history, get_similar_sessions, save_session
from agent.tools import TOOL_REGISTRY, claude_tool_specs, tool_dispatcher
from nlp.decompose import decompose_query

load_dotenv()

logger = logging.getLogger(__name__)

MAX_STEPS = 6
MAX_TOKENS = 1024
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL_MAIN", "claude-opus-4-5")
OBS_PREVIEW_CHARS = 400

SYSTEM_PROMPT = """You are a research assistant agent.

Your job is to answer the user's research question accurately and concisely
by combining your own reasoning with evidence gathered from external tools.

Available tools:
- web_search(query, num_results=5): general-purpose Google web search via
  SerpAPI. Use it for definitions, current events, or background context.
- fetch_papers(query, max_results=5): search ArXiv, PubMed, and Semantic
    Scholar for academic papers.
- retrieve_from_db(query, top_k=5): retrieve relevant chunks from local
    ChromaDB for RAG grounding.
- summarize_doc(text, max_sentences=3): compress long abstracts or retrieved
    evidence into a concise research summary.

Follow the ReAct pattern on every turn:
  THINK: state in one short sentence what you need to do next.
  ACT:   call exactly one tool when external evidence is needed.
  OBS:   read the tool result before deciding the next step.

Rules:
- Prefer calling tools when answering factual or technical questions.
- For academic/research queries, prefer fetch_papers and retrieve_from_db
    before finalizing the answer.
- Use summarize_doc when fetched evidence is long enough that a shorter
    synthesis will improve the final response.
- Cite source URLs inline in parentheses, e.g. (https://example.com).
- Once you have enough evidence, stop calling tools and answer in a single
  well-structured paragraph (3-6 sentences).
- Never invent URLs or facts that did not appear in tool output.
"""


def _safe_decompose(question: str) -> list[str]:
    try:
        return decompose_query(question)
    except Exception as exc:
        logger.warning("query decomposition failed: %s", exc)
        return [question]


def _safe_similar_sessions(question: str) -> list[dict[str, Any]]:
    try:
        return get_similar_sessions(question, top_k=3)
    except Exception as exc:
        logger.warning("similar-session lookup failed: %s", exc)
        return []


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "None"
    return "\n".join(
        f"- {turn.get('role', 'unknown')}: {turn.get('content', '').strip()}" for turn in history
    )


def _format_similar_sessions(sessions: list[dict[str, Any]]) -> str:
    if not sessions:
        return "None"
    lines: list[str] = []
    for session in sessions:
        lines.append(
            "- "
            f"score={session.get('score', 0):.1f}, "
            f"similarity={session.get('similarity', 0):.2f}, "
            f"query={session.get('query', '')}, "
            f"report={str(session.get('report', ''))[:160]}"
        )
    return "\n".join(lines)


def _build_initial_prompt(
    question: str,
    subqueries: list[str],
    history: list[dict[str, str]],
    similar_sessions: list[dict[str, Any]],
) -> str:
    extra_subqueries = [item for item in subqueries if item.strip() and item.strip() != question.strip()]
    parts = [
        f"Original question:\n{question}",
        "Sub-queries to consider:\n" + ("\n".join(f"- {item}" for item in extra_subqueries) if extra_subqueries else "None"),
        "Recent conversation history:\n" + _format_history(history),
        "Related prior sessions:\n" + _format_similar_sessions(similar_sessions),
    ]
    return "\n\n".join(parts)


def _collect_source_rows(output: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(output, list):
        candidates = output
    elif isinstance(output, dict):
        candidates = [output]
    else:
        return rows

    for item in candidates:
        if not isinstance(item, dict):
            continue
        if any(key in item for key in ("source", "url", "title", "doc_title", "page_num")):
            rows.append(dict(item))
    return rows


def _dedupe_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("url", "")).strip(),
            str(row.get("title", row.get("doc_title", ""))).strip().lower(),
            str(row.get("source", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _review_score(review: dict[str, Any]) -> float:
    try:
        return float(review.get("scores", {}).get("overall", MIN_ACCEPTABLE_SCORE))
    except Exception:
        return MIN_ACCEPTABLE_SCORE


def _critique_feedback(question: str, answer: str, review: dict[str, Any]) -> str:
    return (
        "Revise the draft answer using the critique below. Keep the answer concise, factual, and cited.\n\n"
        f"Original question:\n{question}\n\n"
        f"Previous draft:\n{answer}\n\n"
        f"Critique JSON:\n{json.dumps(review, indent=2, default=str)}"
    )


def _preview(text: str, limit: int = OBS_PREVIEW_CHARS) -> str:
    """Return a single-line, length-capped preview of ``text`` for logs."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _log_assistant_turn(content_blocks: list[Any], step: int) -> None:
    """Print THINK / ACT lines from an assistant content block list."""
    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text" and block.text.strip():
            logger.info("step %d  THINK: %s", step, _preview(block.text))
        elif btype == "tool_use":
            logger.info(
                "step %d  ACT:   %s(%s)",
                step,
                block.name,
                json.dumps(block.input, separators=(",", ":")),
            )


def _execute_tool_uses(content_blocks: list[Any], step: int) -> list[dict[str, Any]]:
    """Dispatch every ``tool_use`` block and return ``tool_result`` blocks."""
    results: list[dict[str, Any]] = []
    collected_sources: list[dict[str, Any]] = []
    step_events: list[dict[str, Any]] = []
    for block in content_blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        tool_started_at = _utcnow_iso()
        try:
            output = tool_dispatcher(block.name, dict(block.input))
            payload = json.dumps(output, default=str)
            is_error = False
            collected_sources.extend(_collect_source_rows(output))
        except Exception as exc:  # surface the error to Claude rather than crashing
            logger.exception("step %d  tool %s failed", step, block.name)
            payload = f"Tool {block.name} raised: {exc}"
            is_error = True
        tool_finished_at = _utcnow_iso()

        logger.info("step %d  OBS:   %s", step, _preview(payload))
        step_events.append(
            {
                "step": step,
                "stage": "tool",
                "label": block.name,
                "started_at": tool_started_at,
                "finished_at": tool_finished_at,
                "duration_ms": None,
                "is_error": is_error,
                "preview": _preview(payload),
            }
        )
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": payload,
                "is_error": is_error,
            }
        )
    return results, collected_sources, step_events


def _final_text(content_blocks: list[Any]) -> str:
    """Concatenate all ``text`` blocks in a Claude response."""
    return "\n".join(
        block.text for block in content_blocks if getattr(block, "type", None) == "text"
    ).strip()


def _run_reasoning_pass(
    client: Any,
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_steps: int,
    max_tokens: int,
    tools: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    conversation = list(messages)
    collected_sources: list[dict[str, Any]] = []
    steps_log: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        started_at = _utcnow_iso()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=conversation,
        )
        finished_at = _utcnow_iso()
        _log_assistant_turn(response.content, step)
        steps_log.append(
            {
                "step": step,
                "stage": "reason",
                "label": response.stop_reason,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": None,
                "is_error": False,
                "preview": _preview(_final_text(response.content) or response.stop_reason),
            }
        )
        conversation.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = _final_text(response.content)
            logger.info("ANSWER: %s", _preview(answer, limit=200))
            return answer, collected_sources, conversation, steps_log

        tool_results, source_rows, tool_events = _execute_tool_uses(response.content, step)
        collected_sources.extend(source_rows)
        steps_log.extend(tool_events)
        conversation.append({"role": "user", "content": tool_results})

    logger.warning("Hit max_steps=%d without a final answer.", max_steps)
    return (
        "Agent stopped after reaching the maximum number of reasoning steps.",
        collected_sources,
        conversation,
        steps_log,
    )


def run_agent_session(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """Run the full agent flow and return answer, sources, critique, and UI metadata."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic SDK is not installed. Install requirements first.") from exc

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env first.")
    if not TOOL_REGISTRY:
        raise RuntimeError("No tools registered. Check agent.tools imports.")

    client = anthropic.Anthropic()
    tools = claude_tool_specs()
    prior_history = get_history()
    append_turn("user", question)
    subqueries = _safe_decompose(question)
    similar_sessions = _safe_similar_sessions(question)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _build_initial_prompt(question, subqueries, prior_history, similar_sessions),
        }
    ]

    logger.info("=" * 72)
    logger.info("Q: %s", question)
    logger.info("model=%s  tools=%s  max_steps=%d", model, [t["name"] for t in tools], max_steps)
    logger.info("=" * 72)

    answer = ""
    source_rows: list[dict[str, Any]] = []
    steps_log: list[dict[str, Any]] = []
    review: dict[str, Any] = {
        "scores": {"overall": MIN_ACCEPTABLE_SCORE},
        "issues": [],
        "suggestions": [],
        "model": "unreviewed",
    }

    for revision in range(MAX_REVISIONS + 1):
        answer, current_sources, messages, current_steps = _run_reasoning_pass(
            client,
            messages,
            model=model,
            max_steps=max_steps,
            max_tokens=max_tokens,
            tools=tools,
        )
        source_rows.extend(current_sources)
        steps_log.extend(current_steps)

        critique_started_at = _utcnow_iso()
        try:
            review = critique(answer, question)
        except Exception as exc:
            logger.warning("critique failed: %s", exc)
            review = {
                "scores": {"overall": MIN_ACCEPTABLE_SCORE},
                "issues": [],
                "suggestions": [],
                "model": "critique-error",
            }
        critique_finished_at = _utcnow_iso()
        score = _review_score(review)
        steps_log.append(
            {
                "step": len(steps_log) + 1,
                "stage": "critique",
                "label": f"revision-{revision}",
                "started_at": critique_started_at,
                "finished_at": critique_finished_at,
                "duration_ms": None,
                "is_error": review.get("model") == "critique-error",
                "preview": f"overall={score:.1f}",
            }
        )

        logger.info("critic overall=%.1f model=%s", score, review.get("model", "unknown"))
        if score >= MIN_ACCEPTABLE_SCORE or revision >= MAX_REVISIONS:
            break

        messages.append(
            {
                "role": "user",
                "content": _critique_feedback(question, answer, review),
            }
        )

    deduped_sources = _dedupe_sources(source_rows)
    append_turn("assistant", answer)
    try:
        session_id = save_session(question, answer, deduped_sources, _review_score(review))
    except Exception as exc:
        logger.warning("session persistence failed: %s", exc)
        session_id = None

    return {
        "question": question,
        "answer": answer,
        "sources": deduped_sources,
        "subqueries": subqueries,
        "history": get_history(),
        "similar_sessions": similar_sessions,
        "critique": review,
        "score": _review_score(review),
        "steps_log": steps_log,
        "session_id": session_id,
        "model": model,
    }


def run_agent(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Run the ReAct loop for a single question and return the final answer."""
    return run_agent_session(
        question,
        model=model,
        max_steps=max_steps,
        max_tokens=max_tokens,
    )["answer"]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research agent on one question.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What is RAG in AI?",
        help="Research question (default: %(default)r).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show DEBUG logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    answer = run_agent(args.question, model=args.model, max_steps=args.max_steps)
    print()
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
