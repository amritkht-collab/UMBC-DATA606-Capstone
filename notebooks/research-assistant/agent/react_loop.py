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
import json
import logging
import os
import sys
from typing import Any

import anthropic
from dotenv import load_dotenv

from agent.tools import TOOL_REGISTRY, claude_tool_specs, tool_dispatcher

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

Follow the ReAct pattern on every turn:
  THINK: state in one short sentence what you need to do next.
  ACT:   call exactly one tool when external evidence is needed.
  OBS:   read the tool result before deciding the next step.

Rules:
- Prefer calling a tool when the user asks about facts, news, or definitions
  you are not fully confident about.
- Cite source URLs inline in parentheses, e.g. (https://example.com).
- Once you have enough evidence, stop calling tools and answer in a single
  well-structured paragraph (3-6 sentences).
- Never invent URLs or facts that did not appear in tool output.
"""


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
    for block in content_blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        try:
            output = tool_dispatcher(block.name, dict(block.input))
            payload = json.dumps(output, default=str)
            is_error = False
        except Exception as exc:  # surface the error to Claude rather than crashing
            logger.exception("step %d  tool %s failed", step, block.name)
            payload = f"Tool {block.name} raised: {exc}"
            is_error = True

        logger.info("step %d  OBS:   %s", step, _preview(payload))
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": payload,
                "is_error": is_error,
            }
        )
    return results


def _final_text(content_blocks: list[Any]) -> str:
    """Concatenate all ``text`` blocks in a Claude response."""
    return "\n".join(
        block.text for block in content_blocks if getattr(block, "type", None) == "text"
    ).strip()


def run_agent(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Run the ReAct loop for a single question and return the final answer."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env first.")
    if not TOOL_REGISTRY:
        raise RuntimeError("No tools registered. Check agent.tools imports.")

    client = anthropic.Anthropic()
    tools = claude_tool_specs()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    logger.info("=" * 72)
    logger.info("Q: %s", question)
    logger.info("model=%s  tools=%s  max_steps=%d", model, [t["name"] for t in tools], max_steps)
    logger.info("=" * 72)

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        _log_assistant_turn(response.content, step)

        if response.stop_reason != "tool_use":
            answer = _final_text(response.content)
            logger.info("ANSWER: %s", _preview(answer, limit=200))
            return answer

        tool_results = _execute_tool_uses(response.content, step)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    logger.warning("Hit max_steps=%d without a final answer.", max_steps)
    return "Agent stopped after reaching the maximum number of reasoning steps."


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
