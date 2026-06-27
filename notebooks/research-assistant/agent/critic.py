"""Self-critique module powered by Claude Haiku 4.5.

Phase 3 Week 5 Day 1-2. The critic scores a draft report on completeness,
factual consistency, citation coverage, and clarity, and returns JSON with
specific revision requests. The ReAct loop revises at most twice before
returning the final answer.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

CRITIC_MODEL = os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-4-5")
MIN_ACCEPTABLE_SCORE = 7.0
MAX_REVISIONS = 2

CRITIC_SYSTEM_PROMPT = """You are a research quality reviewer. Evaluate the
following report on: completeness, factual consistency, citation coverage,
and clarity. Return a JSON object with numeric scores (0-10) for each
dimension, an overall score, a list of identified issues, and a list of
specific revision requests."""


def critique(report_text: str, original_query: str) -> dict[str, Any]:
    """Score ``report_text`` against ``original_query``.

    TODO Phase 3 Day 1-2: call Claude Haiku with ``CRITIC_SYSTEM_PROMPT`` and
    parse the JSON response. Return ``{scores, issues, suggestions}``.
    """
    raise NotImplementedError
