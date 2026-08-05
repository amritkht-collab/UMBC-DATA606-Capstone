"""Self-critique module powered by Claude Haiku 4.5.

Phase 3 Week 5 Day 1-2. The critic scores a draft report on completeness,
factual consistency, citation coverage, and clarity, and returns JSON with
specific revision requests. The ReAct loop revises at most twice before
returning the final answer.
"""

from __future__ import annotations

import json
import os
import re
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

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp_score(value: float) -> float:
    return max(0.0, min(10.0, round(float(value), 1)))


def _extract_json_object(text: str) -> dict[str, Any]:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in critic response")
    return json.loads(match.group(0))


def _heuristic_critique(report_text: str, original_query: str) -> dict[str, Any]:
    report = report_text.strip()
    query_terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", original_query)
        if len(token) >= 4
    }
    report_terms = set(re.findall(r"[A-Za-z0-9]+", report.lower()))

    coverage_ratio = 1.0 if not query_terms else len(query_terms & report_terms) / len(query_terms)
    sentence_count = max(1, len([part for part in re.split(r"[.!?]+", report) if part.strip()]))
    report_len = len(report)
    citation_count = len(re.findall(r"https?://", report))

    completeness = _clamp_score(4.0 + (coverage_ratio * 4.0) + min(report_len / 400.0, 2.0))
    factual_consistency = _clamp_score(6.0 + min(citation_count, 2) + (0.5 if report_len >= 120 else -1.0))
    citation_coverage = _clamp_score((citation_count * 3.5) if citation_count else 2.5)
    clarity = _clamp_score(5.5 + min(sentence_count, 4) * 0.8 - (1.0 if report_len < 80 else 0.0))
    overall = _clamp_score(
        (completeness + factual_consistency + citation_coverage + clarity) / 4.0
    )

    issues: list[str] = []
    suggestions: list[str] = []
    if coverage_ratio < 0.6:
        issues.append("The draft does not address enough of the original query.")
        suggestions.append("Add a direct answer that covers the main concepts from the user question.")
    if citation_count == 0:
        issues.append("The draft contains no supporting citations or URLs.")
        suggestions.append("Cite at least one supporting source URL from tool output.")
    if report_len < 120:
        issues.append("The draft is too short to fully justify its conclusion.")
        suggestions.append("Expand the explanation with one or two evidence-backed details.")
    if sentence_count < 2:
        issues.append("The draft is compressed into a single sentence and may be hard to follow.")
        suggestions.append("Split the response into a few concise sentences with a clearer progression.")

    if not issues:
        suggestions.append("Only make minor wording edits before returning the answer.")

    return {
        "scores": {
            "completeness": completeness,
            "factual_consistency": factual_consistency,
            "citation_coverage": citation_coverage,
            "clarity": clarity,
            "overall": overall,
        },
        "issues": issues,
        "suggestions": suggestions,
        "model": "heuristic-fallback",
    }


def critique(report_text: str, original_query: str) -> dict[str, Any]:
    """Score ``report_text`` against ``original_query``.

    Use Claude Haiku when the SDK and API key are available; otherwise fall
    back to a deterministic local rubric so the module remains testable.
    """
    if not report_text.strip():
        raise ValueError("report_text must not be empty")
    if not original_query.strip():
        raise ValueError("original_query must not be empty")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=CRITIC_MODEL,
                max_tokens=600,
                system=CRITIC_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Original query:\n{original_query}\n\n"
                            f"Draft report:\n{report_text}\n\n"
                            "Return JSON only."
                        ),
                    }
                ],
            )
            text_blocks = [
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ]
            payload = _extract_json_object("\n".join(text_blocks))
            scores = payload.get("scores", {})
            issues = payload.get("issues", [])
            suggestions = payload.get("suggestions") or payload.get("revision_requests") or []
            overall = payload.get("overall") or scores.get("overall")
            normalized_scores = {
                "completeness": _clamp_score(scores.get("completeness", 0.0)),
                "factual_consistency": _clamp_score(scores.get("factual_consistency", 0.0)),
                "citation_coverage": _clamp_score(scores.get("citation_coverage", 0.0)),
                "clarity": _clamp_score(scores.get("clarity", 0.0)),
                "overall": _clamp_score(
                    overall
                    if overall is not None
                    else sum(
                        float(scores.get(key, 0.0))
                        for key in (
                            "completeness",
                            "factual_consistency",
                            "citation_coverage",
                            "clarity",
                        )
                    )
                    / 4.0
                ),
            }
            return {
                "scores": normalized_scores,
                "issues": [str(item) for item in issues],
                "suggestions": [str(item) for item in suggestions],
                "model": CRITIC_MODEL,
            }
        except Exception:
            pass

    return _heuristic_critique(report_text, original_query)
