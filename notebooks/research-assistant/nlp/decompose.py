"""Query decomposition with spaCy + KeyBERT.

Phase 2 Week 4 Day 3. Splits a single user question into a list of focused
sub-queries that can be dispatched to data sources in parallel.
"""

from __future__ import annotations

import re

SPACY_MODEL = "en_core_web_lg"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _fallback_phrases(query: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9\s-]", " ", query)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return []

    acronyms = [token for token in tokens if token.isupper() and len(token) > 1]
    keywords = [token.lower() for token in tokens if len(token) >= 4 and token.lower() not in _STOPWORDS]
    phrases: list[str] = []

    if acronyms:
        phrases.append(" ".join(acronyms))
    if keywords:
        phrases.append(" ".join(keywords[:4]))
    if len(keywords) >= 2:
        phrases.append(f"recent {' '.join(keywords[:2])}")
    if len(keywords) >= 3:
        phrases.append(f"applications of {' '.join(keywords[:3])}")
    return phrases


def decompose_query(query: str, max_subqueries: int = 4) -> list[str]:
    """Extract focused sub-queries from a broader research question.

    Example::

        "What are recent advances in RAG for medical AI?" ->
        ["RAG retrieval augmented generation",
         "medical AI NLP 2024",
         "RAG clinical applications"]
    """
    normalized = _normalize(query)
    if not normalized:
        raise ValueError("query must not be empty")

    phrases: list[str] = [normalized]

    try:
        import spacy

        nlp = spacy.load(SPACY_MODEL)
        doc = nlp(normalized)
        entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
        noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
        phrases.extend(entities)
        phrases.extend(noun_chunks[: max_subqueries * 2])
    except Exception:
        pass

    try:
        from keybert import KeyBERT

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            normalized,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=max_subqueries,
        )
        phrases.extend(text for text, _score in keywords if text)
    except Exception:
        phrases.extend(_fallback_phrases(normalized))

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        candidate = _normalize(phrase)
        key = candidate.lower()
        if len(candidate) < 3 or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= max(1, int(max_subqueries)):
            break

    return deduped
