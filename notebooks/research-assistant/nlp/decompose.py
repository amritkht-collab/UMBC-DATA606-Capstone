"""Query decomposition with spaCy + KeyBERT.

Phase 2 Week 4 Day 3. Splits a single user question into a list of focused
sub-queries that can be dispatched to data sources in parallel.
"""

from __future__ import annotations

SPACY_MODEL = "en_core_web_lg"


def decompose_query(query: str, max_subqueries: int = 4) -> list[str]:
    """TODO Phase 2 Day 3: extract NER + KeyBERT keyphrases and emit sub-queries.

    Example::

        "What are recent advances in RAG for medical AI?" ->
        ["RAG retrieval augmented generation",
         "medical AI NLP 2024",
         "RAG clinical applications"]
    """
    raise NotImplementedError
