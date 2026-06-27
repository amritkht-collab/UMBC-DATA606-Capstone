"""Text embeddings wrapper around ``BAAI/bge-small-en-v1.5``.

Phase 2 Week 3 Day 3. Returns 384-dimensional float vectors.
"""

from __future__ import annotations

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


def embed(texts: list[str]) -> list[list[float]]:
    """TODO Phase 2 Day 3: load the model via ``sentence-transformers`` and
    return one vector per input string."""
    raise NotImplementedError
