"""Text embeddings wrapper around ``BAAI/bge-small-en-v1.5``.

Phase 2 Week 3 Day 3. Returns 384-dimensional float vectors.
"""

from __future__ import annotations

from typing import Any

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_MODEL: Any = None


def _get_model() -> Any:
    """Load and cache the sentence-transformers model on first use."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install requirements first."
            ) from exc
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input string."""
    if not texts:
        return []

    model = _get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
