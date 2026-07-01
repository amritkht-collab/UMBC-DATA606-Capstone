import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import ingest, vectorstore
from nlp import embeddings as embeddings_module


@pytest.mark.skipif(
    importlib.util.find_spec("fitz") is None or importlib.util.find_spec("chromadb") is None,
    reason="Requires pymupdf (fitz) and chromadb packages",
)
def test_ingest_pdf_then_retrieve_from_db(tmp_path, monkeypatch):
    import fitz

    db_dir = tmp_path / "chroma"
    pdf_path = tmp_path / "fixture.pdf"

    # Create a tiny one-page PDF fixture locally for end-to-end ingestion.
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "RAG improves retrieval quality in medical AI systems.")
    doc.save(pdf_path)
    doc.close()

    def _fake_embed(texts):
        vectors = []
        for text in texts:
            low = text.lower()
            if "rag" in low:
                vectors.append([1.0, 0.0, 0.0])
            elif "medical" in low:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    monkeypatch.setattr(vectorstore, "CHROMA_DB_DIR", str(db_dir))
    monkeypatch.setattr(ingest, "embed", _fake_embed)
    monkeypatch.setattr(embeddings_module, "embed", _fake_embed)

    count = ingest.ingest_pdf(pdf_path, doc_title="fixture-doc")
    assert count > 0

    rows = vectorstore.retrieve("rag", top_k=1)
    assert len(rows) == 1
    assert "rag" in rows[0]["text"].lower()
    assert rows[0]["doc_title"] == "fixture-doc"
