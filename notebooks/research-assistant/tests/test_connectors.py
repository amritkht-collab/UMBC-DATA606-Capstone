import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sources import arxiv as arxiv_source
from data.sources import pubmed as pubmed_source
from data.sources import semantic_scholar as s2_source


def test_fetch_arxiv_normalization_with_mocked_module(monkeypatch):
    class _FakeSearch:
        def __init__(self, query, max_results, sort_by):
            self.query = query
            self.max_results = max_results
            self.sort_by = sort_by

        def results(self):
            yield SimpleNamespace(
                title="Paper A",
                summary="About RAG.",
                entry_id="https://arxiv.org/abs/1234.5678",
                authors=[SimpleNamespace(name="Alice"), SimpleNamespace(name="Bob")],
                published=SimpleNamespace(year=2025),
            )

    fake_arxiv = ModuleType("arxiv")
    fake_arxiv.Search = _FakeSearch
    fake_arxiv.SortCriterion = SimpleNamespace(Relevance="relevance")
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv)

    rows = arxiv_source.fetch_arxiv("rag", max_results=2)
    assert len(rows) == 1
    assert rows[0] == {
        "title": "Paper A",
        "abstract": "About RAG.",
        "url": "https://arxiv.org/abs/1234.5678",
        "authors": ["Alice", "Bob"],
        "year": 2025,
        "source": "arxiv",
    }


def test_fetch_pubmed_normalization_with_mocked_bio(monkeypatch):
    class _Handle:
        def __init__(self, payload=None):
            self.payload = payload

        def close(self):
            return None

    class _Entrez:
        email = ""

        @staticmethod
        def esearch(db, term, retmax, sort):
            return _Handle({"IdList": ["1"]})

        @staticmethod
        def read(handle):
            return handle.payload

        @staticmethod
        def efetch(db, id, rettype, retmode):
            return _Handle()

    class _Medline:
        @staticmethod
        def parse(handle):
            return iter(
                [
                    {
                        "PMID": "1",
                        "TI": "PubMed Paper",
                        "AB": "Normalized abstract",
                        "AU": ["Author One", "Author Two"],
                        "DP": "2024 Jan",
                    }
                ]
            )

    fake_bio = ModuleType("Bio")
    fake_bio.Entrez = _Entrez
    fake_bio.Medline = _Medline
    monkeypatch.setitem(sys.modules, "Bio", fake_bio)
    monkeypatch.setattr(pubmed_source, "ENTREZ_EMAIL", "tester@example.com")

    rows = pubmed_source.fetch_pubmed("rag", max_results=2)
    assert len(rows) == 1
    assert rows[0] == {
        "title": "PubMed Paper",
        "abstract": "Normalized abstract",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "authors": ["Author One", "Author Two"],
        "year": 2024,
        "source": "pubmed",
    }


def test_fetch_s2_normalization_with_mocked_http(monkeypatch):
    payload = {
        "data": [
            {
                "title": "S2 Paper",
                "abstract": "S2 abstract",
                "url": "https://www.semanticscholar.org/paper/abc",
                "authors": [{"name": "Researcher"}],
                "year": 2023,
                "citationCount": 42,
            }
        ]
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(request, timeout=20):
        return _Response()

    monkeypatch.setattr(s2_source, "urlopen", _fake_urlopen)

    rows = s2_source.fetch_s2("rag", limit=3)
    assert len(rows) == 1
    assert rows[0] == {
        "title": "S2 Paper",
        "abstract": "S2 abstract",
        "url": "https://www.semanticscholar.org/paper/abc",
        "authors": ["Researcher"],
        "year": 2023,
        "citationCount": 42,
        "source": "semantic_scholar",
    }
