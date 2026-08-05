import os
import sys
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sources import serpapi


def test_fetch_serpapi_requires_api_key(monkeypatch):
    monkeypatch.setattr(serpapi, "SERPAPI_API_KEY", "")

    with pytest.raises(RuntimeError, match="SERPAPI_API_KEY"):
        serpapi.fetch_serpapi("python")


@pytest.mark.skipif(
    not os.getenv("SERPAPI_API_KEY") or importlib.util.find_spec("serpapi") is None,
    reason="SERPAPI_API_KEY and serpapi package must be configured",
)
def test_fetch_serpapi_smoke():
    results = serpapi.fetch_serpapi("python", num_results=3)

    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert {"title", "snippet", "url", "source"}.issubset(item.keys())
        assert item["source"] == "web"
