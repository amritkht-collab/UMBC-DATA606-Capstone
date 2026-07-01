import logging
import os
import sys
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import react_loop


@pytest.mark.skipif(
    (
        not os.getenv("ANTHROPIC_API_KEY")
        or not os.getenv("SERPAPI_API_KEY")
        or importlib.util.find_spec("anthropic") is None
        or importlib.util.find_spec("serpapi") is None
    ),
    reason="Keys and runtime dependencies (anthropic, serpapi) must be configured",
)
def test_react_loop_integration_smoke(capsys, caplog):
    caplog.set_level(logging.DEBUG)

    exit_code = react_loop.main(["What is RAG in AI?", "--verbose", "--max-steps", "2"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip()
    assert "ACT:" in caplog.text
    assert "OBS:" in caplog.text
