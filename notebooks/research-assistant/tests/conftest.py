import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Make sure the project root is available for imports during test runs.
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
