import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Make sure the project root is available for imports during test runs.
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))


def pytest_configure(config):
	config.addinivalue_line(
		"markers",
		"live_api: exercises real external APIs and may incur network cost or require credentials",
	)
	config.addinivalue_line(
		"markers",
		"functional: covers end-to-end runtime flows instead of isolated units",
	)


@pytest.fixture
def require_live_api_tests():
	if os.getenv("RUN_LIVE_API_TESTS") != "1":
		pytest.skip("Set RUN_LIVE_API_TESTS=1 to enable live API integration tests.")


def skip_for_live_api_failure(exc: Exception) -> None:
	message = str(exc)
	known_environment_failures = (
		"CERTIFICATE_VERIFY_FAILED",
		"self signed certificate",
		"APIConnectionError",
		"Connection error",
		"timed out",
		"Temporary failure",
		"Name or service not known",
	)
	if any(token.lower() in message.lower() for token in known_environment_failures):
		pytest.skip(f"Live API environment failure: {exc}")
	raise exc
