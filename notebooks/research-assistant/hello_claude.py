"""Week 1 sanity check: confirm the Anthropic SDK and API key are working.

Usage:
    python hello_claude.py
"""

from __future__ import annotations

import os
import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    model = os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-4-5")
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=128,
        messages=[
            {"role": "user", "content": "In one sentence, what is Retrieval-Augmented Generation?"}
        ],
    )

    print(f"Model: {model}")
    for block in response.content:
        if getattr(block, "type", None) == "text":
            print(block.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
