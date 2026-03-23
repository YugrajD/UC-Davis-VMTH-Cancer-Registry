"""Claude API client for Tier 4 free-form reasoning classification."""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from environment
    return _client


def claude_classify(prompt: str, timeout: int = 30) -> str:
    """Send a classification prompt to Claude and return the response text."""
    message = _get_client().messages.create(
        model=_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    return message.content[0].text
