"""OpenAI-compatible local API client for LLM-based annotation (LM Studio)."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_HOST = os.getenv("LLM_HOST", "127.0.0.1")
_PORT = os.getenv("API_PORT", "1234")
_MODEL = os.getenv("LLM_MODEL")


def _base_url() -> str:
    return f"http://{_HOST}:{_PORT}"


def list_models() -> list[dict]:
    """Return available models from the server."""
    response = requests.get(f"{_base_url()}/v1/models", timeout=10)
    response.raise_for_status()
    return response.json()["data"]


def chat(prompt: str, model: str | None = None, timeout: int = 60) -> str:
    """Send a chat message and return the assistant's reply."""
    payload = {
        "model": model or _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(
        f"{_base_url()}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Available models:")
    for m in list_models():
        print(f"  {m['id']}")

    print("\nTest message:")
    print(chat("Hello, respond in one sentence."))
