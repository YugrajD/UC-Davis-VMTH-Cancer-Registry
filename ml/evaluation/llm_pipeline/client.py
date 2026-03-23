"""Ollama local API client for LLM-based evaluation."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_IP = os.getenv("TAILSCALE_IP")
_PORT = os.getenv("API_PORT", "11434")
_MODEL = os.getenv("OLLAMA_MODEL")


def _base_url() -> str:
    return f"http://{_IP}:{_PORT}"


def list_models() -> list[dict]:
    """Return available models from the Ollama server."""
    response = requests.get(f"{_base_url()}/api/tags", timeout=10)
    response.raise_for_status()
    return response.json()["models"]


def chat(prompt: str, model: str | None = None, timeout: int = 60) -> str:
    """Send a chat message and return the assistant's reply."""
    payload = {
        "model": model or _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(
        f"{_base_url()}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


if __name__ == "__main__":
    print("Available models:")
    for m in list_models():
        print(f"  {m['name']}")

    print("\nTest message:")
    print(chat("Hello, respond in one sentence."))
