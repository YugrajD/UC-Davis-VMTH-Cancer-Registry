"""Jan local API client for LLM-based evaluation."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_IP = os.getenv("JAN_TAILSCALE_IP")
_API_KEY = os.getenv("JAN_API_KEY")
_HOSTNAME = os.getenv("JAN_HOSTNAME")
_MODEL = os.getenv("JAN_MODEL")


def _base_url() -> str:
    return f"http://{_IP}:1337/v1"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_API_KEY}",
        "Host": _HOSTNAME,
        "Content-Type": "application/json",
    }


def list_models() -> list[dict]:
    """Return available models from the Jan server."""
    response = requests.get(f"{_base_url()}/models", headers=_headers(), timeout=10)
    response.raise_for_status()
    return response.json()["data"]


def chat(prompt: str, model: str | None = None, timeout: int = 60) -> str:
    """Send a chat message and return the assistant's reply."""
    payload = {
        "model": model or _MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(
        f"{_base_url()}/chat/completions",
        headers=_headers(),
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
