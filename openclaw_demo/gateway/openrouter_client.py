from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class OpenRouterResponse:
    content: str
    raw: dict[str, Any]


def _normalize_key(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def call_openrouter_chat(*, prompt: str, system: str | None = None) -> OpenRouterResponse:
    api_key_raw = os.getenv("OPENROUTER_API_KEY", "")
    api_key = _normalize_key(api_key_raw)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model = os.getenv("OPENCLAW_MODEL", "openrouter/auto").strip()
    if model == "openrouter/openrouter/auto":
        model = "openrouter/auto"

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "dsperse-openclaw-demo"),
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = ""

    return OpenRouterResponse(content=content, raw=data)
