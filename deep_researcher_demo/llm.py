"""OpenAI-compatible chat completion client."""

import os
from typing import Protocol

import httpx


Message = dict[str, str]


class ChatClient(Protocol):
    """Minimal async chat client used by agents."""

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return assistant text for a chat-completion request."""


class OpenAICompatibleClient:
    """Small wrapper around an OpenAI-compatible `/v1/chat/completions` API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:30000/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "dummy")
        self.timeout = timeout

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {data}") from exc
        return content or ""
