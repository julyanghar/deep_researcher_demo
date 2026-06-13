"""OpenAI-compatible chat completion client."""

import json
import os
import threading
import time
from typing import Protocol

import httpx


Message = dict[str, str]

# Optional per-call latency log (JSONL). Enabled via LLM_CALL_LOG=<path>.
# Each line: role tag (inferred from the system prompt's first line),
# wall-clock start/end, elapsed seconds, and prompt/completion sizes.
_CALL_LOG_PATH = os.getenv("LLM_CALL_LOG", "")
_CALL_LOG_LOCK = threading.Lock()


def _infer_call_tag(messages: list[Message]) -> str:
    if messages and messages[0].get("role") == "system":
        first_line = (messages[0].get("content") or "").split("\n", 1)[0].strip()
        if first_line:
            return first_line[:40]
    return "UNKNOWN"


def _log_call(record: dict) -> None:
    if not _CALL_LOG_PATH:
        return
    line = json.dumps(record, ensure_ascii=False)
    with _CALL_LOG_LOCK:
        with open(_CALL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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
        timeout: float | None = None,
    ) -> None:
        if timeout is None:
            # Long-generation workloads (detailed summaries on slow/TP-reduced
            # backends) can exceed the old 120s default and die with
            # httpx.ReadTimeout. Overridable via LLM_TIMEOUT seconds.
            timeout = float(os.getenv("LLM_TIMEOUT", "120"))
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
        start_wall = time.time()
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        elapsed = time.perf_counter() - start

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {data}") from exc

        usage = data.get("usage") or {}
        _log_call(
            {
                "tag": _infer_call_tag(messages),
                "start_ts": round(start_wall, 3),
                "end_ts": round(start_wall + elapsed, 3),
                "elapsed_s": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        return content or ""
