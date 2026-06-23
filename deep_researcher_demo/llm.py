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

# Drift-measurement harvest (Exp A). When DRIFT_HARVEST=<path> is set, the
# relevant chat() calls append a record (prompt messages + decoded content +
# generated token_ids) to that JSONL. The offline harness reconstructs, per
# reused summary / r_t segment, the worker context (for KV^w) and the
# supervisor context (for KV*) by matching segment text across records, so no
# per-request id needs to be threaded through. Unset -> no overhead.
#
# Only the two call types Exp A actually consumes are harvested:
#   RESEARCH_SUMMARY_TEXT    -> worker context + s_i (reconstructs KV^w)
#   SUPERVISOR_DECISION_JSON -> supervisor context (interleaved s_i + r_t; KV*/KV^r)
# The rest (initial questions, query plan, the up-to-10k-token final report,
# repair) are dead weight. Override via DRIFT_HARVEST_TAGS (comma-sep; "*"=all).
_HARVEST_PATH = os.getenv("DRIFT_HARVEST", "")
_HARVEST_LOCK = threading.Lock()
_HARVEST_TAGS_RAW = os.getenv("DRIFT_HARVEST_TAGS", "RESEARCH_SUMMARY_TEXT,SUPERVISOR_DECISION_JSON")
_HARVEST_TAGS = (
    None if _HARVEST_TAGS_RAW.strip() == "*"
    else {t.strip() for t in _HARVEST_TAGS_RAW.split(",") if t.strip()}
)

# Truncated-summary KV-reuse fix (opt-in via KV_REUSE_TOKENIZER=<tokenizer path>).
# When a stored-for-reuse generation hits max_tokens (finish_reason="length"),
# the server only stored output[:-1] (the last sampled token has no KV), yet the
# API `content` includes that last token -> the downstream content-hash misses.
# We re-embed decode(token_ids[:-1]) so it matches the stored M-1 segment. This
# needs the SAME tokenizer the server uses. Unset -> feature off (no behavior
# change; truncated summaries simply remain non-reusable).
_TOKENIZER_PATH = os.getenv("KV_REUSE_TOKENIZER", "")
_TOKENIZER = None
_TOKENIZER_LOADED = False
_TOKENIZER_LOCK = threading.Lock()


def _get_reuse_tokenizer():
    """Lazily load the client tokenizer; returns None if disabled/unavailable."""
    global _TOKENIZER, _TOKENIZER_LOADED
    if _TOKENIZER_LOADED:
        return _TOKENIZER
    with _TOKENIZER_LOCK:
        if not _TOKENIZER_LOADED:
            if _TOKENIZER_PATH:
                try:
                    from transformers import AutoTokenizer

                    _TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_PATH)
                except Exception as exc:  # noqa: BLE001 - best-effort feature
                    print(
                        f"[kv-reuse] tokenizer load failed ({_TOKENIZER_PATH!r}): "
                        f"{exc}; truncated-summary fix disabled",
                        flush=True,
                    )
                    _TOKENIZER = None
            _TOKENIZER_LOADED = True
    return _TOKENIZER


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


def _harvest_call(record: dict) -> None:
    if not _HARVEST_PATH:
        return
    line = json.dumps(record, ensure_ascii=False)
    with _HARVEST_LOCK:
        with open(_HARVEST_PATH, "a", encoding="utf-8") as f:
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
        store_generated_kv: bool = False,
        tag: str | None = None,
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
        store_generated_kv: bool = False,
        tag: str | None = None,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if store_generated_kv:
            # Per-request opt-in: tell the LMCache blend backend to store this
            # request's decode-generated KV as a reusable segment. vLLM forwards
            # top-level kv_transfer_params into sampling_params.extra_args, where
            # LMCache's extract_request_configs picks up the lmcache.* keys.
            payload["kv_transfer_params"] = {"lmcache.blend_store_generated": True}
            # Ask for the exact generated token ids so we can drop the last one
            # on truncation (see the finish_reason="length" fix below).
            payload["return_token_ids"] = True
        harvest_this = bool(_HARVEST_PATH) and (
            _HARVEST_TAGS is None or (tag or _infer_call_tag(messages)) in _HARVEST_TAGS
        )
        if harvest_this:
            # Harvest wants the generated token ids (even for calls not stored for
            # reuse) so the offline harness can verify byte-exactness.
            payload["return_token_ids"] = True

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
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response: {data}") from exc

        finish_reason = choice.get("finish_reason")
        token_ids = choice.get("token_ids")

        # Truncated-summary KV-reuse fix: only for stored-for-reuse requests that
        # ended by hitting max_tokens (finish_reason="length"). The server stored
        # output[:-1] (M-1 tokens); the API content has M tokens -> the downstream
        # content-hash would miss. Re-embed decode(token_ids[:-1]) to match the
        # stored M-1 segment. Natural-EOS ("stop") content already == stored, so
        # it is left untouched (dropping there would over-trim -> a different miss).
        if store_generated_kv and finish_reason == "length":
            tokenizer = _get_reuse_tokenizer()
            if tokenizer is not None and token_ids and len(token_ids) > 1:
                content = tokenizer.decode(token_ids[:-1])

        usage = data.get("usage") or {}
        # Server-side per-request phase timing (present when vLLM is launched
        # with VLLM_RESP_TIMING=1); merged inline so each call log row carries
        # both the role tag and the prefill/decode split (no offline join).
        timing = data.get("timing") or {}
        _log_call(
            {
                # Explicit tag from the caller wins; otherwise fall back to
                # inferring from the system prompt's first line.
                "tag": tag or _infer_call_tag(messages),
                "req_id": data.get("id"),
                "start_ts": round(start_wall, 3),
                "end_ts": round(start_wall + elapsed, 3),
                "elapsed_s": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens": max_tokens,
                "model": model,
                "prefill_s": timing.get("prefill_s"),
                "decode_s": timing.get("decode_s"),
                "ttft_s": timing.get("ttft_s"),
                "queued_s": timing.get("queued_s"),
                "inference_s": timing.get("inference_s"),
            }
        )
        # Full-payload harvest for Exp A drift measurement (only the whitelisted
        # call types; see _HARVEST_TAGS). `content` here is post-truncation-fix
        # (== the byte-exact reusable segment text), and `token_ids` is the full
        # generated id list (stored segment = [:-1]).
        if harvest_this:
            _harvest_call(
                {
                    "tag": tag or _infer_call_tag(messages),
                    "req_id": data.get("id"),
                    "model": model,
                    "store_generated_kv": store_generated_kv,
                    "finish_reason": finish_reason,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "content": content,
                    "token_ids": token_ids,
                }
            )
        return content or ""
