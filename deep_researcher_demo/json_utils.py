"""JSON parsing helpers for prompt-based tool protocols."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class JSONParseError(ValueError):
    """Raised when an LLM response cannot be parsed as a JSON object."""


def strip_json_fence(text: str) -> str:
    """Remove a Markdown JSON fence if the model wrapped its response."""
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return stripped


def extract_json_object(text: str) -> str:
    """Extract the first JSON object-looking span from text."""
    stripped = strip_json_fence(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JSONParseError("Response does not contain a JSON object.")
    return stripped[start : end + 1]


def parse_json_object(text: str) -> dict:
    """Parse a response into a JSON object."""
    try:
        value = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise JSONParseError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JSONParseError("Expected a JSON object.")
    return value


def parse_model_json(text: str, model: type[ModelT]) -> ModelT:
    """Parse and validate an LLM JSON response with a Pydantic model."""
    return model.model_validate(parse_json_object(text))

