"""Progress reporting primitives for realtime CLI feedback."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ProgressEvent:
    """A single workflow progress event."""

    step: str
    message: str
    data: dict[str, Any] | None = None


class ProgressReporter(Protocol):
    """Minimal progress reporter interface."""

    def emit(self, event: ProgressEvent) -> None:
        """Emit one progress event."""


class ConsoleProgressReporter:
    """Print progress events immediately to stdout."""

    def emit(self, event: ProgressEvent) -> None:
        print(format_event(event), flush=True)


class NullProgressReporter:
    """Progress reporter that intentionally does nothing."""

    def emit(self, event: ProgressEvent) -> None:
        return None


class CompositeProgressReporter:
    """Progress reporter that forwards events to multiple reporters."""

    def __init__(self, *reporters: ProgressReporter) -> None:
        self.reporters = reporters

    def emit(self, event: ProgressEvent) -> None:
        for reporter in self.reporters:
            reporter.emit(event)


class MemoryProgressReporter:
    """Progress reporter useful for tests."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def event_to_dict(event: ProgressEvent) -> dict[str, Any]:
    """Serialize one progress event without dropping event data."""
    return {
        "step": event.step,
        "message": event.message,
        "data": dict(event.data or {}),
    }


def format_event(event: ProgressEvent) -> str:
    """Format one event as a concise terminal line."""
    return f"[{event.step}] {truncate(event.message)}"


def truncate(value: str, max_chars: int = 180) -> str:
    """Trim long progress lines while preserving useful signal."""
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def format_list(values: list[str], max_items: int = 5) -> str:
    """Format a short list for progress output."""
    if not values:
        return "none"
    visible = values[:max_items]
    suffix = "" if len(values) <= max_items else f", ... +{len(values) - max_items}"
    return "; ".join(truncate(value, 80) for value in visible) + suffix
