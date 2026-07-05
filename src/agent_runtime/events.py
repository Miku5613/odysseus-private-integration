from __future__ import annotations

import contextvars
import json
import time
from typing import Any, Callable


EventSink = Callable[[dict[str, Any]], None]

_EVENT_SINK: contextvars.ContextVar[EventSink | None] = contextvars.ContextVar(
    "agent_runtime_event_sink",
    default=None,
)

_SUPPRESS_EVENT_PRINT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_runtime_suppress_event_print",
    default=False,
)

_SUPPRESS_DEBUG_PRINT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_runtime_suppress_debug_print",
    default=False,
)


def make_event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "ts": time.time(),
        **payload,
    }


def set_event_sink(
    sink: EventSink | None,
    *,
    suppress_print: bool = False,
    suppress_debug: bool = False,
):
    sink_token = _EVENT_SINK.set(sink)
    print_token = _SUPPRESS_EVENT_PRINT.set(suppress_print)
    debug_token = _SUPPRESS_DEBUG_PRINT.set(suppress_debug)
    return sink_token, print_token, debug_token


def reset_event_sink(tokens) -> None:
    sink_token, print_token, debug_token = tokens
    _EVENT_SINK.reset(sink_token)
    _SUPPRESS_EVENT_PRINT.reset(print_token)
    _SUPPRESS_DEBUG_PRINT.reset(debug_token)


def print_event(event: dict[str, Any]) -> None:
    """Emit event to optional runtime sink and optionally print as JSONL."""
    sink = _EVENT_SINK.get()
    if sink is not None:
        sink(event)

    if not _SUPPRESS_EVENT_PRINT.get():
        print("[event] " + json.dumps(event, ensure_ascii=False, default=str))


def debug_print(message: str) -> None:
    """Print human debug logs only when not suppressed."""
    if not _SUPPRESS_DEBUG_PRINT.get():
        print(message)
