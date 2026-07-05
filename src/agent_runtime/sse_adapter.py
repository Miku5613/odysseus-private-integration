from __future__ import annotations

import json
from typing import Any, Iterable


def to_sse_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Convert internal LangGraph runtime events into Odysseus-style stream payloads.

    This is intentionally UI-agnostic. routes/chat_routes.py can later wrap these
    payloads as SSE frames without knowing LangGraph internals.
    """

    event_type = event.get("type")

    if event_type == "stream_start":
        return {
            "type": "agent_start",
            "runtime": event.get("runtime"),
            "scope": event.get("scope"),
            "workspace": event.get("workspace"),
        }

    if event_type == "agent_model_call_start":
        return {
            "type": "agent_thinking",
            "runtime": event.get("runtime"),
            "tool_rounds": event.get("tool_rounds", 0),
            "scope": event.get("scope"),
        }

    if event_type == "agent_tool_calls_requested":
        return {
            "type": "tool_calls_requested",
            "runtime": event.get("runtime"),
            "tools": event.get("tools", []),
            "count": event.get("count", 0),
        }

    if event_type == "tool_call_start":
        return {
            "type": "tool_start",
            "runtime": event.get("runtime"),
            "tool_name": event.get("name"),
            "tool_args": event.get("args", {}),
            "call_id": event.get("call_id"),
        }

    if event_type == "tool_call_result":
        return {
            "type": "tool_result",
            "runtime": event.get("runtime"),
            "tool_name": event.get("name"),
            "ok": event.get("ok"),
            "call_id": event.get("call_id"),
            "preview": event.get("preview"),
            "chars": event.get("chars"),
        }

    if event_type == "evidence_hit":
        return {
            "type": "evidence",
            "runtime": event.get("runtime"),
            "kind": event.get("kind"),
            "strength": event.get("strength"),
            "summary": event.get("summary"),
            "file": event.get("file"),
            "line": event.get("line"),
            "route": event.get("route"),
            "symbol": event.get("symbol"),
        }

    if event_type == "runtime_stop_condition":
        return {
            "type": "agent_stop_condition",
            "runtime": event.get("runtime"),
            "reason": event.get("reason"),
            "tool_rounds": event.get("tool_rounds"),
        }

    if event_type == "finalizer_start":
        return {
            "type": "finalizer_start",
            "runtime": event.get("runtime"),
            "tool_rounds": event.get("tool_rounds"),
            "strong_evidence": event.get("strong_evidence"),
        }

    if event_type == "final_answer":
        return {
            "type": "message",
            "runtime": event.get("runtime"),
            "role": "assistant",
            "content": event.get("content") or event.get("preview") or "",
            "chars": event.get("chars"),
        }

    if event_type == "agent_done":
        return {
            "type": "agent_done",
            "runtime": event.get("runtime"),
            "reason": event.get("reason"),
        }

    if event_type == "stream_done":
        return {
            "type": "done",
            "runtime": event.get("runtime"),
            "reason": event.get("reason"),
            "final_answer": event.get("final_answer"),
        }

    if event_type == "stream_error":
        return {
            "type": "error",
            "runtime": event.get("runtime"),
            "error_type": event.get("error_type"),
            "error": event.get("error"),
        }

    return {
        "type": "debug_event",
        "runtime": event.get("runtime"),
        "source_type": event_type,
        "event": event,
    }


def to_sse_frame(payload: dict[str, Any]) -> str:
    """Encode payload as a simple SSE data frame."""
    return "data: " + json.dumps(payload, ensure_ascii=False, default=str) + "\n\n"


def adapt_events(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for event in events:
        yield to_sse_payload(event)


def adapt_events_to_frames(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        yield to_sse_frame(to_sse_payload(event))
