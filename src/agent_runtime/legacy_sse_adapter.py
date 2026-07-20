from __future__ import annotations

import json
from typing import Any, Iterable


def to_legacy_payload(event: dict[str, Any]) -> dict[str, Any] | str | None:
    """Convert LangGraph runtime events into the current Odysseus frontend event shape.

    This adapter is intentionally conservative:
    - final_answer becomes {"delta": "..."}
    - tool starts/results use existing-looking tool_start/tool_output events
    - completion becomes [DONE]
    - unknown internal events are omitted instead of exposing new UI contracts
    """

    event_type = event.get("type")

    if event_type == "tool_confirmation_required":
        tool_name = str(event.get("name") or "This action")

        return {
            "type": "ask_user",
            "data": {
                "interaction_mode": "tool_confirmation",
                "question": (
                    f"{tool_name} requires your confirmation. Continue?"
                ),
                "options": [
                    {
                        "label": "Yes",
                        "description": "Approve this exact action once.",
                    },
                    {
                        "label": "No",
                        "description": "Cancel this action.",
                    },
                ],
                "multi": False,
                "session_id": event.get("session_id"),
                "call_id": event.get("call_id"),
                "fingerprint": event.get("fingerprint"),
                "tool": event.get("name"),
                "reason": event.get("reason"),
                "confirmation_available": bool(
                    event.get("confirmation_available")
                ),
            },
        }


    if event_type == "stream_start":
        return {
            "type": "agent_prep",
            "data": {
                "runtime": event.get("runtime"),
                "scope": event.get("scope"),
                "workspace": event.get("workspace"),
            },
        }

    if event_type == "agent_model_call_start":
        return {
            "type": "agent_step",
            "data": {
                "label": "Thinking",
                "runtime": event.get("runtime"),
                "tool_rounds": event.get("tool_rounds", 0),
                "scope": event.get("scope"),
            },
        }

    if event_type == "tool_call_start":
        return {
            "type": "tool_start",
            "tool": event.get("name"),
            "command": event.get("args", {}),
            "call_id": event.get("call_id"),
        }

    if event_type == "tool_call_result":
        return {
            "type": "tool_output",
            "tool": event.get("name"),
            "output": event.get("preview") or "",
            "ok": event.get("ok"),
            "call_id": event.get("call_id"),
            "chars": event.get("chars"),
        }

    if event_type == "evidence_hit":
        return {
            "type": "tool_output",
            "tool": "evidence",
            "output": event.get("summary") or "",
            "ok": True,
            "metadata": {
                "kind": event.get("kind"),
                "strength": event.get("strength"),
                "file": event.get("file"),
                "line": event.get("line"),
                "route": event.get("route"),
                "symbol": event.get("symbol"),
            },
        }

    if event_type == "runtime_stop_condition":
        return {
            "type": "agent_step",
            "data": {
                "label": "Stop condition",
                "reason": event.get("reason"),
                "tool_rounds": event.get("tool_rounds"),
                "runtime": event.get("runtime"),
            },
        }

    if event_type == "finalizer_start":
        return {
            "type": "agent_step",
            "data": {
                "label": "Finalizing",
                "strong_evidence": event.get("strong_evidence"),
                "tool_rounds": event.get("tool_rounds"),
                "runtime": event.get("runtime"),
            },
        }

    if event_type == "final_answer":
        return {
            "delta": event.get("content") or event.get("preview") or "",
        }

    if event_type == "stream_done":
        return "[DONE]"

    if event_type == "stream_error":
        return {
            "type": "error",
            "error": event.get("error"),
            "error_type": event.get("error_type"),
            "status": 500,
        }

    # Do not surface internal-only events to the legacy frontend by default.
    if event_type in {
        "agent_tool_calls_requested",
        "agent_done",
        "agent_model_final_candidate",
    }:
        return None

    return None


def to_legacy_sse_frame(event: dict[str, Any]) -> str | None:
    payload = to_legacy_payload(event)

    if payload is None:
        return None

    if payload == "[DONE]":
        return "data: [DONE]\n\n"

    if isinstance(payload, dict) and payload.get("type") == "error":
        return "event: error\ndata: " + json.dumps(payload, ensure_ascii=False, default=str) + "\n\n"

    return "data: " + json.dumps(payload, ensure_ascii=False, default=str) + "\n\n"


def adapt_events_to_legacy_payloads(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any] | str]:
    for event in events:
        payload = to_legacy_payload(event)
        if payload is not None:
            yield payload


def adapt_events_to_legacy_frames(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        frame = to_legacy_sse_frame(event)
        if frame is not None:
            yield frame
