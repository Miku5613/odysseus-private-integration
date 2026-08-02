from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _event_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload")
    if isinstance(payload, dict):
        return payload
    return raw


def _properties(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("properties")
    if isinstance(value, dict):
        return value
    value = event.get("data")
    if isinstance(value, dict):
        return value
    return {}


def event_session_id(raw: dict[str, Any]) -> str:
    event = _event_payload(raw)
    props = _properties(event)
    for source in (
        props,
        props.get("part"),
        props.get("info"),
        event.get("data"),
    ):
        if not isinstance(source, dict):
            continue
        value = source.get("sessionID") or source.get("session_id")
        if isinstance(value, str):
            return value
    durable = event.get("durable")
    if isinstance(durable, dict):
        aggregate = durable.get("aggregateID")
        if isinstance(aggregate, str) and aggregate.startswith("ses_"):
            return aggregate
    return ""


def _message_info(message: dict[str, Any]) -> dict[str, Any]:
    info = message.get("info")
    return info if isinstance(info, dict) else message


def _message_id(message: dict[str, Any]) -> str:
    return str(_message_info(message).get("id") or "")


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    info = _message_info(message)
    parts = message.get("parts")
    if not isinstance(parts, list):
        parts = info.get("parts")
    if not isinstance(parts, list):
        return set()
    return {
        str(part.get("callID") or part.get("id") or "")
        for part in parts
        if isinstance(part, dict) and part.get("type") == "tool"
    } - {""}


def snapshot_boundaries(
    messages: Iterable[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    message_ids: set[str] = set()
    call_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_id = _message_id(message)
        if message_id:
            message_ids.add(message_id)
        call_ids.update(_tool_call_ids(message))
    return message_ids, call_ids


def _safe_tool_summary(tool: str, state: dict[str, Any]) -> str:
    source = state.get("input")
    if not isinstance(source, dict):
        return tool
    for key in (
        "filePath",
        "path",
        "pattern",
        "query",
        "command",
        "description",
    ):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:300]
    return tool


def _safe_output_summary(state: dict[str, Any]) -> str:
    value = state.get("output")
    if value is None:
        value = state.get("error")
    if isinstance(value, (dict, list)):
        return str(value)[:1200]
    return str(value or "")[:1200]


@dataclass(slots=True)
class OpenCodeEventNormalizer:
    """Convert OpenCode global events into Odysseus runtime events.

    Reasoning text is deliberately never emitted. Only status, tool metadata,
    final-answer deltas, and terminal lifecycle events cross the boundary.
    """

    session_id: str
    part_types: dict[str, str] = field(default_factory=dict)
    tool_statuses: dict[str, str] = field(default_factory=dict)
    tool_started_calls: set[str] = field(default_factory=set)
    tool_completed_calls: set[str] = field(default_factory=set)
    baseline_message_ids: set[str] = field(default_factory=set)
    baseline_call_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_snapshot(
        cls,
        session_id: str,
        messages: Iterable[dict[str, Any]],
    ) -> "OpenCodeEventNormalizer":
        message_ids, call_ids = snapshot_boundaries(messages)
        return cls(
            session_id=session_id,
            baseline_message_ids=message_ids,
            baseline_call_ids=call_ids,
        )

    def normalize(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        event = _event_payload(raw)
        if event_session_id(raw) != self.session_id:
            return []

        event_type = str(event.get("type") or "")
        props = _properties(event)
        output: list[dict[str, Any]] = []

        if event_type == "todo.updated":
            todos = props.get("todos")
            if not isinstance(todos, list) or not todos:
                return []

            status_map = {
                "pending": "pending",
                "in_progress": "running",
                "in-progress": "running",
                "active": "running",
                "running": "running",
                "completed": "completed",
                "done": "completed",
                "success": "completed",
                "cancelled": "cancelled",
                "canceled": "cancelled",
                "failed": "failed",
                "error": "failed",
            }
            steps: list[dict[str, Any]] = []

            for index, todo in enumerate(todos):
                if not isinstance(todo, dict):
                    continue

                todo_id = str(
                    todo.get("id") or "opencode-todo-%s" % (index + 1)
                ).strip()
                objective = str(
                    todo.get("content")
                    or todo.get("title")
                    or todo_id
                ).strip()
                status = status_map.get(
                    str(todo.get("status") or "pending").lower(),
                    "pending",
                )

                if not todo_id or not objective:
                    continue

                steps.append({
                    "id": todo_id,
                    "objective": objective[:500],
                    "depends_on": [],
                    "work_domains": ["code"],
                    "status": status,
                })

            if not steps:
                return []

            return [{
                "type": "runtime.plan.updated",
                "runtime": "opencode",
                "status": "running",
                "label": "OpenCode checklist",
                "detail": "Checklist updated",
                "steps": steps,
            }]

        if event_type == "message.part.updated":
            part = props.get("part")
            if not isinstance(part, dict):
                return []
            part_id = str(part.get("id") or "")
            part_type = str(part.get("type") or "")
            message_id = str(
                part.get("messageID")
                or part.get("message_id")
                or props.get("messageID")
                or props.get("message_id")
                or ""
            )
            if message_id and message_id in self.baseline_message_ids:
                return []
            if part_id and part_type:
                self.part_types[part_id] = part_type

            if part_type == "tool":
                output.extend(self._normalize_tool_part(part))
            elif part_type == "step-start":
                output.append({
                    "type": "runtime.status",
                    "runtime": "opencode",
                    "status": "running",
                    "label": "Working",
                })
            elif part_type == "step-finish":
                reason = str(part.get("reason") or "")
                label = (
                    "Continuing after tools"
                    if reason in {"tool-calls", "tool_calls"}
                    else "Finalizing response"
                )
                output.append({
                    "type": "runtime.status",
                    "runtime": "opencode",
                    "status": "running",
                    "label": label,
                })
            return output

        if event_type == "message.part.delta":
            part_id = str(props.get("partID") or "")
            message_id = str(
                props.get("messageID") or props.get("message_id") or ""
            )
            if message_id and message_id in self.baseline_message_ids:
                return []
            field_name = str(props.get("field") or "")
            delta = props.get("delta")
            if (
                self.part_types.get(part_id) == "text"
                and field_name == "text"
                and isinstance(delta, str)
                and delta
            ):
                output.append({
                    "type": "assistant.delta",
                    "runtime": "opencode",
                    "delta": delta,
                })
            return output

        if event_type == "session.status":
            status = props.get("status")
            if isinstance(status, dict):
                status = status.get("type")
            output.append({
                "type": "runtime.status",
                "runtime": "opencode",
                "status": str(status or "running"),
                "label": "OpenCode runtime",
            })
            return output

        if event_type == "session.idle":
            output.append({
                "type": "runtime.idle",
                "runtime": "opencode",
                "status": "idle",
            })
            return output

        return []

    def reconcile_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reconcile structured tool parts from session message snapshots.

        OpenCode's global event stream is authoritative for status and text
        deltas, but installed versions may omit tool-part lifecycle events.
        Session messages remain the durable source of truth for tool calls.
        """

        output: list[dict[str, Any]] = []
        for message in messages:
            if _message_id(message) in self.baseline_message_ids:
                continue
            info = message.get("info")
            if not isinstance(info, dict):
                info = message
            if info.get("role") != "assistant":
                continue

            parts = message.get("parts")
            if not isinstance(parts, list):
                parts = info.get("parts")
            if not isinstance(parts, list):
                continue

            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_id = str(part.get("id") or "")
                part_type = str(part.get("type") or "")
                if part_id and part_type:
                    self.part_types[part_id] = part_type
                if part_type == "tool":
                    output.extend(self._normalize_tool_part(part))

        return output

    def _normalize_tool_part(
        self,
        part: dict[str, Any],
    ) -> list[dict[str, Any]]:
        call_id = str(part.get("callID") or part.get("id") or "")
        tool = str(part.get("tool") or "tool")
        state = part.get("state")
        if not isinstance(state, dict):
            state = {}
        status = str(state.get("status") or "pending").lower()

        if call_id in self.baseline_call_ids:
            return []
        if call_id in self.tool_completed_calls:
            return []

        base = {
            "runtime": "opencode",
            "call_id": call_id,
            "tool": tool,
            "summary": _safe_tool_summary(tool, state),
        }

        if status == "pending":
            self.tool_statuses.setdefault(call_id, status)
            return []

        if status == "running":
            self.tool_statuses[call_id] = status
            if call_id in self.tool_started_calls:
                return []
            self.tool_started_calls.add(call_id)
            return [{
                "type": "tool.started",
                **base,
                "status": "running",
            }]

        self.tool_statuses[call_id] = status
        events: list[dict[str, Any]] = []
        if call_id not in self.tool_started_calls:
            self.tool_started_calls.add(call_id)
            events.append({
                "type": "tool.started",
                **base,
                "status": "running",
            })

        self.tool_completed_calls.add(call_id)
        events.append({
            "type": "tool.completed",
            **base,
            "status": status,
            "success": status in {"completed", "success"},
            "output_summary": _safe_output_summary(state),
        })
        return events


def extract_final_assistant_message(
    messages: list[dict[str, Any]],
    *,
    excluded_message_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Return the latest completed assistant text and model metadata."""

    excluded = excluded_message_ids or set()
    for message in reversed(messages):
        if _message_id(message) in excluded:
            continue
        info = message.get("info")
        if not isinstance(info, dict):
            info = message
        if info.get("role") != "assistant":
            continue

        finish = str(info.get("finish") or "")
        if finish in {"tool-calls", "tool_calls", ""}:
            continue

        parts = message.get("parts")
        if not isinstance(parts, list):
            parts = info.get("parts")
        texts: list[str] = []
        if isinstance(parts, list):
            for part in parts:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    and part["text"].strip()
                ):
                    texts.append(part["text"].strip())

        if not texts:
            continue

        provider_id = str(
            info.get("providerID")
            or (
                info.get("model", {}).get("providerID")
                if isinstance(info.get("model"), dict)
                else ""
            )
            or ""
        )
        model_id = str(
            info.get("modelID")
            or (
                info.get("model", {}).get("modelID")
                if isinstance(info.get("model"), dict)
                else ""
            )
            or ""
        )
        return {
            "text": "\n\n".join(texts),
            "provider_id": provider_id,
            "model_id": model_id,
            "finish": finish,
            "message_id": str(info.get("id") or ""),
        }

    return {
        "text": "",
        "provider_id": "",
        "model_id": "",
        "finish": "",
        "message_id": "",
    }
