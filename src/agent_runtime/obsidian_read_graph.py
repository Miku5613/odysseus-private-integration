
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .obsidian_read_adapter import ObsidianReadAdapter, ObsidianReadError


TaskKind = Literal["status", "active_note", "list_dir", "read_note", "unsupported"]


@dataclass
class ObsidianReadObservation:
    tool: str
    args: dict[str, Any]
    success: bool
    output_preview: str


@dataclass
class ObsidianReadState:
    raw_query: str
    task_kind: TaskKind = "unsupported"
    target_path: str = ""
    final_answer: str = ""
    observations: list[ObsidianReadObservation] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _preview(text: Any, limit: int = 800) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n")
    return s[:limit]


def _record(state: ObsidianReadState, tool: str, args: dict[str, Any], success: bool, output: Any) -> None:
    state.observations.append(
        ObsidianReadObservation(
            tool=tool,
            args=args,
            success=bool(success),
            output_preview=_preview(output),
        )
    )


def _extract_quoted_path(query: str) -> str:
    q = str(query or "").strip()
    for quote in ['"', "'"]:
        if quote in q:
            parts = q.split(quote)
            if len(parts) >= 3 and parts[1].strip():
                return parts[1].strip()
    return ""


def plan_read_task(query: str) -> tuple[TaskKind, str]:
    low = str(query or "").lower()

    # Hard safety block: this graph must never interpret mutation as read work.
    mutation_words = [
        "write", "append", "edit", "update", "create", "delete", "remove",
        "rename", "move", "purge", "save", "replace", "modify",
    ]
    if any(w in low for w in mutation_words):
        return "unsupported", ""

    if "active" in low and ("note" in low or "file" in low):
        return "active_note", ""

    quoted = _extract_quoted_path(query)
    if quoted:
        if quoted.endswith("/") or "list" in low or "folder" in low or "directory" in low:
            return "list_dir", quoted
        return "read_note", quoted

    if "status" in low or "health" in low:
        return "status", ""

    if "list" in low or "folder" in low or "directory" in low or "vault" in low:
        # Default to vault root unless the user quoted a path.
        return "list_dir", ""

    return "unsupported", ""


def _failure_answer(reason: str) -> str:
    return "\n".join([
        "I could not handle this with the Obsidian Read Graph.",
        "",
        f"Reason: {reason}",
        "",
        "This graph is GET-only and can currently:",
        "- check Obsidian REST status",
        "- list vault folders",
        "- read a quoted note path",
        "- read the active note",
        "",
        "No write, edit, create, delete, rename, move, save, command, or filesystem operation was attempted.",
    ])


def run_obsidian_read_graph(query: str, *, adapter: ObsidianReadAdapter | None = None) -> ObsidianReadState:
    start = time.time()
    state = ObsidianReadState(raw_query=query)
    adapter = adapter or ObsidianReadAdapter()

    kind, target = plan_read_task(query)
    state.task_kind = kind
    state.target_path = target

    try:
        if kind == "status":
            data = adapter.status()
            _record(state, "obsidian_status", {}, True, data)
            state.final_answer = "Obsidian REST status: OK\n\n" + _preview(data, 1200)

        elif kind == "active_note":
            text = adapter.active_note()
            _record(state, "obsidian_active_note", {}, bool(text.strip()), text)
            state.final_answer = "Active Obsidian note content:\n\n" + text

        elif kind == "list_dir":
            files = adapter.list_dir(target)
            _record(state, "obsidian_list_dir", {"path": target}, True, "\n".join(files))
            label = target or "/"
            state.final_answer = f"Obsidian vault listing for `{label}`:\n\n" + "\n".join(f"- {x}" for x in files)

        elif kind == "read_note":
            text = adapter.read_note(target)
            _record(state, "obsidian_read_note", {"path": target}, bool(text.strip()), text)
            state.final_answer = f"Obsidian note `{target}`:\n\n{text}"

        else:
            state.final_answer = _failure_answer("No safe read-only Obsidian task recognized.")

    except ObsidianReadError as exc:
        state.final_answer = _failure_answer(str(exc))
        _record(state, "obsidian_read_error", {"task": kind, "path": target}, False, str(exc))

    state.metrics = {
        "runtime": "v0.16-obsidian-read-graph",
        "model": "deterministic-obsidian-read",
        "elapsed_s": round(time.time() - start, 3),
        "tool_calls": len(state.observations),
        "task_kind": state.task_kind,
        "target_path": state.target_path,
        "read_only": True,
        "write_capable": False,
    }
    return state
