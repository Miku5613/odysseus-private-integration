from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from src.agent_runtime.streaming import run_stream
from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult


CODE_WORKER_CAPABILITY = "code_inspection"


def _normalize_workspace(handoff: WorkerHandoff) -> str | None:
    workspace = handoff.context.get("workspace")

    if workspace is None:
        return None

    workspace_text = str(workspace).strip()
    return workspace_text or None


def _collect_worker_result(
    handoff: WorkerHandoff,
    events: Iterable[dict[str, Any]],
) -> WorkerResult:
    final_answer = ""
    evidence: list[dict[str, Any]] = []
    error: str | None = None
    event_count = 0

    for event in events:
        event_count += 1
        event_type = str(event.get("type") or "")

        if event_type == "evidence_hit":
            evidence.append(dict(event))

        if event_type == "final_answer":
            content = event.get("content")
            if isinstance(content, str):
                final_answer = content

        if event_type == "stream_done":
            content = event.get("final_answer")
            if isinstance(content, str) and content.strip():
                final_answer = content

        if event_type == "stream_error":
            message = event.get("error")
            error = str(message or "Code Worker runtime failed.")

    if error is not None:
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="failed",
            summary="Code inspection worker failed.",
            data={
                "objective": handoff.objective,
                "events_processed": event_count,
            },
            evidence=tuple(evidence),
            error=error,
        )

    if not final_answer.strip():
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="partial",
            summary="Code inspection completed without a final answer.",
            data={
                "objective": handoff.objective,
                "events_processed": event_count,
            },
            evidence=tuple(evidence),
        )

    return WorkerResult(
        capability=CODE_WORKER_CAPABILITY,
        status="completed",
        summary=final_answer,
        data={
            "objective": handoff.objective,
            "events_processed": event_count,
            "supports_active_doc": True,
        },
        evidence=tuple(evidence),
    )


def run_code_worker(
    handoff: WorkerHandoff,
    *,
    session_id: str | None = None,
    stream_factory: Callable[..., Iterable[dict[str, Any]]] = run_stream,
) -> WorkerResult:
    """Execute the existing code-inspection runtime behind a worker boundary.

    This adapter intentionally does not alter the underlying evidence-heavy
    graph. It translates generic Supervisor state into the specialist runtime
    and normalizes specialist events back into a WorkerResult.
    """

    if handoff.capability != CODE_WORKER_CAPABILITY:
        raise ValueError(
            "Code Worker requires capability=code_inspection"
        )

    objective = str(handoff.objective or "").strip()

    if not objective:
        raise ValueError("Code Worker requires a non-empty objective")

    workspace = _normalize_workspace(handoff)

    events = stream_factory(
        objective,
        workspace=workspace,
        session_id=session_id,
    )

    return _collect_worker_result(handoff, events)


__all__ = [
    "CODE_WORKER_CAPABILITY",
    "run_code_worker",
]
