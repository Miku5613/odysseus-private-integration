from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult
from src.tools.research import do_manage_research, do_trigger_research


RESEARCH_WORKER_CAPABILITY = "deep_research"

ResearchBackend = Callable[..., Awaitable[dict[str, Any]]]


def _normalize_action(handoff: WorkerHandoff) -> str:
    action = str(handoff.context.get("action") or "start").strip().lower()

    aliases = {
        "trigger": "start",
        "research": "start",
        "open": "read",
        "view": "read",
        "get": "read",
        "retrieve": "read",
        "status": "list",
        "find": "list",
    }

    return aliases.get(action, action)


def _backend_failed(payload: dict[str, Any]) -> bool:
    if payload.get("error"):
        return True

    exit_code = payload.get("exit_code")
    return isinstance(exit_code, int) and exit_code != 0


async def run_research_worker(
    handoff: WorkerHandoff,
    *,
    session_id: str | None = None,
    trigger_backend: ResearchBackend = do_trigger_research,
    manage_backend: ResearchBackend = do_manage_research,
) -> WorkerResult:
    """Execute existing deep-research tools behind an async worker boundary."""

    if handoff.capability != RESEARCH_WORKER_CAPABILITY:
        raise ValueError(
            "Research Worker requires capability=deep_research"
        )

    objective = str(handoff.objective or "").strip()

    if not objective:
        raise ValueError("Research Worker requires a non-empty objective")

    action = _normalize_action(handoff)
    owner = handoff.context.get("owner")

    try:
        if action == "start":
            topic = str(
                handoff.context.get("topic")
                or handoff.context.get("research_question")
                or objective
            ).strip()

            if not topic:
                return WorkerResult(
                    capability=RESEARCH_WORKER_CAPABILITY,
                    status="needs_clarification",
                    summary="A research topic is required.",
                    missing_information=("research topic",),
                )

            request = {"topic": topic}
            payload = await trigger_backend(
                json.dumps(request),
                owner=owner,
            )

        elif action in {"list", "read", "delete"}:
            request: dict[str, Any] = {"action": action}

            research_id = str(
                handoff.context.get("research_id")
                or handoff.context.get("id")
                or ""
            ).strip()

            search = str(handoff.context.get("search") or "").strip()

            if action in {"read", "delete"} and not research_id:
                return WorkerResult(
                    capability=RESEARCH_WORKER_CAPABILITY,
                    status="needs_clarification",
                    summary=f"Research ID is required for {action}.",
                    missing_information=("research_id",),
                )

            if research_id:
                request["id"] = research_id

            if search:
                request["search"] = search

            payload = await manage_backend(
                json.dumps(request),
                owner=owner,
            )

        else:
            return WorkerResult(
                capability=RESEARCH_WORKER_CAPABILITY,
                status="needs_clarification",
                summary=f"Unsupported research action: {action}",
                missing_information=("research action",),
            )

    except Exception as exc:
        return WorkerResult(
            capability=RESEARCH_WORKER_CAPABILITY,
            status="failed",
            summary="Research runtime failed.",
            data={
                "action": action,
                "session_id": session_id,
            },
            error=str(exc),
        )

    if not isinstance(payload, dict):
        return WorkerResult(
            capability=RESEARCH_WORKER_CAPABILITY,
            status="failed",
            summary="Research runtime returned an invalid payload.",
            data={"action": action},
            error="Expected a dictionary result.",
        )

    if _backend_failed(payload):
        return WorkerResult(
            capability=RESEARCH_WORKER_CAPABILITY,
            status="failed",
            summary="Research operation failed.",
            data={
                "action": action,
                "payload": payload,
                "session_id": session_id,
            },
            error=str(payload.get("error") or payload.get("output") or "Unknown research failure."),
        )

    summary = str(
        payload.get("output")
        or payload.get("message")
        or f"Research action {action} completed."
    )

    return WorkerResult(
        capability=RESEARCH_WORKER_CAPABILITY,
        status="completed",
        summary=summary,
        data={
            "action": action,
            "payload": payload,
            "session_id": session_id,
        },
    )


__all__ = [
    "RESEARCH_WORKER_CAPABILITY",
    "run_research_worker",
]
