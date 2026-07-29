from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult
from src.tools.research import (
    do_cancel_research_job,
    do_manage_research,
    do_research_result_peek,
    do_research_status,
    do_trigger_research,
)


RESEARCH_WORKER_CAPABILITY = "deep_research"

ResearchBackend = Callable[..., Awaitable[dict[str, Any]]]
ResearchJobBackend = Callable[..., Awaitable[dict[str, Any]]]
ResearchSemanticValidator = Callable[[str, str], tuple[bool, str]]


def _default_semantic_validator(
    objective: str,
    report: str,
) -> tuple[bool, str]:
    normalized = str(report or "").strip()
    if len(normalized) < 120:
        return False, "Completed research report is too short."
    objective_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}", objective)
        if term.lower() not in {
            "about", "after", "before", "could", "investigate",
            "research", "report", "should", "their", "there",
            "these", "those", "using", "would",
        }
    }
    report_lower = normalized.lower()
    if objective_terms and not any(
        term in report_lower
        for term in objective_terms
    ):
        return (
            False,
            "Completed report does not address the PlanStep objective.",
        )
    return True, "Report contains substantive objective-related content."


def _normalize_research_status(payload: dict[str, Any]) -> str:
    raw = str(payload.get("status") or "").strip().lower()
    aliases = {
        "pending": "queued",
        "starting": "queued",
        "done": "completed",
        "complete": "completed",
        "error": "failed",
        "stopped": "cancelled",
        "canceled": "cancelled",
    }
    return aliases.get(raw, raw)


def _notify_job_observer(
    observer: Any,
    research_job_id: str,
    objective: str,
    status: str,
) -> None:
    if not callable(observer):
        return
    try:
        observer(research_job_id, objective, status)
    except TypeError:
        observer(research_job_id, objective)


async def cancel_owned_research_jobs(
    checkpoint_state: dict[str, Any],
    *,
    owner: str | None,
    cancel_backend: ResearchJobBackend = do_cancel_research_job,
) -> tuple[str, ...]:
    cancelled: list[str] = []
    for research_job_id, job in (
        checkpoint_state.get("owned_research_jobs") or {}
    ).items():
        if (
            not isinstance(job, dict)
            or job.get("owned") is not True
            or job.get("status") not in {"queued", "running"}
        ):
            continue
        payload = await cancel_backend(research_job_id, owner=owner)
        if (
            isinstance(payload, dict)
            and not _backend_failed(payload)
            and payload.get("cancelled") is not False
        ):
            cancelled.append(research_job_id)
    return tuple(cancelled)


def _normalize_action(handoff: WorkerHandoff) -> str:
    action = str(handoff.context.get("action") or "start").strip().lower()

    aliases = {
        "trigger": "start",
        "trigger_research": "start",
        "research": "start",
        "open": "read",
        "view": "read",
        "get": "read",
        "retrieve": "read",
        "retrieve_research": "read",
        "reuse": "read",
        "reuse_research_context": "read",
        "manage_research": "list",
        "status": "list",
        "find": "list",
    }

    return aliases.get(action, action)


def _first_research_id(payload: dict[str, Any]) -> str:
    """Resolve the first ID from manage_research public list output."""

    output = str(
        payload.get("output")
        or payload.get("results")
        or payload.get("content")
        or ""
    )

    marker = "#research-"
    start = output.find(marker)

    if start < 0:
        return ""

    start += len(marker)
    end = output.find(")", start)

    if end < 0:
        return ""

    candidate = output[start:end].strip()

    if not candidate:
        return ""

    if not all(
        char.isalnum() or char in "_-"
        for char in candidate
    ):
        return ""

    return candidate


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
    status_backend: ResearchJobBackend = do_research_status,
    result_backend: ResearchJobBackend = do_research_result_peek,
    cancel_backend: ResearchJobBackend = do_cancel_research_job,
    semantic_validator: ResearchSemanticValidator = (
        _default_semantic_validator
    ),
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
    research_job_id = str(
        handoff.context.get("research_job_id") or ""
    ).strip()
    research_id = ""
    research_job_owned = bool(
        handoff.context.get("research_job_owned", False)
    )

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

            if research_job_id:
                if not research_job_owned:
                    return WorkerResult(
                        capability=RESEARCH_WORKER_CAPABILITY,
                        status="failed",
                        summary=(
                            "A pre-existing research report cannot be resumed "
                            "as an owned new-research job."
                        ),
                        data={
                            "action": action,
                            "research_job_id": research_job_id,
                            "semantic_complete": False,
                        },
                        error="Research ownership was not established by this run.",
                    )
                payload = {
                    "research_session_id": research_job_id,
                    "output": "Resuming owned research job.",
                    "exit_code": 0,
                }
            else:
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

            if action == "read" and not research_id:
                list_request: dict[str, Any] = {"action": "list"}

                if search:
                    list_request["search"] = search

                list_payload = await manage_backend(
                    json.dumps(list_request),
                    owner=owner,
                )

                if not isinstance(list_payload, dict):
                    return WorkerResult(
                        capability=RESEARCH_WORKER_CAPABILITY,
                        status="failed",
                        summary="Research library returned an invalid response.",
                    )

                if _backend_failed(list_payload):
                    payload = list_payload
                else:
                    research_id = _first_research_id(list_payload)

                    if not research_id:
                        return WorkerResult(
                            capability=RESEARCH_WORKER_CAPABILITY,
                            status="needs_clarification",
                            summary=(
                                "No completed research report could be resolved "
                                "from the existing research library."
                            ),
                            missing_information=("completed research report",),
                        )

                    payload = await manage_backend(
                        json.dumps(
                            {
                                "action": "read",
                                "id": research_id,
                            }
                        ),
                        owner=owner,
                    )

            else:
                if action == "delete" and not research_id:
                    return WorkerResult(
                        capability=RESEARCH_WORKER_CAPABILITY,
                        status="needs_clarification",
                        summary="Research ID is required for delete.",
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

    if action == "start":
        research_job_id = research_job_id or str(
            payload.get("research_session_id")
            or payload.get("session_id")
            or payload.get("task_id")
            or ""
        ).strip()
        if not research_job_id or research_job_id == "?":
            return WorkerResult(
                capability=RESEARCH_WORKER_CAPABILITY,
                status="failed",
                summary=(
                    "Research started without a stable job ID, so its "
                    "completion cannot be observed safely."
                ),
                data={
                    "action": action,
                    "payload": payload,
                    "session_id": session_id,
                    "semantic_complete": False,
                },
                error="Research trigger did not return a stable job ID.",
            )

        observer = handoff.context.get("research_job_observer")
        _notify_job_observer(
            observer,
            research_job_id,
            objective,
            "running",
        )

        cancellation_scope = handoff.context.get("cancellation_scope")
        if cancellation_scope is not None and not research_job_owned:
            cancellation_scope.own_cancellation(
                f"research:{research_job_id}",
                lambda: cancel_backend(research_job_id, owner=owner),
            )

        timeout_value = handoff.context.get("research_timeout_s")
        timeout_s = max(
            0.0,
            float(900 if timeout_value is None else timeout_value),
        )
        poll_value = handoff.context.get("research_poll_interval_s")
        poll_interval_s = max(
            0.0,
            float(1.5 if poll_value is None else poll_value),
        )
        started_at = asyncio.get_running_loop().time()
        observed_states: list[str] = []

        try:
            while True:
                status_payload = await status_backend(
                    research_job_id,
                    owner=owner,
                )
                if not isinstance(status_payload, dict):
                    raise ValueError(
                        "Research status backend returned an invalid payload."
                    )
                research_status = _normalize_research_status(status_payload)
                if (
                    _backend_failed(status_payload)
                    and research_status not in {"failed", "cancelled"}
                ):
                    return WorkerResult(
                        capability=RESEARCH_WORKER_CAPABILITY,
                        status="failed",
                        summary="Research status observation failed.",
                        data={
                            "action": action,
                            "research_job_id": research_job_id,
                            "observed_states": observed_states,
                            "semantic_complete": False,
                        },
                        error=str(
                            status_payload.get("error")
                            or "Research status unavailable."
                        ),
                    )

                if research_status and (
                    not observed_states
                    or observed_states[-1] != research_status
                ):
                    observed_states.append(research_status)
                    _notify_job_observer(
                        observer,
                        research_job_id,
                        objective,
                        research_status,
                    )

                if research_status in {"queued", "running"}:
                    if (
                        asyncio.get_running_loop().time() - started_at
                        >= timeout_s
                    ):
                        return WorkerResult(
                            capability=RESEARCH_WORKER_CAPABILITY,
                            status="partial",
                            summary=(
                                "Research is still running; its job ID was "
                                "preserved for durable continuation."
                            ),
                            data={
                                "action": action,
                                "research_job_id": research_job_id,
                                "research_status": research_status,
                                "observed_states": observed_states,
                                "research_job_owned": True,
                                "semantic_complete": False,
                            },
                        )
                    await asyncio.sleep(poll_interval_s)
                    continue

                if research_status == "cancelled":
                    return WorkerResult(
                        capability=RESEARCH_WORKER_CAPABILITY,
                        status="failed",
                        summary="Owned research job was cancelled.",
                        data={
                            "action": action,
                            "research_job_id": research_job_id,
                            "research_status": "cancelled",
                            "observed_states": observed_states,
                            "semantic_complete": False,
                        },
                        error="Research job cancelled.",
                    )

                if research_status == "failed":
                    return WorkerResult(
                        capability=RESEARCH_WORKER_CAPABILITY,
                        status="failed",
                        summary="Research job failed.",
                        data={
                            "action": action,
                            "research_job_id": research_job_id,
                            "research_status": "failed",
                            "observed_states": observed_states,
                            "semantic_complete": False,
                        },
                        error=str(
                            status_payload.get("error")
                            or "Research job failed."
                        ),
                    )

                if research_status == "completed":
                    break

                return WorkerResult(
                    capability=RESEARCH_WORKER_CAPABILITY,
                    status="failed",
                    summary="Research returned an unsupported state.",
                    data={
                        "action": action,
                        "research_job_id": research_job_id,
                        "research_status": research_status,
                        "observed_states": observed_states,
                        "semantic_complete": False,
                    },
                    error=f"Unsupported research status: {research_status!r}",
                )
        except asyncio.CancelledError:
            if cancellation_scope is not None:
                cancellation_scope.cancel()
            else:
                await cancel_backend(research_job_id, owner=owner)
            raise

        result_payload = await result_backend(
            research_job_id,
            owner=owner,
        )
        if (
            not isinstance(result_payload, dict)
            or _backend_failed(result_payload)
        ):
            return WorkerResult(
                capability=RESEARCH_WORKER_CAPABILITY,
                status="failed",
                summary="Completed research report is unavailable.",
                data={
                    "action": action,
                    "research_job_id": research_job_id,
                    "observed_states": observed_states,
                    "semantic_complete": False,
                },
                error=str(
                    (result_payload or {}).get("error")
                    or "Final research report unavailable."
                ),
            )

        report = str(result_payload.get("result") or "").strip()
        accepted, acceptance_reason = semantic_validator(
            objective,
            report,
        )
        if not accepted:
            return WorkerResult(
                capability=RESEARCH_WORKER_CAPABILITY,
                status="failed",
                summary="Completed research report was insufficient.",
                data={
                    "action": action,
                    "research_job_id": research_job_id,
                    "research_status": "completed",
                    "observed_states": observed_states,
                    "semantic_complete": False,
                    "acceptance_reason": acceptance_reason,
                },
                error=acceptance_reason,
            )

        sources = list(result_payload.get("sources") or [])
        raw_findings = list(result_payload.get("raw_findings") or [])
        artifacts = [{
            "type": "research_report",
            "research_id": research_job_id,
            "href": f"#research-{research_job_id}",
        }]
        return WorkerResult(
            capability=RESEARCH_WORKER_CAPABILITY,
            status="completed",
            summary=report,
            data={
                "action": action,
                "research_job_id": research_job_id,
                "research_status": "completed",
                "observed_states": observed_states,
                "research_job_owned": True,
                "semantic_complete": True,
                "acceptance_reason": acceptance_reason,
                "report": report,
                "sources": sources,
                "raw_findings": raw_findings,
                "artifacts": artifacts,
                "dependency_state": {
                    "report": report,
                    "sources": sources,
                    "artifacts": artifacts,
                },
            },
            evidence=tuple(
                source
                for source in sources
                if isinstance(source, dict)
            ),
        )

    if action == "read":
        report = str(
            payload.get("output")
            or payload.get("content")
            or payload.get("result")
            or ""
        ).strip()
        accepted, acceptance_reason = semantic_validator(
            objective,
            report,
        )
        if not accepted:
            return WorkerResult(
                capability=RESEARCH_WORKER_CAPABILITY,
                status="failed",
                summary="Existing research report was insufficient.",
                data={
                    "action": action,
                    "research_id": research_id,
                    "research_job_owned": False,
                    "semantic_complete": False,
                    "acceptance_reason": acceptance_reason,
                },
                error=acceptance_reason,
            )
        artifacts = (
            [{
                "type": "research_report",
                "research_id": research_id,
                "href": f"#research-{research_id}",
            }]
            if research_id
            else []
        )
        return WorkerResult(
            capability=RESEARCH_WORKER_CAPABILITY,
            status="completed",
            summary=report,
            data={
                "action": action,
                "payload": payload,
                "session_id": session_id,
                "research_id": research_id,
                "research_job_owned": False,
                "semantic_complete": True,
                "acceptance_reason": acceptance_reason,
                "report": report,
                "artifacts": artifacts,
                "dependency_state": {
                    "report": report,
                    "artifacts": artifacts,
                },
            },
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
            "semantic_complete": True,
        },
    )


__all__ = [
    "RESEARCH_WORKER_CAPABILITY",
    "cancel_owned_research_jobs",
    "run_research_worker",
]
