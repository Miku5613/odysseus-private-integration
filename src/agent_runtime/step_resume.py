from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from src.agent_runtime.run_lifecycle import LangGraphRunLifecycle
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.worker_dispatcher import (
    WorkerRunner,
    build_worker_handoff,
    builtin_worker_runners,
)


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _result_payload(result: WorkerResult) -> dict[str, Any]:
    return {
        "capability": result.capability,
        "status": result.status,
        "summary": result.summary,
        "data": result.data,
        "evidence": list(result.evidence),
        "missing_information": list(result.missing_information),
        "error": result.error,
    }


async def resume_interrupted_step(
    *,
    checkpoint_store: Any,
    checkpoint_state: dict[str, Any],
    owner: str,
    step_id: str | None = None,
    worker_runners: Mapping[str, WorkerRunner] | None = None,
) -> dict[str, Any]:
    """Restart one explicitly safe PlanStep boundary, never the whole query."""

    run_id = str(checkpoint_state.get("run_id") or "")
    if not run_id:
        return {"outcome": "NOT_RESUMABLE", "reason": "Missing run identity."}
    run = checkpoint_store.run_record(run_id)
    if run is None or str(run.get("owner") or "") != str(owner):
        raise PermissionError("runtime run ownership mismatch")
    if str(checkpoint_state.get("status") or "").upper() not in {
        "INTERRUPTED",
        "SUSPENDED",
    }:
        return {
            "outcome": "NOT_RESUMABLE",
            "reason": "Only interrupted or suspended runs can resume a step.",
        }
    attempt = checkpoint_store.latest_step_attempt(
        run_id=run_id,
        step_id=step_id,
    )
    if attempt is None:
        return {
            "outcome": "NOT_RESUMABLE",
            "reason": "No durable PlanStep attempt is available.",
        }
    outcome = str(attempt.get("resumability") or "NOT_RESUMABLE")
    if attempt.get("status") == "COMPLETED":
        return {
            "outcome": "ALREADY_COMPLETED",
            "attempt_id": attempt["attempt_id"],
            "step_id": attempt["step_id"],
        }
    if checkpoint_store.has_pending_decision(
        run_id=run_id,
        kind="approval",
    ):
        return {
            "outcome": "REQUIRES_APPROVAL",
            "attempt_id": attempt["attempt_id"],
            "step_id": attempt["step_id"],
        }
    if outcome != "SAFE_TO_RESTART":
        return {
            "outcome": (
                outcome
                if outcome in {
                    "REQUIRES_VERIFICATION",
                    "REQUIRES_APPROVAL",
                    "NOT_RESUMABLE",
                }
                else "NOT_RESUMABLE"
            ),
            "attempt_id": attempt["attempt_id"],
            "step_id": attempt["step_id"],
            "receipt_status": attempt.get("receipt_status"),
        }
    runners = (
        builtin_worker_runners()
        if worker_runners is None
        else dict(worker_runners)
    )
    runner = runners.get(str(attempt["worker_kind"]))
    if runner is None:
        return {
            "outcome": "NOT_RESUMABLE",
            "reason": "The original worker is unavailable.",
            "step_id": attempt["step_id"],
        }
    inputs = dict(attempt.get("input_refs") or {})
    lifecycle = LangGraphRunLifecycle.recover(
        checkpoint_store=checkpoint_store,
        thread_id=str(checkpoint_state["thread_id"]),
        checkpoint_ns=str(checkpoint_state["checkpoint_namespace"]),
    )
    if lifecycle is None:
        return {"outcome": "NOT_RESUMABLE", "reason": "Lifecycle unavailable."}
    next_attempt = checkpoint_store.create_step_attempt(
        run_id=run_id,
        plan_id=str(attempt["plan_id"]),
        plan_revision=int(attempt["plan_revision"]),
        step_id=str(attempt["step_id"]),
        worker_kind=str(attempt["worker_kind"]),
        input_refs=inputs,
        side_effect_classification=str(
            attempt["side_effect_classification"]
        ),
    )
    supervisor_state = {
        "worker_context": {
            "run_id": run_id,
            "owner": owner,
            "plan_step_id": attempt["step_id"],
            "attempt_id": next_attempt["attempt_id"],
            "dependency_outputs": dict(
                inputs.get("dependency_outputs") or {}
            ),
            "cancellation_scope": lifecycle.cancellation,
        }
    }
    handoff = build_worker_handoff(
        str(attempt["worker_kind"]),
        str(inputs.get("objective") or ""),
        supervisor_state=supervisor_state,
    )
    result = await _await(
        runner(
            handoff,
            session_id=inputs.get("session_id"),
        )
    )
    if not isinstance(result, WorkerResult):
        checkpoint_store.complete_step_attempt(
            str(next_attempt["attempt_id"]),
            status="FAILED",
            error="Resumed worker returned an invalid result.",
        )
        return {
            "outcome": "NOT_RESUMABLE",
            "reason": "Resumed worker returned an invalid result.",
        }
    checkpoint_store.complete_step_attempt(
        str(next_attempt["attempt_id"]),
        status="COMPLETED" if result.status == "completed" else "FAILED",
        result=_result_payload(result),
        error=result.error,
    )
    if result.status != "completed":
        lifecycle.suspend(
            "INTERRUPTED",
            stage="step_resume",
            reason=result.error or result.summary,
        )
        return {
            "outcome": "NOT_RESUMABLE",
            "attempt_id": next_attempt["attempt_id"],
            "step_id": attempt["step_id"],
            "result": _result_payload(result),
        }
    completed = list(lifecycle.completed_steps)
    if attempt["step_id"] not in completed:
        completed.append(str(attempt["step_id"]))
    dependencies = dict(lifecycle.dependency_results)
    dependencies[str(attempt["step_id"])] = _result_payload(result)
    lifecycle.update_runtime_state(
        completed_steps=completed,
        dependency_results=dependencies,
        artifact_references=lifecycle.artifact_references,
    )
    lifecycle.suspend(
        "SUSPENDED",
        stage="step_resume",
        reason="Safe PlanStep completed; remaining plan requires continuation.",
        resumed_step_id=attempt["step_id"],
        attempt_id=next_attempt["attempt_id"],
    )
    return {
        "outcome": "SAFE_TO_RESTART",
        "execution_status": "COMPLETED",
        "attempt_id": next_attempt["attempt_id"],
        "previous_attempt_id": attempt["attempt_id"],
        "step_id": attempt["step_id"],
        "run_status": "SUSPENDED",
        "result": _result_payload(result),
    }


__all__ = ["resume_interrupted_step"]
