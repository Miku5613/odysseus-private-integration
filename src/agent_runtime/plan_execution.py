from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Literal

from src.agent_runtime.task_plan import (
    PlanStep,
    TaskPlan,
    blocked_step_ids,
    ready_step_ids,
    topological_step_ids,
)


StepStatus = Literal[
    "pending",
    "ready",
    "running",
    "completed",
    "failed",
    "blocked",
]

_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "blocked",
}


@dataclass(frozen=True)
class PlanStepExecution:
    """Immutable execution state for one semantic PlanStep.

    This records orchestration state only. It does not select or identify any
    worker, tool, capability, action, runtime, provider, or backend.
    """

    step_id: str
    status: StepStatus
    output: Any = None
    artifact_refs: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class PlanExecutionState:
    """Immutable runtime state for one TaskPlan revision."""

    plan: TaskPlan
    step_states: tuple[PlanStepExecution, ...]
    revision: int = 1
    prior_plan_fingerprints: tuple[str, ...] = ()
    replan_reason: str = ""


@dataclass(frozen=True)
class ReplanRequest:
    """Backend-independent request to reconsider the current semantic plan."""

    reason: str
    from_revision: int
    plan_fingerprint: str
    trigger_step_id: str | None
    completed_step_ids: tuple[str, ...]
    failed_step_ids: tuple[str, ...]
    blocked_step_ids: tuple[str, ...]


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")

    return normalized


def _clean_artifact_refs(
    values: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = _required_text(
            value,
            field="artifact reference",
        )
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    return tuple(result)


def task_plan_fingerprint(plan: TaskPlan) -> str:
    """Return a stable semantic fingerprint for plan revision lineage."""

    payload = {
        "objective": plan.objective,
        "steps": [
            {
                "id": step.id,
                "objective": step.objective,
                "depends_on": list(step.depends_on),
                "work_domains": list(step.work_domains),
                "requires_external_state": (
                    step.requires_external_state
                ),
                "requires_mutation": step.requires_mutation,
            }
            for step in plan.steps
        ],
        "reasoning_summary": plan.reasoning_summary,
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def _state_map(
    state: PlanExecutionState,
) -> dict[str, PlanStepExecution]:
    return {
        step_state.step_id: step_state
        for step_state in state.step_states
    }


def _validate_execution_state(
    state: PlanExecutionState,
) -> None:
    if state.revision < 1:
        raise ValueError(
            "PlanExecutionState revision must be at least 1"
        )

    plan_ids = tuple(
        step.id
        for step in state.plan.steps
    )
    state_ids = tuple(
        step_state.step_id
        for step_state in state.step_states
    )

    if len(state_ids) != len(set(state_ids)):
        raise ValueError(
            "PlanExecutionState step ids must be unique"
        )

    if set(plan_ids) != set(state_ids):
        raise ValueError(
            "PlanExecutionState must contain exactly one state "
            "for every PlanStep"
        )

    valid_statuses = {
        "pending",
        "ready",
        "running",
        "completed",
        "failed",
        "blocked",
    }

    for step_state in state.step_states:
        if step_state.status not in valid_statuses:
            raise ValueError(
                "Unsupported PlanStepExecution status: "
                + str(step_state.status)
            )

        if (
            step_state.status == "failed"
            and not step_state.error.strip()
        ):
            raise ValueError(
                f"Failed step {step_state.step_id} "
                "must include an error"
            )

        if (
            step_state.status != "failed"
            and step_state.error
        ):
            raise ValueError(
                f"Only failed step {step_state.step_id} "
                "may contain an error"
            )

    topological_step_ids(state.plan)


def _refresh_waiting_statuses(
    state: PlanExecutionState,
) -> PlanExecutionState:
    states = _state_map(state)

    completed = tuple(
        step_id
        for step_id, step_state in states.items()
        if step_state.status == "completed"
    )
    failed = tuple(
        step_id
        for step_id, step_state in states.items()
        if step_state.status == "failed"
    )

    ready = set(
        ready_step_ids(
            state.plan,
            completed_step_ids=completed,
            failed_step_ids=failed,
        )
    )
    blocked = set(
        blocked_step_ids(
            state.plan,
            completed_step_ids=completed,
            failed_step_ids=failed,
        )
    )

    refreshed: list[PlanStepExecution] = []

    for step in state.plan.steps:
        current = states[step.id]

        if current.status in {
            "running",
            "completed",
            "failed",
        }:
            refreshed.append(current)
            continue

        if step.id in blocked:
            refreshed.append(
                replace(
                    current,
                    status="blocked",
                )
            )
            continue

        if step.id in ready:
            refreshed.append(
                replace(
                    current,
                    status="ready",
                )
            )
            continue

        refreshed.append(
            replace(
                current,
                status="pending",
            )
        )

    result = replace(
        state,
        step_states=tuple(refreshed),
    )
    _validate_execution_state(result)
    return result


def initialize_plan_execution(
    plan: TaskPlan,
    *,
    revision: int = 1,
    prior_plan_fingerprints: tuple[str, ...] = (),
    replan_reason: str = "",
) -> PlanExecutionState:
    """Create immutable execution state without choosing an executor."""

    if revision < 1:
        raise ValueError(
            "PlanExecutionState revision must be at least 1"
        )

    topological_step_ids(plan)

    initial = PlanExecutionState(
        plan=plan,
        step_states=tuple(
            PlanStepExecution(
                step_id=step.id,
                status="pending",
            )
            for step in plan.steps
        ),
        revision=revision,
        prior_plan_fingerprints=tuple(
            prior_plan_fingerprints
        ),
        replan_reason=str(
            replan_reason or ""
        ).strip(),
    )

    return _refresh_waiting_statuses(initial)


def step_execution(
    state: PlanExecutionState,
    step_id: str,
) -> PlanStepExecution:
    normalized = _required_text(
        step_id,
        field="step_id",
    )

    for step_state in state.step_states:
        if step_state.step_id == normalized:
            return step_state

    raise KeyError(normalized)


def _replace_step_state(
    state: PlanExecutionState,
    replacement: PlanStepExecution,
) -> PlanExecutionState:
    if replacement.step_id not in {
        step.id
        for step in state.plan.steps
    }:
        raise KeyError(replacement.step_id)

    updated = replace(
        state,
        step_states=tuple(
            replacement
            if current.step_id == replacement.step_id
            else current
            for current in state.step_states
        ),
    )

    return _refresh_waiting_statuses(updated)


def start_step(
    state: PlanExecutionState,
    step_id: str,
) -> PlanExecutionState:
    """Transition one ready semantic step to running."""

    current = step_execution(
        state,
        step_id,
    )

    if current.status != "ready":
        raise ValueError(
            f"Step {current.step_id} cannot start from "
            f"status {current.status}"
        )

    return _replace_step_state(
        state,
        replace(
            current,
            status="running",
        ),
    )


def complete_step(
    state: PlanExecutionState,
    step_id: str,
    *,
    output: Any = None,
    artifact_refs: tuple[str, ...] | list[str] = (),
) -> PlanExecutionState:
    """Record normalized step completion and unlock dependents."""

    current = step_execution(
        state,
        step_id,
    )

    if current.status != "running":
        raise ValueError(
            f"Step {current.step_id} cannot complete from "
            f"status {current.status}"
        )

    return _replace_step_state(
        state,
        replace(
            current,
            status="completed",
            output=output,
            artifact_refs=_clean_artifact_refs(
                artifact_refs
            ),
            error="",
        ),
    )


def fail_step(
    state: PlanExecutionState,
    step_id: str,
    *,
    error: str,
) -> PlanExecutionState:
    """Record failure and transitively block dependent steps."""

    current = step_execution(
        state,
        step_id,
    )

    if current.status != "running":
        raise ValueError(
            f"Step {current.step_id} cannot fail from "
            f"status {current.status}"
        )

    normalized_error = _required_text(
        error,
        field="error",
    )

    return _replace_step_state(
        state,
        replace(
            current,
            status="failed",
            error=normalized_error,
        ),
    )


def request_replan(
    state: PlanExecutionState,
    *,
    reason: str,
    trigger_step_id: str | None = None,
) -> ReplanRequest:
    """Capture why orchestration needs a revised semantic TaskPlan."""

    normalized_reason = _required_text(
        reason,
        field="replan reason",
    )

    normalized_trigger: str | None = None

    if trigger_step_id is not None:
        normalized_trigger = _required_text(
            trigger_step_id,
            field="trigger_step_id",
        )
        step_execution(
            state,
            normalized_trigger,
        )

    completed = tuple(
        step_state.step_id
        for step_state in state.step_states
        if step_state.status == "completed"
    )
    failed = tuple(
        step_state.step_id
        for step_state in state.step_states
        if step_state.status == "failed"
    )
    blocked = tuple(
        step_state.step_id
        for step_state in state.step_states
        if step_state.status == "blocked"
    )

    return ReplanRequest(
        reason=normalized_reason,
        from_revision=state.revision,
        plan_fingerprint=task_plan_fingerprint(
            state.plan
        ),
        trigger_step_id=normalized_trigger,
        completed_step_ids=completed,
        failed_step_ids=failed,
        blocked_step_ids=blocked,
    )


def _matching_step(
    plan: TaskPlan,
    step_id: str,
) -> PlanStep:
    return plan.step_by_id(step_id)


def begin_replanned_execution(
    previous_state: PlanExecutionState,
    revised_plan: TaskPlan,
    *,
    reason: str,
    reusable_step_ids: tuple[str, ...] | list[str] = (),
) -> PlanExecutionState:
    """Start a new plan revision with explicit, safe output reuse.

    Completed outputs are never reused implicitly. A caller may explicitly
    preserve a completed step only when the revised plan contains the exact
    same semantic PlanStep definition.
    """

    normalized_reason = _required_text(
        reason,
        field="replan reason",
    )

    requested_reuse: list[str] = []
    seen: set[str] = set()

    for value in reusable_step_ids:
        step_id = _required_text(
            value,
            field="reusable_step_id",
        )
        if step_id in seen:
            continue
        seen.add(step_id)
        requested_reuse.append(step_id)

    new_state = initialize_plan_execution(
        revised_plan,
        revision=previous_state.revision + 1,
        prior_plan_fingerprints=(
            previous_state.prior_plan_fingerprints
            + (
                task_plan_fingerprint(
                    previous_state.plan
                ),
            )
        ),
        replan_reason=normalized_reason,
    )

    if not requested_reuse:
        return new_state

    previous_by_id = _state_map(previous_state)
    new_by_id = _state_map(new_state)

    for step_id in requested_reuse:
        previous_execution = previous_by_id.get(
            step_id
        )

        if previous_execution is None:
            raise ValueError(
                f"Reusable step {step_id} did not exist "
                "in the previous plan"
            )

        if previous_execution.status != "completed":
            raise ValueError(
                f"Reusable step {step_id} must be completed"
            )

        try:
            previous_step = _matching_step(
                previous_state.plan,
                step_id,
            )
            revised_step = _matching_step(
                revised_plan,
                step_id,
            )
        except KeyError as exc:
            raise ValueError(
                f"Reusable step {step_id} must exist "
                "in both plan revisions"
            ) from exc

        if previous_step != revised_step:
            raise ValueError(
                f"Reusable step {step_id} changed semantic "
                "definition between plan revisions"
            )

        new_by_id[step_id] = replace(
            new_by_id[step_id],
            status="completed",
            output=previous_execution.output,
            artifact_refs=(
                previous_execution.artifact_refs
            ),
            error="",
        )

    reused = replace(
        new_state,
        step_states=tuple(
            new_by_id[step.id]
            for step in revised_plan.steps
        ),
    )

    return _refresh_waiting_statuses(reused)


def terminal_step_ids(
    state: PlanExecutionState,
) -> tuple[str, ...]:
    return tuple(
        step_state.step_id
        for step_state in state.step_states
        if step_state.status in _TERMINAL_STATUSES
    )


__all__ = [
    "PlanExecutionState",
    "PlanStepExecution",
    "ReplanRequest",
    "StepStatus",
    "begin_replanned_execution",
    "complete_step",
    "fail_step",
    "initialize_plan_execution",
    "request_replan",
    "start_step",
    "step_execution",
    "task_plan_fingerprint",
    "terminal_step_ids",
]
