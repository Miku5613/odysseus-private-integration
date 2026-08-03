from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol

from src.agent_runtime.execution_resolver import ExecutionResolution
from src.agent_runtime.plan_execution import (
    PlanExecutionState,
    ReplanRequest,
    complete_step,
    fail_step,
    partial_step,
    request_replan,
)
from src.agent_runtime.supervisor_contracts import (
    SupervisorState,
    WorkerResult,
)
from src.agent_runtime.task_plan import PlanStep
from src.agent_runtime.worker_dispatcher import (
    build_worker_handoff,
    builtin_worker_runners,
)


WorkerRunner = Callable[..., Any]


class StepSurfaceExecutor(Protocol):
    """Structured executor for one non-native execution surface."""

    def __call__(
        self,
        step: PlanStep,
        resolution: ExecutionResolution,
        context: "StepExecutionContext",
    ) -> WorkerResult | Awaitable[WorkerResult]:
        ...


@dataclass(frozen=True)
class StepExecutionContext:
    """Runtime context propagated unchanged into one planned step execution."""

    supervisor_state: SupervisorState | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class StepExecutionOutcome:
    """Result of recording one WorkerResult against immutable plan state."""

    state: PlanExecutionState
    worker_result: WorkerResult
    replan_request: ReplanRequest | None = None


class StepExecutionUnavailable(RuntimeError):
    """Raised when a resolved surface has no structured step executor."""


def _selected_surface(
    step: PlanStep,
    resolution: ExecutionResolution,
):
    if resolution.step_id != step.id:
        raise ValueError(
            "ExecutionResolution step_id does not match PlanStep: "
            f"{resolution.step_id!r} != {step.id!r}"
        )

    if resolution.status != "resolved":
        raise ValueError(
            "Only resolved PlanSteps may be executed; "
            f"got status={resolution.status!r}"
        )

    surface = resolution.selected_surface
    if surface is None:
        raise ValueError(
            "Resolved ExecutionResolution requires selected_surface"
        )

    return surface


def _validate_worker_result(
    result: Any,
    *,
    expected_capability: str | None = None,
) -> WorkerResult:
    if not isinstance(result, WorkerResult):
        raise TypeError(
            "Step executor must return WorkerResult; "
            f"got {type(result).__name__}"
        )

    if (
        expected_capability is not None
        and result.capability != expected_capability
    ):
        raise ValueError(
            "WorkerResult capability does not match resolved capability: "
            f"{result.capability!r} != {expected_capability!r}"
        )

    return result


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def execute_resolved_step(
    step: PlanStep,
    resolution: ExecutionResolution,
    *,
    context: StepExecutionContext | None = None,
    worker_runners: Mapping[str, WorkerRunner] | None = None,
    surface_executors: Mapping[str, StepSurfaceExecutor] | None = None,
) -> WorkerResult:
    """Execute one resolved semantic step through its existing execution owner.

    Native LangGraph resolutions reuse the canonical WorkerHandoff/WorkerResult
    boundary and the already-implemented capability runners. This function does
    not fabricate a supervisor routing decision and does not select capabilities.

    Legacy and other registered surfaces require an explicitly registered
    structured surface executor. They are never silently rerouted through a
    native worker or parsed from a turn-level SSE stream.
    """

    surface = _selected_surface(
        step,
        resolution,
    )
    execution_context = (
        context
        if context is not None
        else StepExecutionContext()
    )

    if surface.kind == "langgraph":
        capability = surface.capability_name
        if not capability:
            raise ValueError(
                "Native LangGraph execution surface requires capability_name"
            )

        runners = (
            builtin_worker_runners()
            if worker_runners is None
            else dict(worker_runners)
        )
        runner = runners.get(capability)

        if runner is None:
            raise StepExecutionUnavailable(
                "No implemented native worker runner for resolved capability "
                f"{capability!r}"
            )

        handoff = build_worker_handoff(
            capability,
            step.objective,
            supervisor_state=(
                execution_context.supervisor_state
            ),
        )

        value = runner(
            handoff,
            session_id=execution_context.session_id,
        )
        result = await _await_if_needed(value)

        return _validate_worker_result(
            result,
            expected_capability=capability,
        )

    executors = (
        {}
        if surface_executors is None
        else dict(surface_executors)
    )
    executor = executors.get(surface.id)

    if executor is None:
        raise StepExecutionUnavailable(
            "Resolved execution surface has no structured step executor: "
            f"{surface.id!r} ({surface.kind})"
        )

    value = executor(
        step,
        resolution,
        execution_context,
    )
    result = await _await_if_needed(value)

    return _validate_worker_result(result)


def _clarification_reason(
    result: WorkerResult,
) -> str:
    summary = str(result.summary or "").strip()
    reason = (
        summary
        or "The worker requires clarification before execution can continue."
    )

    missing = tuple(
        str(value).strip()
        for value in result.missing_information
        if str(value).strip()
    )
    if missing:
        reason += " Missing information: " + "; ".join(missing)

    return reason


def apply_worker_result_to_plan_state(
    state: PlanExecutionState,
    step_id: str,
    result: WorkerResult,
) -> StepExecutionOutcome:
    """Record a normalized WorkerResult without collapsing clarification to failure.

    The orchestration layer owns lifecycle transitions. A completed worker marks
    the running step completed; a failed worker marks it failed. Clarification
    preserves the current plan state and emits a ReplanRequest so P3-C3 can
    decide whether to ask the user or revise the plan.
    """

    if not isinstance(result, WorkerResult):
        raise TypeError(
            "apply_worker_result_to_plan_state requires WorkerResult"
        )

    if result.status == "completed":
        updated = complete_step(
            state,
            step_id,
            output=result,
        )
        return StepExecutionOutcome(
            state=updated,
            worker_result=result,
        )

    if result.status == "failed":
        error = str(
            result.error
            or result.summary
            or "Worker execution failed."
        ).strip()
        updated = fail_step(
            state,
            step_id,
            error=error,
        )
        return StepExecutionOutcome(
            state=updated,
            worker_result=result,
        )

    if result.status == "partial":
        return StepExecutionOutcome(
            state=partial_step(
                state,
                step_id,
                output=result,
            ),
            worker_result=result,
        )

    if result.status == "needs_clarification":
        replan = request_replan(
            state,
            reason=_clarification_reason(result),
            trigger_step_id=step_id,
        )
        return StepExecutionOutcome(
            state=state,
            worker_result=result,
            replan_request=replan,
        )

    raise ValueError(
        "Unsupported WorkerResult status for planned-step execution: "
        f"{result.status!r}"
    )


async def execute_and_record_resolved_step(
    state: PlanExecutionState,
    step: PlanStep,
    resolution: ExecutionResolution,
    *,
    context: StepExecutionContext | None = None,
    worker_runners: Mapping[str, WorkerRunner] | None = None,
    surface_executors: Mapping[str, StepSurfaceExecutor] | None = None,
) -> StepExecutionOutcome:
    """Execute one already-running step and record its normalized result."""

    result = await execute_resolved_step(
        step,
        resolution,
        context=context,
        worker_runners=worker_runners,
        surface_executors=surface_executors,
    )

    return apply_worker_result_to_plan_state(
        state,
        step.id,
        result,
    )


__all__ = [
    "StepExecutionContext",
    "StepExecutionOutcome",
    "StepExecutionUnavailable",
    "StepSurfaceExecutor",
    "apply_worker_result_to_plan_state",
    "execute_and_record_resolved_step",
    "execute_resolved_step",
]
