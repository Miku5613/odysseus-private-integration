from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from src.agent_runtime.execution_resolver import (
    ExecutionResolverModel,
    ExecutionSurfaceDescriptor,
    resolve_plan_step,
)
from src.agent_runtime.plan_execution import (
    PlanExecutionState,
    ReplanRequest,
    fail_step,
    initialize_plan_execution,
    request_replan,
    start_step,
    step_execution,
)
from src.agent_runtime.step_execution import (
    StepExecutionContext,
    StepExecutionUnavailable,
    StepSurfaceExecutor,
    execute_and_record_resolved_step,
)
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.task_plan import TaskPlan
from src.agent_runtime.worker_dispatcher import WorkerRunner


PlanRunStatus = Literal[
    "completed",
    "failed",
    "needs_replan",
    "execution_unavailable",
    "stalled",
]


@dataclass(frozen=True)
class PlanOrchestrationResult:
    """Immutable outcome of one dependency-aware orchestration run."""

    state: PlanExecutionState
    status: PlanRunStatus
    worker_results: tuple[WorkerResult, ...] = ()
    execution_order: tuple[str, ...] = ()
    replan_request: ReplanRequest | None = None
    message: str = ""


def _status_ids(
    state: PlanExecutionState,
    status: str,
) -> tuple[str, ...]:
    return tuple(
        item.step_id
        for item in state.step_states
        if item.status == status
    )


def _all_completed(
    state: PlanExecutionState,
) -> bool:
    return bool(state.step_states) and all(
        item.status == "completed"
        for item in state.step_states
    )


def _clarification_reason(
    resolution,
) -> str:
    summary = str(
        resolution.reasoning_summary or ""
    ).strip()
    reason = (
        summary
        or "Execution resolution requires clarification before the step can run."
    )

    missing = tuple(
        str(value).strip()
        for value in resolution.missing_information
        if str(value).strip()
    )
    if missing:
        reason += " Missing information: " + "; ".join(missing)

    return reason


def _unsupported_reason(
    resolution,
) -> str:
    return str(
        resolution.reasoning_summary
        or "No compatible execution surface is available for this PlanStep."
    ).strip()


async def run_task_plan(
    plan: TaskPlan,
    *,
    resolver_model: ExecutionResolverModel,
    state: PlanExecutionState | None = None,
    surfaces: tuple[ExecutionSurfaceDescriptor, ...] | None = None,
    context: StepExecutionContext | None = None,
    worker_runners: Mapping[str, WorkerRunner] | None = None,
    surface_executors: Mapping[str, StepSurfaceExecutor] | None = None,
    max_step_executions: int = 64,
) -> PlanOrchestrationResult:
    """Execute ready PlanSteps in stable dependency order until a terminal boundary.

    This orchestrator owns sequencing only. Semantic decomposition belongs to the
    Task Planner, execution-surface selection belongs to the Execution Resolver,
    and concrete execution belongs to existing workers or explicitly registered
    structured surface executors.

    Resolution clarification and worker clarification stop the run with a
    ReplanRequest. Missing structured execution for a resolved surface is
    reported as execution_unavailable rather than being mislabeled as semantic
    plan failure. Independent ready work continues after an actual step failure.
    """

    if max_step_executions < 1:
        raise ValueError(
            "max_step_executions must be at least 1"
        )

    current = (
        initialize_plan_execution(plan)
        if state is None
        else state
    )

    if current.plan != plan:
        raise ValueError(
            "Provided PlanExecutionState belongs to a different TaskPlan"
        )

    worker_results: list[WorkerResult] = []
    execution_order: list[str] = []
    executions = 0

    while True:
        if _all_completed(current):
            return PlanOrchestrationResult(
                state=current,
                status="completed",
                worker_results=tuple(worker_results),
                execution_order=tuple(execution_order),
                message="All planned steps completed.",
            )

        ready_ids = _status_ids(
            current,
            "ready",
        )

        if not ready_ids:
            failed_ids = _status_ids(
                current,
                "failed",
            )
            blocked_ids = _status_ids(
                current,
                "blocked",
            )
            running_ids = _status_ids(
                current,
                "running",
            )

            if failed_ids or blocked_ids:
                return PlanOrchestrationResult(
                    state=current,
                    status="failed",
                    worker_results=tuple(worker_results),
                    execution_order=tuple(execution_order),
                    message=(
                        "Plan reached a terminal failure boundary. "
                        f"failed={list(failed_ids)}, "
                        f"blocked={list(blocked_ids)}"
                    ),
                )

            return PlanOrchestrationResult(
                state=current,
                status="stalled",
                worker_results=tuple(worker_results),
                execution_order=tuple(execution_order),
                message=(
                    "Plan has no ready steps and is not complete."
                    + (
                        " Running steps: "
                        + ", ".join(running_ids)
                        if running_ids
                        else ""
                    )
                ),
            )

        progressed = False

        for step_id in ready_ids:
            if executions >= max_step_executions:
                return PlanOrchestrationResult(
                    state=current,
                    status="stalled",
                    worker_results=tuple(worker_results),
                    execution_order=tuple(execution_order),
                    message=(
                        "Step execution safety limit reached before the plan "
                        "reached a terminal state."
                    ),
                )

            step = plan.step_by_id(step_id)

            resolution = resolve_plan_step(
                step,
                model=resolver_model,
                surfaces=surfaces,
            )

            if resolution.status == "clarification":
                replan = request_replan(
                    current,
                    reason=_clarification_reason(
                        resolution
                    ),
                    trigger_step_id=step_id,
                )
                return PlanOrchestrationResult(
                    state=current,
                    status="needs_replan",
                    worker_results=tuple(worker_results),
                    execution_order=tuple(execution_order),
                    replan_request=replan,
                    message=(
                        "Execution resolution requires clarification."
                    ),
                )

            if resolution.status == "unsupported":
                current = start_step(
                    current,
                    step_id,
                )
                current = fail_step(
                    current,
                    step_id,
                    error=_unsupported_reason(
                        resolution
                    ),
                )
                execution_order.append(
                    step_id
                )
                executions += 1
                progressed = True
                continue

            pre_step_state = current
            started_state = start_step(
                current,
                step_id,
            )

            try:
                outcome = await execute_and_record_resolved_step(
                    started_state,
                    step,
                    resolution,
                    context=context,
                    worker_runners=worker_runners,
                    surface_executors=surface_executors,
                )
            except StepExecutionUnavailable as exc:
                replan = request_replan(
                    pre_step_state,
                    reason=str(exc),
                    trigger_step_id=step_id,
                )
                return PlanOrchestrationResult(
                    state=pre_step_state,
                    status="execution_unavailable",
                    worker_results=tuple(worker_results),
                    execution_order=tuple(execution_order),
                    replan_request=replan,
                    message=str(exc),
                )

            current = outcome.state
            worker_results.append(
                outcome.worker_result
            )
            execution_order.append(
                step_id
            )
            executions += 1
            progressed = True

            if outcome.replan_request is not None:
                return PlanOrchestrationResult(
                    state=current,
                    status="needs_replan",
                    worker_results=tuple(worker_results),
                    execution_order=tuple(execution_order),
                    replan_request=outcome.replan_request,
                    message=(
                        "Worker execution requires clarification or replanning."
                    ),
                )

        if not progressed:
            return PlanOrchestrationResult(
                state=current,
                status="stalled",
                worker_results=tuple(worker_results),
                execution_order=tuple(execution_order),
                message=(
                    "Orchestrator made no progress across the ready frontier."
                ),
            )


__all__ = [
    "PlanOrchestrationResult",
    "PlanRunStatus",
    "run_task_plan",
]
