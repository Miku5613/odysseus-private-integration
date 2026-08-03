from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Mapping

from src.agent_runtime.execution_resolver import (
    ExecutionResolverModel,
    ExecutionSurfaceDescriptor,
)
from src.agent_runtime.plan_execution import (
    PlanExecutionState,
    ReplanRequest,
    begin_replanned_execution,
    task_plan_fingerprint,
)
from src.agent_runtime.plan_orchestrator import (
    PlanOrchestrationResult,
    run_task_plan,
)
from src.agent_runtime.plan_source_contract import (
    constrain_replanned_sources,
)
from src.agent_runtime.step_execution import (
    StepExecutionContext,
    StepSurfaceExecutor,
)
from src.agent_runtime.structured_llm import (
    invoke_structured_model_async,
)
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.task_plan import (
    PlanStep,
    TaskPlan,
)
from src.agent_runtime.task_planner import (
    TaskPlannerModel,
    parse_task_plan_response,
)
from src.agent_runtime.worker_dispatcher import WorkerRunner


ReplanStopReason = Literal[
    "terminal",
    "awaiting_replan_context",
    "revision_limit",
]


@dataclass(frozen=True)
class ReplanContext:
    """New information supplied by the higher-level runtime for one replan."""

    planning_context: str
    execution_context: StepExecutionContext | None = None


ReplanContextProvider = Callable[
    [ReplanRequest, PlanExecutionState],
    ReplanContext
    | str
    | None
    | Awaitable[ReplanContext | str | None],
]


@dataclass(frozen=True)
class ReplanControlResult:
    """Outcome of executing a plan through zero or more semantic revisions."""

    orchestration: PlanOrchestrationResult
    revisions_applied: int
    plan_history: tuple[TaskPlan, ...]
    stopped_reason: ReplanStopReason


def _required_text(
    value: Any,
    *,
    field: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a string"
        )

    normalized = value.strip()
    if not normalized:
        raise ValueError(
            f"{field} must not be empty"
        )

    return normalized


def _step_payload(
    step: PlanStep,
) -> dict[str, Any]:
    return {
        "id": step.id,
        "objective": step.objective,
        "depends_on": list(
            step.depends_on
        ),
        "work_domains": list(
            step.work_domains
        ),
        "uses_dependency_state": (
            step.uses_dependency_state
        ),
        "requires_external_state": (
            step.requires_external_state
        ),
        "requires_mutation": (
            step.requires_mutation
        ),
        "requested_sources": list(step.requested_sources),
        "excluded_sources": list(step.excluded_sources),
        "retrieval_concepts": list(step.retrieval_concepts),
    }


def _plan_payload(
    plan: TaskPlan,
) -> dict[str, Any]:
    return {
        "objective": plan.objective,
        "steps": [
            _step_payload(step)
            for step in plan.steps
        ],
        "reasoning_summary": (
            plan.reasoning_summary
        ),
    }


def _output_summary(
    output: Any,
) -> Any:
    if isinstance(
        output,
        WorkerResult,
    ):
        return {
            "capability": output.capability,
            "status": output.status,
            "summary": output.summary,
            "missing_information": list(
                output.missing_information
            ),
            "error": output.error,
        }

    if output is None:
        return None

    return str(output)


def _execution_payload(
    state: PlanExecutionState,
) -> list[dict[str, Any]]:
    return [
        {
            "step_id": item.step_id,
            "status": item.status,
            "output_summary": _output_summary(
                item.output
            ),
            "artifact_refs": list(
                item.artifact_refs
            ),
            "error": item.error,
        }
        for item in state.step_states
    ]


def _validate_replan_request(
    state: PlanExecutionState,
    request: ReplanRequest,
) -> None:
    if request.from_revision != state.revision:
        raise ValueError(
            "ReplanRequest revision does not match PlanExecutionState"
        )

    fingerprint = task_plan_fingerprint(
        state.plan
    )
    if request.plan_fingerprint != fingerprint:
        raise ValueError(
            "ReplanRequest fingerprint does not match current TaskPlan"
        )


def build_task_replanner_prompt(
    query: str,
    state: PlanExecutionState,
    request: ReplanRequest,
    *,
    replan_context: str,
) -> str:
    """Build a backend-independent prompt for one semantic plan revision."""

    normalized_query = _required_text(
        query,
        field="replan query",
    )
    normalized_context = _required_text(
        replan_context,
        field="replan_context",
    )
    _validate_replan_request(
        state,
        request,
    )

    schema = {
        "objective": "overall user-facing objective",
        "steps": [
            {
                "id": "stable_short_step_id",
                "objective": (
                    "semantic unit of work to accomplish"
                ),
                "depends_on": [
                    "prior_step_id"
                ],
                "work_domains": [
                    "semantic_domain_label"
                ],
                "uses_dependency_state": False,
                "requires_external_state": False,
                "requires_mutation": False,
                "requested_sources": [
                    "previously admitted source required by this step"
                ],
                "excluded_sources": [
                    "sources forbidden by the current plan"
                ],
                "retrieval_concepts": [
                    "semantic evidence concept for this step"
                ],
            }
        ],
        "reasoning_summary": (
            "brief explanation of the revised decomposition"
        ),
    }

    rules = [
        "You are the Odysseus Task Replanner.",
        "",
        "Revise the semantic TaskPlan using the new context and prior execution "
        "outcomes.",
        "",
        "PERMANENT OWNERSHIP RULES:",
        "1. Revise WHAT work must be accomplished and dependency structure only.",
        "2. Do NOT select workers, tools, capabilities, actions, providers, "
        "models, runtimes, backends, MCP servers, LangGraph, or Legacy.",
        "3. Never delete a semantically necessary step merely because a runtime "
        "or backend could not execute it.",
        "4. Do not fabricate missing facts. Use only the user request, prior "
        "execution outcomes, and the supplied replan context.",
        "5. Preserve an already-completed PlanStep exactly when its semantic work "
        "is still valid. Exact preservation allows deterministic output reuse.",
        "6. If a completed step must change semantically, change it honestly; its "
        "previous output will not be reused.",
        "7. Dependencies must reference ids in the revised plan and remain "
        "acyclic.",
        "8. Preserve uses_dependency_state semantics: true means the step "
        "directly consumes completed depends_on outputs. Pure reasoning over "
        "dependency outputs does not require new external state.",
        "9. requires_external_state is only for NEW state beyond the request, "
        "context, and completed dependency outputs.",
        "10. Planning never grants permission to mutate persistent or external "
        "state.",
        "11. work_domains remain semantic labels only.",
        "12. Preserve each step's admitted requested_sources and "
        "excluded_sources. A revision must never widen to a source absent from "
        "the current plan.",
        "13. Keep retrieval_concepts scoped to the revised step objective.",
        "14. Return only one JSON object matching the TaskPlan schema.",
        "",
        "ORIGINAL USER REQUEST:",
        normalized_query,
        "",
        "CURRENT PLAN:",
        json.dumps(
            _plan_payload(
                state.plan
            ),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "",
        "CURRENT EXECUTION STATE:",
        json.dumps(
            _execution_payload(
                state
            ),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "",
        "REPLAN REQUEST:",
        json.dumps(
            {
                "reason": request.reason,
                "from_revision": (
                    request.from_revision
                ),
                "trigger_step_id": (
                    request.trigger_step_id
                ),
                "completed_step_ids": list(
                    request.completed_step_ids
                ),
                "partial_step_ids": list(
                    request.partial_step_ids
                ),
                "failed_step_ids": list(
                    request.failed_step_ids
                ),
                "blocked_step_ids": list(
                    request.blocked_step_ids
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "",
        "NEW REPLAN CONTEXT:",
        normalized_context,
        "",
        "REQUIRED JSON SCHEMA:",
        json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        ),
    ]

    return "\n".join(
        rules
    )


def replan_task_from_request(
    query: str,
    state: PlanExecutionState,
    request: ReplanRequest,
    *,
    replan_context: str,
    model: TaskPlannerModel,
) -> TaskPlan:
    """Produce one validated revision through a synchronous model seam."""

    prompt = build_task_replanner_prompt(
        query,
        state,
        request,
        replan_context=replan_context,
    )
    response = model.invoke(
        prompt
    )

    if inspect.isawaitable(response):
        if inspect.iscoroutine(response):
            response.close()
        raise TypeError(
            "TaskPlannerModel returned an awaitable; use "
            "replan_task_from_request_async"
        )

    return constrain_replanned_sources(
        state.plan,
        parse_task_plan_response(response),
    )


async def replan_task_from_request_async(
    query: str,
    state: PlanExecutionState,
    request: ReplanRequest,
    *,
    replan_context: str,
    model: TaskPlannerModel,
) -> TaskPlan:
    """Produce one validated revision without blocking async runtimes."""

    prompt = build_task_replanner_prompt(
        query,
        state,
        request,
        replan_context=replan_context,
    )
    return await invoke_structured_model_async(
        model,
        prompt,
        lambda response: constrain_replanned_sources(
            state.plan,
            parse_task_plan_response(response),
        ),
    )


def exact_reusable_completed_step_ids(
    previous_state: PlanExecutionState,
    revised_plan: TaskPlan,
) -> tuple[str, ...]:
    """Return only completed step ids whose PlanStep definition is unchanged."""

    reusable: list[str] = []

    for execution in previous_state.step_states:
        if execution.status != "completed":
            continue

        try:
            previous_step = (
                previous_state.plan.step_by_id(
                    execution.step_id
                )
            )
            revised_step = (
                revised_plan.step_by_id(
                    execution.step_id
                )
            )
        except KeyError:
            continue

        if previous_step == revised_step:
            reusable.append(
                execution.step_id
            )

    return tuple(
        reusable
    )


def exact_reusable_partial_step_ids(
    previous_state: PlanExecutionState,
    revised_plan: TaskPlan,
) -> tuple[str, ...]:
    """Return partial evidence steps whose semantic definition is unchanged."""

    reusable: list[str] = []
    for execution in previous_state.step_states:
        if execution.status != "partial":
            continue
        try:
            previous_step = previous_state.plan.step_by_id(execution.step_id)
            revised_step = revised_plan.step_by_id(execution.step_id)
        except KeyError:
            continue
        if previous_step == revised_step:
            reusable.append(execution.step_id)
    return tuple(reusable)


def begin_replanned_execution_from_request(
    previous_state: PlanExecutionState,
    revised_plan: TaskPlan,
    request: ReplanRequest,
) -> PlanExecutionState:
    """Begin one new revision with exact completed-step reuse only."""

    _validate_replan_request(
        previous_state,
        request,
    )
    reusable = (
        *exact_reusable_completed_step_ids(
            previous_state,
            revised_plan,
        ),
        *exact_reusable_partial_step_ids(
            previous_state,
            revised_plan,
        ),
    )

    return begin_replanned_execution(
        previous_state,
        revised_plan,
        reason=request.reason,
        reusable_step_ids=reusable,
    )


async def _resolve_replan_context(
    provider: ReplanContextProvider,
    request: ReplanRequest,
    state: PlanExecutionState,
) -> ReplanContext | None:
    value = provider(
        request,
        state,
    )
    if inspect.isawaitable(
        value
    ):
        value = await value

    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):
        return ReplanContext(
            planning_context=_required_text(
                value,
                field="replan_context",
            )
        )

    if not isinstance(
        value,
        ReplanContext,
    ):
        raise TypeError(
            "Replan context provider must return ReplanContext, str, or None"
        )

    return ReplanContext(
        planning_context=_required_text(
            value.planning_context,
            field="replan_context",
        ),
        execution_context=(
            value.execution_context
        ),
    )


async def run_task_plan_with_replanning(
    query: str,
    initial_plan: TaskPlan,
    *,
    resolver_model: ExecutionResolverModel,
    replanner_model: TaskPlannerModel,
    replan_context_provider: ReplanContextProvider | None = None,
    surfaces: tuple[ExecutionSurfaceDescriptor, ...] | None = None,
    context: StepExecutionContext | None = None,
    worker_runners: Mapping[str, WorkerRunner] | None = None,
    surface_executors: Mapping[str, StepSurfaceExecutor] | None = None,
    max_step_executions: int = 64,
    max_revisions: int = 3,
) -> ReplanControlResult:
    """Execute a TaskPlan and apply bounded semantic revisions when context arrives.

    The controller never invents clarification answers. A needs_replan boundary
    is revised only when the higher-level runtime supplies explicit new context.
    execution_unavailable remains an execution-routing boundary and is not
    silently converted into semantic replanning.
    """

    _required_text(
        query,
        field="replan query",
    )
    if max_revisions < 0:
        raise ValueError(
            "max_revisions must be zero or greater"
        )

    current_plan = initial_plan
    current_state: PlanExecutionState | None = None
    current_context = context
    history: list[TaskPlan] = [
        initial_plan
    ]
    revisions_applied = 0

    while True:
        orchestration = await run_task_plan(
            current_plan,
            resolver_model=resolver_model,
            state=current_state,
            surfaces=surfaces,
            context=current_context,
            worker_runners=worker_runners,
            surface_executors=surface_executors,
            max_step_executions=max_step_executions,
        )

        if orchestration.status != "needs_replan":
            return ReplanControlResult(
                orchestration=orchestration,
                revisions_applied=(
                    revisions_applied
                ),
                plan_history=tuple(
                    history
                ),
                stopped_reason="terminal",
            )

        request = orchestration.replan_request
        if request is None:
            raise RuntimeError(
                "needs_replan orchestration result requires ReplanRequest"
            )

        if revisions_applied >= max_revisions:
            return ReplanControlResult(
                orchestration=orchestration,
                revisions_applied=(
                    revisions_applied
                ),
                plan_history=tuple(
                    history
                ),
                stopped_reason="revision_limit",
            )

        if replan_context_provider is None:
            return ReplanControlResult(
                orchestration=orchestration,
                revisions_applied=(
                    revisions_applied
                ),
                plan_history=tuple(
                    history
                ),
                stopped_reason=(
                    "awaiting_replan_context"
                ),
            )

        supplied = await _resolve_replan_context(
            replan_context_provider,
            request,
            orchestration.state,
        )
        if supplied is None:
            return ReplanControlResult(
                orchestration=orchestration,
                revisions_applied=(
                    revisions_applied
                ),
                plan_history=tuple(
                    history
                ),
                stopped_reason=(
                    "awaiting_replan_context"
                ),
            )

        revised_plan = await replan_task_from_request_async(
            query,
            orchestration.state,
            request,
            replan_context=(
                supplied.planning_context
            ),
            model=replanner_model,
        )
        current_state = (
            begin_replanned_execution_from_request(
                orchestration.state,
                revised_plan,
                request,
            )
        )
        current_plan = revised_plan
        history.append(
            revised_plan
        )
        revisions_applied += 1

        if supplied.execution_context is not None:
            current_context = (
                supplied.execution_context
            )


__all__ = [
    "ReplanContext",
    "ReplanContextProvider",
    "ReplanControlResult",
    "ReplanStopReason",
    "begin_replanned_execution_from_request",
    "build_task_replanner_prompt",
    "exact_reusable_completed_step_ids",
    "exact_reusable_partial_step_ids",
    "replan_task_from_request",
    "replan_task_from_request_async",
    "run_task_plan_with_replanning",
]
