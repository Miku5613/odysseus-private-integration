from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from dataclasses import dataclass
from typing import Any

from src.agent_runtime.code_worker import run_code_worker
from src.agent_runtime.knowledge_worker import run_knowledge_worker
from src.agent_runtime.research_worker import run_research_worker
from src.agent_runtime.mcp_worker import run_mcp_worker
from src.agent_runtime.shared_tool_worker import run_web_search_worker
from src.agent_runtime.supervisor_contracts import (
    SupervisorState,
    WorkerHandoff,
    WorkerResult,
)
from src.agent_runtime.supervisor_decision import SupervisorDecision


WorkerRunner = Callable[..., WorkerResult]


@dataclass(frozen=True)
class DispatchResult:
    """Normalized result of one Supervisor orchestration dispatch."""

    requested_capabilities: tuple[str, ...]
    completed_capabilities: tuple[str, ...]
    worker_results: tuple[WorkerResult, ...]
    undispatched_capabilities: tuple[str, ...] = ()


def builtin_worker_runners(
    *,
    code_finalizer_candidates: Sequence[tuple] | None = None,
    code_finalizer_prompt_type: str | None = None,
) -> dict[str, WorkerRunner]:
    """Return configured worker backends currently implemented by the runtime."""

    code_runner: WorkerRunner = run_code_worker
    if code_finalizer_candidates:
        code_runner = partial(
            run_code_worker,
            finalizer_candidates=tuple(
                code_finalizer_candidates
            ),
            finalizer_prompt_type=code_finalizer_prompt_type,
        )

    return {
        "code_inspection": code_runner,
        "knowledge_search": run_knowledge_worker,
        "deep_research": run_research_worker,
        "mcp_specialized": run_mcp_worker,
        "web_search": run_web_search_worker,
    }


def build_worker_handoff(
    capability: str,
    objective: str,
    *,
    supervisor_state: SupervisorState | None = None,
) -> WorkerHandoff:
    """Translate general Supervisor state into a domain-neutral handoff."""

    state = supervisor_state or {}

    context = dict(state.get("worker_context", {}))
    capability_actions = dict(state.get("capability_actions", {}) or {})
    selected_action = capability_actions.get(capability)

    if selected_action:
        context["action"] = selected_action

    return WorkerHandoff(
        capability=capability,
        objective=objective,
        context=context,
        constraints={
            "missing_information": list(
                state.get("missing_information", [])
            ),
        },
    )


def dispatch_supervisor_decision(
    decision: SupervisorDecision,
    objective: str,
    *,
    supervisor_state: SupervisorState | None = None,
    worker_runners: dict[str, WorkerRunner] | None = None,
    session_id: str | None = None,
) -> DispatchResult:
    """Dispatch selected worker capabilities through one common boundary.

    This function performs no capability selection itself. It consumes an
    already validated SupervisorDecision and dispatches only implemented
    worker capabilities. Unknown/unimplemented selections are returned as
    undispatched rather than silently rerouted.
    """

    runners = (
        builtin_worker_runners()
        if worker_runners is None
        else worker_runners
    )

    results: list[WorkerResult] = []
    completed: list[str] = []
    undispatched: list[str] = []

    for capability in decision.selected_capabilities:
        runner = runners.get(capability)

        if runner is None:
            undispatched.append(capability)
            continue

        handoff = build_worker_handoff(
            capability,
            objective,
            supervisor_state=supervisor_state,
        )

        result = runner(
            handoff,
            session_id=session_id,
        )

        results.append(result)

        if result.status == "completed":
            completed.append(capability)

    return DispatchResult(
        requested_capabilities=decision.selected_capabilities,
        completed_capabilities=tuple(completed),
        worker_results=tuple(results),
        undispatched_capabilities=tuple(undispatched),
    )


async def dispatch_supervisor_decision_async(
    decision: SupervisorDecision,
    objective: str,
    *,
    supervisor_state: SupervisorState | None = None,
    worker_runners: dict[str, WorkerRunner] | None = None,
    session_id: str | None = None,
) -> DispatchResult:
    """Dispatch both synchronous and asynchronous workers safely."""

    import inspect

    runners = (
        builtin_worker_runners()
        if worker_runners is None
        else worker_runners
    )

    results: list[WorkerResult] = []
    completed: list[str] = []
    undispatched: list[str] = []

    for capability in decision.selected_capabilities:
        runner = runners.get(capability)

        if runner is None:
            undispatched.append(capability)
            continue

        handoff = build_worker_handoff(
            capability,
            objective,
            supervisor_state=supervisor_state,
        )

        value = runner(
            handoff,
            session_id=session_id,
        )

        result = await value if inspect.isawaitable(value) else value
        results.append(result)

        if result.status == "completed":
            completed.append(capability)

    return DispatchResult(
        requested_capabilities=decision.selected_capabilities,
        completed_capabilities=tuple(completed),
        worker_results=tuple(results),
        undispatched_capabilities=tuple(undispatched),
    )


__all__ = [
    "DispatchResult",
    "build_worker_handoff",
    "builtin_worker_runners",
    "dispatch_supervisor_decision",
    "dispatch_supervisor_decision_async",
]
