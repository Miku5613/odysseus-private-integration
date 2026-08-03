from __future__ import annotations

import asyncio

import pytest

from src.agent_runtime.execution_resolver import (
    ExecutionSurfaceDescriptor,
)
from src.agent_runtime.plan_execution import (
    initialize_plan_execution,
    start_step,
    step_execution,
)
from src.agent_runtime.plan_orchestrator import (
    run_task_plan,
)
from src.agent_runtime.step_execution import (
    StepExecutionContext,
)
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.task_plan import PlanStep, TaskPlan


class _ResolverModel:
    def __init__(self, mapping):
        self.mapping = mapping
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        for needle, payload in self.mapping:
            if needle in prompt:
                return payload
        raise AssertionError(
            "No resolver payload matched prompt"
        )


def _step(
    step_id,
    *,
    depends_on=(),
):
    return PlanStep(
        id=step_id,
        objective=f"Execute {step_id}.",
        depends_on=depends_on,
        work_domains=("semantic_domain",),
        requires_external_state=True,
        requires_mutation=False,
    )


def _plan(*steps):
    return TaskPlan(
        objective="Execute orchestration test.",
        steps=tuple(steps),
    )


def _surface(
    capability,
    *,
    kind="langgraph",
    surface_id=None,
):
    return ExecutionSurfaceDescriptor(
        id=surface_id or f"langgraph:{capability}",
        kind=kind,
        description=f"{capability} execution surface.",
        capability_name=(
            capability
            if kind == "langgraph"
            else None
        ),
        supports_external_state=True,
        supports_mutation=False,
    )


def _resolved_payload(surface_id):
    return {
        "compatible_surface_ids": [
            surface_id
        ],
        "clarification_needed": False,
        "missing_information": [],
        "reasoning_summary": "compatible",
    }


def _runner(capability, calls, *, fail=False, clarify=False, partial=False):
    def run(handoff, *, session_id=None):
        calls.append(
            (
                capability,
                handoff.objective,
                session_id,
            )
        )

        if fail:
            return WorkerResult(
                capability=capability,
                status="failed",
                summary=f"{capability} failed",
                error=f"{capability} error",
            )

        if clarify:
            return WorkerResult(
                capability=capability,
                status="needs_clarification",
                summary="Need more scope.",
                missing_information=("scope",),
            )

        if partial:
            return WorkerResult(
                capability=capability,
                status="partial",
                summary=f"{capability} partial evidence",
                data={"artifact": f"{capability}-partial"},
                missing_information=("uncovered evidence",),
            )

        return WorkerResult(
            capability=capability,
            status="completed",
            summary=f"{capability} complete",
            data={"capability": capability},
        )

    return run


def test_single_step_executes_to_completion():
    step = _step("first")
    surface = _surface("cap_first")
    calls = []

    result = asyncio.run(
        run_task_plan(
            _plan(step),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute first.",
                        _resolved_payload(
                            surface.id
                        ),
                    )
                ]
            ),
            surfaces=(surface,),
            worker_runners={
                "cap_first": _runner(
                    "cap_first",
                    calls,
                )
            },
        )
    )

    assert result.status == "completed"
    assert result.execution_order == ("first",)
    assert step_execution(
        result.state,
        "first",
    ).status == "completed"
    assert len(result.worker_results) == 1


def test_dependency_chain_runs_in_order_and_unlocks_steps():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    third = _step(
        "third",
        depends_on=("second",),
    )
    surfaces = (
        _surface("cap_first"),
        _surface("cap_second"),
        _surface("cap_third"),
    )
    calls = []

    result = asyncio.run(
        run_task_plan(
            _plan(
                first,
                second,
                third,
            ),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute first.",
                        _resolved_payload(
                            "langgraph:cap_first"
                        ),
                    ),
                    (
                        "Execute second.",
                        _resolved_payload(
                            "langgraph:cap_second"
                        ),
                    ),
                    (
                        "Execute third.",
                        _resolved_payload(
                            "langgraph:cap_third"
                        ),
                    ),
                ]
            ),
            surfaces=surfaces,
            worker_runners={
                "cap_first": _runner(
                    "cap_first",
                    calls,
                ),
                "cap_second": _runner(
                    "cap_second",
                    calls,
                ),
                "cap_third": _runner(
                    "cap_third",
                    calls,
                ),
            },
        )
    )

    assert result.status == "completed"
    assert result.execution_order == (
        "first",
        "second",
        "third",
    )
    assert [
        item[0]
        for item in calls
    ] == [
        "cap_first",
        "cap_second",
        "cap_third",
    ]


def test_parallel_ready_frontier_executes_in_stable_plan_order():
    first = _step("first")
    second = _step("second")
    join = _step(
        "join",
        depends_on=(
            "first",
            "second",
        ),
    )
    calls = []

    result = asyncio.run(
        run_task_plan(
            _plan(
                first,
                second,
                join,
            ),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute first.",
                        _resolved_payload(
                            "langgraph:first_cap"
                        ),
                    ),
                    (
                        "Execute second.",
                        _resolved_payload(
                            "langgraph:second_cap"
                        ),
                    ),
                    (
                        "Execute join.",
                        _resolved_payload(
                            "langgraph:join_cap"
                        ),
                    ),
                ]
            ),
            surfaces=(
                _surface("first_cap"),
                _surface("second_cap"),
                _surface("join_cap"),
            ),
            worker_runners={
                "first_cap": _runner(
                    "first_cap",
                    calls,
                ),
                "second_cap": _runner(
                    "second_cap",
                    calls,
                ),
                "join_cap": _runner(
                    "join_cap",
                    calls,
                ),
            },
        )
    )

    assert result.execution_order == (
        "first",
        "second",
        "join",
    )
    assert result.status == "completed"


def test_actual_failure_blocks_descendants_but_independent_ready_work_continues():
    failing = _step("failing")
    dependent = _step(
        "dependent",
        depends_on=("failing",),
    )
    independent = _step("independent")
    calls = []

    result = asyncio.run(
        run_task_plan(
            _plan(
                failing,
                dependent,
                independent,
            ),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute failing.",
                        _resolved_payload(
                            "langgraph:fail_cap"
                        ),
                    ),
                    (
                        "Execute independent.",
                        _resolved_payload(
                            "langgraph:independent_cap"
                        ),
                    ),
                ]
            ),
            surfaces=(
                _surface("fail_cap"),
                _surface("independent_cap"),
            ),
            worker_runners={
                "fail_cap": _runner(
                    "fail_cap",
                    calls,
                    fail=True,
                ),
                "independent_cap": _runner(
                    "independent_cap",
                    calls,
                ),
            },
        )
    )

    assert result.status == "failed"
    assert result.execution_order == (
        "failing",
        "independent",
    )
    assert step_execution(
        result.state,
        "failing",
    ).status == "failed"
    assert step_execution(
        result.state,
        "dependent",
    ).status == "blocked"
    assert step_execution(
        result.state,
        "independent",
    ).status == "completed"


def test_resolution_clarification_stops_before_starting_step():
    step = _step("first")
    surface = _surface("cap_first")

    result = asyncio.run(
        run_task_plan(
            _plan(step),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute first.",
                        {
                            "compatible_surface_ids": [
                                surface.id
                            ],
                            "clarification_needed": True,
                            "missing_information": [
                                "workspace"
                            ],
                            "reasoning_summary": (
                                "Workspace is ambiguous."
                            ),
                        },
                    )
                ]
            ),
            surfaces=(surface,),
            worker_runners={},
        )
    )

    assert result.status == "needs_replan"
    assert result.execution_order == ()
    assert step_execution(
        result.state,
        "first",
    ).status == "ready"
    assert result.replan_request is not None
    assert (
        result.replan_request.trigger_step_id
        == "first"
    )


def test_worker_clarification_stops_with_replan_request_not_failure():
    step = _step("first")
    surface = _surface("cap_first")

    result = asyncio.run(
        run_task_plan(
            _plan(step),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute first.",
                        _resolved_payload(
                            surface.id
                        ),
                    )
                ]
            ),
            surfaces=(surface,),
            worker_runners={
                "cap_first": _runner(
                    "cap_first",
                    [],
                    clarify=True,
                )
            },
        )
    )

    assert result.status == "needs_replan"
    assert step_execution(
        result.state,
        "first",
    ).status == "running"
    assert result.replan_request is not None
    assert "scope" in result.replan_request.reason


def test_unsupported_resolution_is_real_step_failure_but_independent_work_continues():
    unsupported = _step("unsupported")
    independent = _step("independent")
    surface = _surface("independent_cap")

    result = asyncio.run(
        run_task_plan(
            _plan(
                unsupported,
                independent,
            ),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute unsupported.",
                        {
                            "compatible_surface_ids": [],
                            "clarification_needed": False,
                            "missing_information": [],
                            "reasoning_summary": (
                                "No known execution surface can perform this step."
                            ),
                        },
                    ),
                    (
                        "Execute independent.",
                        _resolved_payload(
                            surface.id
                        ),
                    ),
                ]
            ),
            surfaces=(surface,),
            worker_runners={
                "independent_cap": _runner(
                    "independent_cap",
                    [],
                )
            },
        )
    )

    assert result.status == "failed"
    assert result.execution_order == (
        "unsupported",
        "independent",
    )
    assert step_execution(
        result.state,
        "unsupported",
    ).status == "failed"
    assert step_execution(
        result.state,
        "independent",
    ).status == "completed"


def test_partial_step_preserves_evidence_and_independent_steps_continue():
    partial = _step("partial")
    independent_a = _step("independent_a")
    independent_b = _step("independent_b")
    dependent = _step("dependent", depends_on=("partial",))
    surfaces = (
        _surface("partial_cap"),
        _surface("independent_a_cap"),
        _surface("independent_b_cap"),
    )
    calls = []

    result = asyncio.run(
        run_task_plan(
            _plan(partial, independent_a, independent_b, dependent),
            resolver_model=_ResolverModel([
                ("Execute partial.", _resolved_payload(surfaces[0].id)),
                ("Execute independent_a.", _resolved_payload(surfaces[1].id)),
                ("Execute independent_b.", _resolved_payload(surfaces[2].id)),
            ]),
            surfaces=surfaces,
            worker_runners={
                "partial_cap": _runner("partial_cap", calls, partial=True),
                "independent_a_cap": _runner("independent_a_cap", calls),
                "independent_b_cap": _runner("independent_b_cap", calls),
            },
        )
    )

    assert result.status == "partial"
    assert result.execution_order == (
        "partial",
        "independent_a",
        "independent_b",
    )
    partial_state = step_execution(result.state, "partial")
    assert partial_state.status == "partial"
    assert partial_state.output.data["artifact"] == "partial_cap-partial"
    assert step_execution(result.state, "independent_a").status == "completed"
    assert step_execution(result.state, "independent_b").status == "completed"
    assert step_execution(result.state, "dependent").status == "blocked"
    assert result.replan_request is None


def test_unbound_legacy_execution_is_not_mislabeled_as_semantic_failure():
    step = _step("legacy_step")
    legacy = _surface(
        "ignored",
        kind="legacy",
        surface_id="legacy:agent_loop",
    )

    result = asyncio.run(
        run_task_plan(
            _plan(step),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute legacy_step.",
                        _resolved_payload(
                            "legacy:agent_loop"
                        ),
                    )
                ]
            ),
            surfaces=(legacy,),
            worker_runners={},
            surface_executors={},
        )
    )

    assert result.status == "execution_unavailable"
    assert result.execution_order == ()
    assert step_execution(
        result.state,
        "legacy_step",
    ).status == "ready"
    assert result.replan_request is not None


def test_explicit_registered_executor_participates_in_plan_execution():
    step = _step("external")
    surface = _surface(
        "ignored",
        kind="registered",
        surface_id="registered:external",
    )

    async def executor(step, resolution, context):
        return WorkerResult(
            capability="external",
            status="completed",
            summary="external complete",
        )

    result = asyncio.run(
        run_task_plan(
            _plan(step),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute external.",
                        _resolved_payload(
                            "registered:external"
                        ),
                    )
                ]
            ),
            surfaces=(surface,),
            surface_executors={
                "registered:external": executor,
            },
        )
    )

    assert result.status == "completed"
    assert result.execution_order == (
        "external",
    )


def test_context_and_session_propagate_through_orchestrator_to_worker():
    step = _step("first")
    surface = _surface("cap_first")
    captured = {}

    def runner(handoff, *, session_id=None):
        captured["context"] = handoff.context
        captured["session_id"] = session_id
        return WorkerResult(
            capability="cap_first",
            status="completed",
            summary="done",
        )

    result = asyncio.run(
        run_task_plan(
            _plan(step),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute first.",
                        _resolved_payload(
                            surface.id
                        ),
                    )
                ]
            ),
            surfaces=(surface,),
            context=StepExecutionContext(
                supervisor_state={
                    "worker_context": {
                        "workspace": "Vault",
                    },
                    "missing_information": [],
                },
                session_id="session-42",
            ),
            worker_runners={
                "cap_first": runner,
            },
        )
    )

    assert result.status == "completed"
    assert captured["context"]["workspace"] == "Vault"
    assert captured["session_id"] == "session-42"


def test_existing_state_can_resume_ready_work():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    plan = _plan(
        first,
        second,
    )
    initial = initialize_plan_execution(
        plan
    )
    running = start_step(
        initial,
        "first",
    )

    # A state with an already-running step is intentionally not auto-replayed.
    stalled = asyncio.run(
        run_task_plan(
            plan,
            resolver_model=_ResolverModel([]),
            state=running,
            surfaces=(),
            worker_runners={},
        )
    )
    assert stalled.status == "stalled"
    assert "first" in stalled.message


def test_state_from_different_plan_is_rejected():
    first = _step("first")
    other = _step("other")

    state = initialize_plan_execution(
        _plan(first)
    )

    with pytest.raises(
        ValueError,
        match="different TaskPlan",
    ):
        asyncio.run(
            run_task_plan(
                _plan(other),
                resolver_model=_ResolverModel([]),
                state=state,
                surfaces=(),
            )
        )


def test_max_step_execution_guard_is_product_safety_boundary():
    first = _step("first")
    second = _step("second")

    result = asyncio.run(
        run_task_plan(
            _plan(
                first,
                second,
            ),
            resolver_model=_ResolverModel(
                [
                    (
                        "Execute first.",
                        _resolved_payload(
                            "langgraph:first_cap"
                        ),
                    ),
                    (
                        "Execute second.",
                        _resolved_payload(
                            "langgraph:second_cap"
                        ),
                    ),
                ]
            ),
            surfaces=(
                _surface("first_cap"),
                _surface("second_cap"),
            ),
            worker_runners={
                "first_cap": _runner(
                    "first_cap",
                    [],
                ),
                "second_cap": _runner(
                    "second_cap",
                    [],
                ),
            },
            max_step_executions=1,
        )
    )

    assert result.status == "stalled"
    assert result.execution_order == (
        "first",
    )
    assert step_execution(
        result.state,
        "first",
    ).status == "completed"
    assert step_execution(
        result.state,
        "second",
    ).status == "ready"


def test_orchestrator_does_not_own_planning_backend_selection_or_synthesis():
    import src.agent_runtime.plan_orchestrator as module

    text = open(
        module.__file__,
        "r",
        encoding="utf-8",
    ).read()

    forbidden = (
        "TaskPlannerModel",
        "plan_task(",
        "stream_agent_loop(",
        "dispatch_supervisor_decision(",
        "SupervisorDecision(",
        "final_user_narrative",
        "synthesize",
    )

    for marker in forbidden:
        assert marker not in text
