from __future__ import annotations

import asyncio

import pytest

from src.agent_runtime.execution_resolver import (
    ExecutionResolution,
    ExecutionSurfaceDescriptor,
)
from src.agent_runtime.plan_execution import (
    initialize_plan_execution,
    start_step,
    step_execution,
)
from src.agent_runtime.step_execution import (
    StepExecutionContext,
    StepExecutionUnavailable,
    apply_worker_result_to_plan_state,
    execute_and_record_resolved_step,
    execute_resolved_step,
)
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.task_plan import PlanStep, TaskPlan


def _step(
    step_id="step_1",
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


def _surface(
    *,
    surface_id="langgraph:knowledge_search",
    kind="langgraph",
    capability_name="knowledge_search",
):
    return ExecutionSurfaceDescriptor(
        id=surface_id,
        kind=kind,
        description=f"{surface_id} description",
        capability_name=capability_name,
        supports_external_state=True,
        supports_mutation=False,
    )


def _resolution(
    step,
    *,
    surface=None,
    status="resolved",
):
    return ExecutionResolution(
        step_id=step.id,
        status=status,
        selected_surface=surface,
    )


def _running_state(*steps):
    plan = TaskPlan(
        objective="Execute test plan.",
        steps=tuple(steps),
    )
    state = initialize_plan_execution(plan)
    return start_step(state, steps[0].id)


def test_nonresolved_resolution_is_rejected_before_execution():
    step = _step()
    resolution = _resolution(
        step,
        surface=None,
        status="clarification",
    )

    with pytest.raises(
        ValueError,
        match="Only resolved PlanSteps",
    ):
        asyncio.run(
            execute_resolved_step(
                step,
                resolution,
                worker_runners={},
            )
        )


def test_resolution_step_id_must_match_plan_step():
    step = _step()
    resolution = ExecutionResolution(
        step_id="different_step",
        status="resolved",
        selected_surface=_surface(),
    )

    with pytest.raises(
        ValueError,
        match="does not match PlanStep",
    ):
        asyncio.run(
            execute_resolved_step(
                step,
                resolution,
                worker_runners={},
            )
        )


def test_native_sync_runner_reuses_handoff_and_propagates_context():
    step = _step()
    surface = _surface(
        surface_id="langgraph:native_test",
        capability_name="native_test",
    )
    resolution = _resolution(
        step,
        surface=surface,
    )
    captured = {}

    def runner(handoff, *, session_id=None):
        captured["handoff"] = handoff
        captured["session_id"] = session_id
        return WorkerResult(
            capability="native_test",
            status="completed",
            summary="done",
        )

    result = asyncio.run(
        execute_resolved_step(
            step,
            resolution,
            context=StepExecutionContext(
                supervisor_state={
                    "worker_context": {
                        "workspace": "Vault",
                        "tool_policy": {"mode": "read"},
                    },
                    "capability_actions": {
                        "native_test": "retrieve",
                    },
                    "missing_information": [],
                },
                session_id="session-1",
            ),
            worker_runners={
                "native_test": runner,
            },
        )
    )

    assert result.status == "completed"
    assert captured["session_id"] == "session-1"
    assert captured["handoff"].capability == "native_test"
    assert captured["handoff"].objective == step.objective
    assert captured["handoff"].context["workspace"] == "Vault"
    assert captured["handoff"].context["tool_policy"] == {
        "mode": "read"
    }
    assert captured["handoff"].context["action"] == "retrieve"


def test_native_async_runner_is_awaited():
    step = _step()
    surface = _surface(
        surface_id="langgraph:async_test",
        capability_name="async_test",
    )

    async def runner(handoff, *, session_id=None):
        return WorkerResult(
            capability=handoff.capability,
            status="completed",
            summary="async done",
        )

    result = asyncio.run(
        execute_resolved_step(
            step,
            _resolution(
                step,
                surface=surface,
            ),
            worker_runners={
                "async_test": runner,
            },
        )
    )

    assert result.summary == "async done"


def test_missing_native_runner_is_explicitly_unavailable():
    step = _step()
    surface = _surface(
        surface_id="langgraph:missing",
        capability_name="missing",
    )

    with pytest.raises(
        StepExecutionUnavailable,
        match="No implemented native worker runner",
    ):
        asyncio.run(
            execute_resolved_step(
                step,
                _resolution(
                    step,
                    surface=surface,
                ),
                worker_runners={},
            )
        )


def test_executor_must_return_worker_result():
    step = _step()
    surface = _surface(
        surface_id="langgraph:bad",
        capability_name="bad",
    )

    def runner(handoff, *, session_id=None):
        return {"status": "completed"}

    with pytest.raises(
        TypeError,
        match="must return WorkerResult",
    ):
        asyncio.run(
            execute_resolved_step(
                step,
                _resolution(
                    step,
                    surface=surface,
                ),
                worker_runners={
                    "bad": runner,
                },
            )
        )


def test_native_worker_result_capability_must_match_resolution():
    step = _step()
    surface = _surface(
        surface_id="langgraph:expected",
        capability_name="expected",
    )

    def runner(handoff, *, session_id=None):
        return WorkerResult(
            capability="different",
            status="completed",
            summary="wrong",
        )

    with pytest.raises(
        ValueError,
        match="does not match resolved capability",
    ):
        asyncio.run(
            execute_resolved_step(
                step,
                _resolution(
                    step,
                    surface=surface,
                ),
                worker_runners={
                    "expected": runner,
                },
            )
        )


def test_unbound_legacy_surface_is_not_sse_parsed_or_silently_rerouted():
    step = _step()
    surface = _surface(
        surface_id="legacy:agent_loop",
        kind="legacy",
        capability_name=None,
    )

    with pytest.raises(
        StepExecutionUnavailable,
        match="no structured step executor",
    ):
        asyncio.run(
            execute_resolved_step(
                step,
                _resolution(
                    step,
                    surface=surface,
                ),
                worker_runners={
                    "knowledge_search": lambda *a, **k: None,
                },
            )
        )


def test_registered_surface_can_use_explicit_structured_executor():
    step = _step()
    surface = _surface(
        surface_id="registered:external",
        kind="registered",
        capability_name=None,
    )
    captured = {}

    async def executor(exec_step, resolution, context):
        captured["step"] = exec_step
        captured["resolution"] = resolution
        captured["session_id"] = context.session_id
        return WorkerResult(
            capability="external",
            status="completed",
            summary="external done",
        )

    result = asyncio.run(
        execute_resolved_step(
            step,
            _resolution(
                step,
                surface=surface,
            ),
            context=StepExecutionContext(
                session_id="session-2",
            ),
            surface_executors={
                "registered:external": executor,
            },
        )
    )

    assert result.summary == "external done"
    assert captured["step"] == step
    assert captured["session_id"] == "session-2"


def test_completed_worker_result_completes_running_step_and_unlocks_dependency():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    state = _running_state(
        first,
        second,
    )
    result = WorkerResult(
        capability="knowledge_search",
        status="completed",
        summary="evidence found",
        data={"value": 42},
    )

    outcome = apply_worker_result_to_plan_state(
        state,
        "first",
        result,
    )

    first_state = step_execution(
        outcome.state,
        "first",
    )
    second_state = step_execution(
        outcome.state,
        "second",
    )

    assert first_state.status == "completed"
    assert first_state.output is result
    assert second_state.status == "ready"
    assert outcome.replan_request is None


def test_failed_worker_result_fails_step_and_blocks_descendants():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    state = _running_state(
        first,
        second,
    )
    result = WorkerResult(
        capability="knowledge_search",
        status="failed",
        summary="retrieval failed",
        error="backend unavailable",
    )

    outcome = apply_worker_result_to_plan_state(
        state,
        "first",
        result,
    )

    assert step_execution(
        outcome.state,
        "first",
    ).status == "failed"
    assert step_execution(
        outcome.state,
        "first",
    ).error == "backend unavailable"
    assert step_execution(
        outcome.state,
        "second",
    ).status == "blocked"


def test_clarification_preserves_state_and_emits_replan_request():
    first = _step("first")
    state = _running_state(first)
    result = WorkerResult(
        capability="knowledge_search",
        status="needs_clarification",
        summary="Need a source scope.",
        missing_information=(
            "workspace",
            "date range",
        ),
    )

    outcome = apply_worker_result_to_plan_state(
        state,
        "first",
        result,
    )

    assert outcome.state is state
    assert step_execution(
        outcome.state,
        "first",
    ).status == "running"
    assert outcome.replan_request is not None
    assert outcome.replan_request.trigger_step_id == "first"
    assert "workspace" in outcome.replan_request.reason
    assert "date range" in outcome.replan_request.reason


def test_execute_and_record_composes_native_execution_with_plan_state():
    step = _step()
    state = _running_state(step)
    surface = _surface(
        surface_id="langgraph:test",
        capability_name="test",
    )

    def runner(handoff, *, session_id=None):
        return WorkerResult(
            capability="test",
            status="completed",
            summary="composed",
        )

    outcome = asyncio.run(
        execute_and_record_resolved_step(
            state,
            step,
            _resolution(
                step,
                surface=surface,
            ),
            worker_runners={
                "test": runner,
            },
        )
    )

    assert outcome.worker_result.summary == "composed"
    assert step_execution(
        outcome.state,
        step.id,
    ).status == "completed"


def test_recording_requires_running_step_lifecycle_owned_by_orchestrator():
    step = _step()
    plan = TaskPlan(
        objective="Test.",
        steps=(step,),
    )
    state = initialize_plan_execution(plan)
    result = WorkerResult(
        capability="test",
        status="completed",
        summary="done",
    )

    with pytest.raises(
        ValueError,
        match="cannot complete from status ready",
    ):
        apply_worker_result_to_plan_state(
            state,
            step.id,
            result,
        )


def test_unknown_worker_status_is_not_collapsed_to_failure():
    step = _step()
    state = _running_state(step)
    result = WorkerResult(
        capability="test",
        status="future_status",
        summary="future",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported WorkerResult status",
    ):
        apply_worker_result_to_plan_state(
            state,
            step.id,
            result,
        )


def test_module_does_not_duplicate_or_bypass_existing_execution_owners():
    import src.agent_runtime.step_execution as module

    text = open(
        module.__file__,
        "r",
        encoding="utf-8",
    ).read()

    forbidden = (
        "class WorkerHandoff",
        "class WorkerResult",
        "dispatch_supervisor_decision(",
        "dispatch_supervisor_decision_async(",
        "stream_agent_loop(",
        "execute_tool_block(",
    )

    for marker in forbidden:
        assert marker not in text
