from __future__ import annotations

import asyncio

import pytest

from src.agent_runtime.execution_resolver import (
    ExecutionSurfaceDescriptor,
)
from src.agent_runtime.plan_execution import (
    complete_step,
    initialize_plan_execution,
    request_replan,
    start_step,
    step_execution,
)
from src.agent_runtime.replan_controller import (
    ReplanContext,
    begin_replanned_execution_from_request,
    build_task_replanner_prompt,
    exact_reusable_completed_step_ids,
    replan_task_from_request,
    run_task_plan_with_replanning,
)
from src.agent_runtime.step_execution import (
    StepExecutionContext,
)
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.task_plan import (
    PlanStep,
    TaskPlan,
)


class _SequenceModel:
    def __init__(self, responses):
        self.responses = list(
            responses
        )
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(
            prompt
        )
        if not self.responses:
            raise AssertionError(
                "Unexpected model invocation"
            )
        return self.responses.pop(
            0
        )


class _NeverModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise AssertionError(
            "Model must not be invoked"
        )


def _step(
    step_id,
    *,
    depends_on=(),
    objective=None,
):
    return PlanStep(
        id=step_id,
        objective=(
            objective
            or f"Execute {step_id}."
        ),
        depends_on=depends_on,
        work_domains=("semantic_domain",),
        requires_external_state=True,
        requires_mutation=False,
        requested_sources=("attached_document",),
        excluded_sources=("public_web",),
        retrieval_concepts=(step_id,),
    )


def _plan(*steps):
    return TaskPlan(
        objective="Complete the user request.",
        steps=tuple(steps),
        reasoning_summary="test plan",
    )


def _plan_payload(plan):
    return {
        "objective": plan.objective,
        "steps": [
            {
                "id": step.id,
                "objective": step.objective,
                "depends_on": list(
                    step.depends_on
                ),
                "work_domains": list(
                    step.work_domains
                ),
                "requires_external_state": (
                    step.requires_external_state
                ),
                "requires_mutation": (
                    step.requires_mutation
                ),
                "requested_sources": list(
                    step.requested_sources
                ),
                "excluded_sources": list(
                    step.excluded_sources
                ),
                "retrieval_concepts": list(
                    step.retrieval_concepts
                ),
            }
            for step in plan.steps
        ],
        "reasoning_summary": (
            plan.reasoning_summary
        ),
    }


def _surface(
    capability,
    *,
    kind="langgraph",
    surface_id=None,
):
    return ExecutionSurfaceDescriptor(
        id=(
            surface_id
            or f"langgraph:{capability}"
        ),
        kind=kind,
        description=f"{capability} surface",
        capability_name=(
            capability
            if kind == "langgraph"
            else None
        ),
        supports_external_state=True,
        supports_mutation=False,
    )


def _resolved(surface_id):
    return {
        "compatible_surface_ids": [
            surface_id
        ],
        "clarification_needed": False,
        "missing_information": [],
        "reasoning_summary": "compatible",
    }


def _clarification():
    return {
        "compatible_surface_ids": [],
        "clarification_needed": True,
        "missing_information": [
            "workspace"
        ],
        "reasoning_summary": (
            "Workspace must be clarified."
        ),
    }


def test_exact_completed_steps_are_reused_and_not_rerun_after_replan():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    plan = _plan(
        first,
        second,
    )

    resolver = _SequenceModel(
        [
            _resolved(
                "langgraph:first_cap"
            ),
            _clarification(),
            _resolved(
                "langgraph:second_cap"
            ),
        ]
    )
    replanner = _SequenceModel(
        [
            _plan_payload(
                plan
            )
        ]
    )
    calls = []

    def first_runner(handoff, *, session_id=None):
        calls.append(
            ("first", handoff.context)
        )
        return WorkerResult(
            capability="first_cap",
            status="completed",
            summary="first complete",
        )

    def second_runner(handoff, *, session_id=None):
        calls.append(
            ("second", handoff.context)
        )
        return WorkerResult(
            capability="second_cap",
            status="completed",
            summary="second complete",
        )

    async def provider(request, state):
        assert request.trigger_step_id == "second"
        assert step_execution(
            state,
            "first",
        ).status == "completed"
        return ReplanContext(
            planning_context=(
                "The user clarified the active workspace is Vault."
            ),
            execution_context=StepExecutionContext(
                supervisor_state={
                    "worker_context": {
                        "workspace": "Vault",
                    },
                    "missing_information": [],
                },
                session_id="session-2",
            ),
        )

    result = asyncio.run(
        run_task_plan_with_replanning(
            "Use my notes to complete the task.",
            plan,
            resolver_model=resolver,
            replanner_model=replanner,
            replan_context_provider=provider,
            surfaces=(
                _surface("first_cap"),
                _surface("second_cap"),
            ),
            worker_runners={
                "first_cap": first_runner,
                "second_cap": second_runner,
            },
        )
    )

    assert result.orchestration.status == "completed"
    assert result.revisions_applied == 1
    assert len(
        result.plan_history
    ) == 2
    assert result.orchestration.state.revision == 2
    assert [
        name
        for name, _ in calls
    ] == [
        "first",
        "second",
    ]
    assert calls[1][1]["workspace"] == "Vault"


def test_replan_rejects_source_widening():
    plan = _plan(_step("first"))
    state = initialize_plan_execution(plan)
    request = request_replan(
        state,
        reason="Need revised evidence.",
        trigger_step_id="first",
    )
    payload = _plan_payload(plan)
    payload["steps"][0]["requested_sources"] = ["public_web"]
    payload["steps"][0]["excluded_sources"] = []

    with pytest.raises(
        ValueError,
        match="widens admitted sources: public_web",
    ):
        replan_task_from_request(
            "Use the attached evidence.",
            state,
            request,
            replan_context="Use the same admitted source.",
            model=_SequenceModel([payload]),
        )


def test_changed_completed_step_is_not_reused():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    previous_plan = _plan(
        first,
        second,
    )
    state = initialize_plan_execution(
        previous_plan
    )
    state = start_step(
        state,
        "first",
    )
    state = complete_step(
        state,
        "first",
        output="old evidence",
    )
    request = request_replan(
        state,
        reason="The scope changed.",
        trigger_step_id="second",
    )

    changed_first = _step(
        "first",
        objective="Execute first with the new scope.",
    )
    revised = _plan(
        changed_first,
        _step(
            "second",
            depends_on=("first",),
        ),
    )

    assert exact_reusable_completed_step_ids(
        state,
        revised,
    ) == ()

    new_state = (
        begin_replanned_execution_from_request(
            state,
            revised,
            request,
        )
    )

    assert new_state.revision == 2
    assert step_execution(
        new_state,
        "first",
    ).status == "ready"
    assert step_execution(
        new_state,
        "first",
    ).output is None


def test_unchanged_completed_step_is_reused_exactly():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    plan = _plan(
        first,
        second,
    )
    state = initialize_plan_execution(
        plan
    )
    state = start_step(
        state,
        "first",
    )
    marker = object()
    state = complete_step(
        state,
        "first",
        output=marker,
        artifact_refs=("artifact://one",),
    )
    request = request_replan(
        state,
        reason="New information arrived.",
        trigger_step_id="second",
    )

    revised_state = (
        begin_replanned_execution_from_request(
            state,
            plan,
            request,
        )
    )

    reused = step_execution(
        revised_state,
        "first",
    )
    assert reused.status == "completed"
    assert reused.output is marker
    assert reused.artifact_refs == (
        "artifact://one",
    )
    assert step_execution(
        revised_state,
        "second",
    ).status == "ready"


def test_stale_replan_request_is_rejected():
    first = _step("first")
    plan = _plan(first)
    state = initialize_plan_execution(
        plan
    )
    request = request_replan(
        state,
        reason="Need context.",
        trigger_step_id="first",
    )
    newer = initialize_plan_execution(
        plan,
        revision=2,
    )

    with pytest.raises(
        ValueError,
        match="revision does not match",
    ):
        build_task_replanner_prompt(
            "Do the task.",
            newer,
            request,
            replan_context="New context.",
        )


def test_missing_replan_context_stops_without_invoking_replanner():
    step = _step("first")
    plan = _plan(step)
    resolver = _SequenceModel(
        [
            _clarification()
        ]
    )
    replanner = _NeverModel()

    result = asyncio.run(
        run_task_plan_with_replanning(
            "Do the task.",
            plan,
            resolver_model=resolver,
            replanner_model=replanner,
            replan_context_provider=None,
            surfaces=(
                _surface("first_cap"),
            ),
            worker_runners={},
        )
    )

    assert result.orchestration.status == "needs_replan"
    assert (
        result.stopped_reason
        == "awaiting_replan_context"
    )
    assert result.revisions_applied == 0
    assert replanner.calls == 0


def test_provider_can_decline_context_without_model_replan():
    step = _step("first")
    plan = _plan(step)
    resolver = _SequenceModel(
        [
            _clarification()
        ]
    )
    replanner = _NeverModel()

    async def provider(request, state):
        return None

    result = asyncio.run(
        run_task_plan_with_replanning(
            "Do the task.",
            plan,
            resolver_model=resolver,
            replanner_model=replanner,
            replan_context_provider=provider,
            surfaces=(
                _surface("first_cap"),
            ),
            worker_runners={},
        )
    )

    assert (
        result.stopped_reason
        == "awaiting_replan_context"
    )
    assert replanner.calls == 0


def test_revision_limit_prevents_unbounded_replan_loop():
    step = _step("first")
    plan = _plan(step)
    resolver = _SequenceModel(
        [
            _clarification()
        ]
    )
    replanner = _NeverModel()
    provider_calls = []

    def provider(request, state):
        provider_calls.append(
            request
        )
        return "new context"

    result = asyncio.run(
        run_task_plan_with_replanning(
            "Do the task.",
            plan,
            resolver_model=resolver,
            replanner_model=replanner,
            replan_context_provider=provider,
            surfaces=(
                _surface("first_cap"),
            ),
            worker_runners={},
            max_revisions=0,
        )
    )

    assert result.stopped_reason == "revision_limit"
    assert result.revisions_applied == 0
    assert provider_calls == []
    assert replanner.calls == 0


def test_execution_unavailable_is_not_silently_semantically_replanned():
    step = _step("legacy")
    plan = _plan(step)
    legacy = _surface(
        "ignored",
        kind="legacy",
        surface_id="legacy:agent_loop",
    )
    resolver = _SequenceModel(
        [
            _resolved(
                "legacy:agent_loop"
            )
        ]
    )
    replanner = _NeverModel()
    provider_calls = []

    def provider(request, state):
        provider_calls.append(
            request
        )
        return "try another plan"

    result = asyncio.run(
        run_task_plan_with_replanning(
            "Do the task.",
            plan,
            resolver_model=resolver,
            replanner_model=replanner,
            replan_context_provider=provider,
            surfaces=(legacy,),
            worker_runners={},
            surface_executors={},
        )
    )

    assert (
        result.orchestration.status
        == "execution_unavailable"
    )
    assert result.stopped_reason == "terminal"
    assert result.revisions_applied == 0
    assert provider_calls == []
    assert replanner.calls == 0


def test_replanner_prompt_is_semantic_and_contains_execution_outcomes():
    first = _step("first")
    second = _step(
        "second",
        depends_on=("first",),
    )
    plan = _plan(
        first,
        second,
    )
    state = initialize_plan_execution(
        plan
    )
    state = start_step(
        state,
        "first",
    )
    state = complete_step(
        state,
        "first",
        output=WorkerResult(
            capability="knowledge_search",
            status="completed",
            summary="Found the relevant note.",
        ),
    )
    request = request_replan(
        state,
        reason="Need a clarified scope.",
        trigger_step_id="second",
    )

    prompt = build_task_replanner_prompt(
        "Use my notes.",
        state,
        request,
        replan_context="Use the Vault workspace.",
    )

    assert "Found the relevant note." in prompt
    assert "Use the Vault workspace." in prompt
    assert (
        "Do NOT select workers, tools, capabilities"
        in prompt
    )
    assert "compatible_surface_ids" not in prompt
    assert "selected_surface" not in prompt


def test_replan_controller_does_not_own_routes_or_final_synthesis():
    import src.agent_runtime.replan_controller as module

    text = open(
        module.__file__,
        "r",
        encoding="utf-8",
    ).read()

    forbidden = (
        "chat_routes",
        "stream_agent_loop(",
        "dispatch_supervisor_decision(",
        "SupervisorDecision(",
        "final_user_narrative",
        "synthesize(",
        "execute_tool_block(",
    )

    for marker in forbidden:
        assert marker not in text
