from __future__ import annotations

import asyncio

from src.agent_runtime.execution_resolver import (
    ExecutionSurfaceDescriptor,
)
from src.agent_runtime.intent_complexity import (
    IntentComplexityAssessment,
    IntentGoal,
)
from src.agent_runtime.plan_execution import (
    initialize_plan_execution,
    request_replan,
)
from src.agent_runtime.plan_orchestrator import (
    run_task_plan,
)
from src.agent_runtime.replan_controller import (
    replan_task_from_request_async,
)
from src.agent_runtime.step_execution import (
    StepExecutionContext,
)
from src.agent_runtime.supervisor_contracts import (
    WorkerResult,
)
from src.agent_runtime.task_plan import (
    PlanStep,
    TaskPlan,
)
from src.agent_runtime.task_planner import (
    plan_task_from_assessment_async,
)


class AsyncModel:
    def __init__(self, responder):
        self.responder = responder
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        async def respond():
            value = self.responder(prompt)
            if asyncio.iscoroutine(value):
                return await value
            return value

        return respond()


def _assessment():
    return IntentComplexityAssessment(
        execution_class="multi_graph",
        goals=(
            IntentGoal(
                id="gather",
                objective="Gather source evidence.",
                work_domains=("knowledge",),
                requires_external_state=True,
            ),
            IntentGoal(
                id="combine",
                objective="Combine the evidence.",
                work_domains=("synthesis",),
            ),
        ),
        needs_planning=True,
        missing_information=(),
        clarification_needed=False,
        reasoning_summary="Two dependent semantic goals.",
    )


def _plan_payload():
    return {
        "objective": "Gather and combine evidence.",
        "steps": [
            {
                "id": "gather",
                "objective": "Gather source evidence.",
                "depends_on": [],
                "work_domains": ["knowledge"],
                "requires_external_state": True,
                "requires_mutation": False,
            },
            {
                "id": "combine",
                "objective": "Combine the gathered evidence.",
                "depends_on": ["gather"],
                "work_domains": ["synthesis"],
                "requires_external_state": False,
                "requires_mutation": False,
            },
        ],
        "reasoning_summary": "Gather before combining.",
    }


def test_async_task_planner_accepts_awaitable_model():
    model = AsyncModel(lambda prompt: _plan_payload())

    plan = asyncio.run(
        plan_task_from_assessment_async(
            "Gather and combine evidence.",
            _assessment(),
            model=model,
        )
    )

    assert [step.id for step in plan.steps] == [
        "gather",
        "combine",
    ]
    assert len(model.prompts) == 1


def test_orchestrator_awaits_resolver_and_propagates_dependency_outputs():
    plan = TaskPlan(
        objective="Gather and combine evidence.",
        steps=(
            PlanStep(
                id="gather",
                objective="Gather source evidence.",
            ),
            PlanStep(
                id="combine",
                objective="Combine the gathered evidence.",
                depends_on=("gather",),
            ),
        ),
    )
    surfaces = (
        ExecutionSurfaceDescriptor(
            id="langgraph:gather_cap",
            kind="langgraph",
            description="Gather evidence.",
            capability_name="gather_cap",
        ),
        ExecutionSurfaceDescriptor(
            id="langgraph:combine_cap",
            kind="langgraph",
            description="Combine dependency outputs.",
            capability_name="combine_cap",
        ),
    )

    def resolver_response(prompt):
        surface_id = (
            "langgraph:gather_cap"
            if '"id": "gather"' in prompt
            else "langgraph:combine_cap"
        )
        return {
            "compatible_surface_ids": [surface_id],
            "clarification_needed": False,
            "missing_information": [],
            "reasoning_summary": "Compatible.",
        }

    captured = {}

    def gather_runner(handoff, *, session_id=None):
        return WorkerResult(
            capability="gather_cap",
            status="completed",
            summary="Evidence A",
            data={"items": ["A"]},
        )

    def combine_runner(handoff, *, session_id=None):
        captured.update(handoff.context)
        return WorkerResult(
            capability="combine_cap",
            status="completed",
            summary="Combined A",
        )

    result = asyncio.run(
        run_task_plan(
            plan,
            resolver_model=AsyncModel(
                resolver_response
            ),
            surfaces=surfaces,
            context=StepExecutionContext(
                supervisor_state={
                    "worker_context": {
                        "query": (
                            "Gather and combine evidence."
                        ),
                    }
                },
                session_id="session-1",
            ),
            worker_runners={
                "gather_cap": gather_runner,
                "combine_cap": combine_runner,
            },
        )
    )

    assert result.status == "completed"
    assert result.execution_order == (
        "gather",
        "combine",
    )
    assert captured["plan_step_id"] == "combine"
    dependency = captured[
        "dependency_outputs"
    ]["gather"]
    assert dependency["summary"] == "Evidence A"
    assert dependency["data"] == {
        "items": ["A"],
    }


def test_async_replanner_accepts_awaitable_model():
    plan = TaskPlan(
        objective="Original plan.",
        steps=(
            PlanStep(
                id="first",
                objective="First step.",
            ),
            PlanStep(
                id="second",
                objective="Second step.",
                depends_on=("first",),
            ),
        ),
    )
    state = initialize_plan_execution(plan)
    request = request_replan(
        state,
        reason="Need a narrower scope.",
        trigger_step_id="first",
    )
    revised = _plan_payload()
    revised["objective"] = "Revised plan."

    result = asyncio.run(
        replan_task_from_request_async(
            "Gather and combine evidence.",
            state,
            request,
            replan_context=(
                "Use the project workspace only."
            ),
            model=AsyncModel(
                lambda prompt: revised
            ),
        )
    )

    assert result.objective == "Revised plan."
    assert [step.id for step in result.steps] == [
        "gather",
        "combine",
    ]
