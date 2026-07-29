from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.agent_runtime.intent_complexity import (
    IntentComplexityAssessment,
    IntentGoal,
)
from src.agent_runtime.task_planner import (
    build_task_planner_prompt,
    parse_task_plan_response,
    plan_task_from_assessment,
)


def _multi_assessment():
    return IntentComplexityAssessment(
        execution_class="multi_graph",
        goals=(
            IntentGoal(
                id="goal_1",
                objective=(
                    "Compare project state with current research."
                ),
                work_domains=(
                    "user_project_state",
                    "current_public_research",
                ),
                requires_external_state=True,
                requires_mutation=False,
            ),
        ),
        needs_planning=True,
        missing_information=(),
        clarification_needed=False,
        reasoning_summary=(
            "Two independent acquisition surfaces feed one result."
        ),
    )


def _conversation_assessment():
    return IntentComplexityAssessment(
        execution_class="conversation",
        goals=(),
        needs_planning=False,
        missing_information=(),
        clarification_needed=False,
        reasoning_summary="No execution required.",
    )


def _valid_plan_payload():
    return {
        "objective": (
            "Compare project state with current research."
        ),
        "steps": [
            {
                "id": "project_state",
                "objective": "Retrieve relevant project state.",
                "depends_on": [],
                "work_domains": ["user_project_state"],
                "requires_external_state": True,
                "requires_mutation": False,
            },
            {
                "id": "current_research",
                "objective": "Acquire current external research.",
                "depends_on": [],
                "work_domains": ["current_public_research"],
                "requires_external_state": True,
                "requires_mutation": False,
            },
            {
                "id": "synthesize",
                "objective": "Combine the acquired evidence.",
                "depends_on": [
                    "project_state",
                    "current_research",
                ],
                "work_domains": ["cross_domain_synthesis"],
                "uses_dependency_state": True,
                "requires_external_state": False,
                "requires_mutation": False,
            },
        ],
        "reasoning_summary": (
            "Independent acquisitions feed a dependent synthesis."
        ),
    }


@dataclass
class _Response:
    content: str


class _Model:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.response, list):
            return self.response[len(self.prompts) - 1]
        return self.response


def test_prompt_is_backend_independent_and_dependency_aware():
    prompt = build_task_planner_prompt(
        "Compare my project with current options.",
        _multi_assessment(),
    )

    assert "semantic decomposition only" in prompt
    assert "Do NOT select workers, tools, capabilities" in prompt
    assert "Missing native LangGraph coverage must NEVER" in prompt
    assert "downstream Execution Resolver" in prompt
    assert "deliberately choose Legacy" in prompt
    assert "depends_on" in prompt
    assert "uses_dependency_state=true" in prompt
    assert "dependency outputs" in prompt
    assert "requires_external_state=true means" in prompt
    assert "work_domains are semantic labels only" in prompt


def test_prompt_rejects_non_multi_graph_assessment():
    with pytest.raises(
        ValueError,
        match="requires a multi_graph assessment",
    ):
        build_task_planner_prompt(
            "Just answer this.",
            _conversation_assessment(),
        )


def test_parse_plain_json_response():
    import json

    plan = parse_task_plan_response(
        json.dumps(_valid_plan_payload())
    )

    assert len(plan.steps) == 3
    assert plan.steps[2].depends_on == (
        "project_state",
        "current_research",
    )
    assert plan.steps[2].uses_dependency_state is True
    assert plan.steps[2].requires_external_state is False


def test_parse_fenced_json_response():
    import json

    plan = parse_task_plan_response(
        "```json\n"
        + json.dumps(_valid_plan_payload())
        + "\n```"
    )

    assert plan.steps[0].id == "project_state"


def test_parse_response_object_content():
    import json

    plan = parse_task_plan_response(
        _Response(
            content=json.dumps(
                _valid_plan_payload()
            )
        )
    )

    assert plan.steps[-1].id == "synthesize"


def test_parser_reuses_task_plan_contract_for_backend_rejection():
    payload = _valid_plan_payload()
    payload["steps"][0]["backend"] = "legacy"

    with pytest.raises(
        ValueError,
        match="unsupported fields: backend",
    ):
        parse_task_plan_response(payload)


def test_plan_task_invokes_supplied_model_and_validates_plan():
    import json

    model = _Model(
        json.dumps(_valid_plan_payload())
    )

    plan = plan_task_from_assessment(
        "Compare my project with current options.",
        _multi_assessment(),
        model=model,
    )

    assert len(model.prompts) == 1
    assert "VALIDATED INTENT / COMPLEXITY ASSESSMENT" in (
        model.prompts[0]
    )
    assert plan.steps[2].id == "synthesize"


def test_multi_graph_plan_cannot_collapse_to_one_step():
    import json

    payload = _valid_plan_payload()
    payload["steps"] = [payload["steps"][0]]

    model = _Model(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="at least two semantic steps",
    ):
        plan_task_from_assessment(
            "Do multi-stage work.",
            _multi_assessment(),
            model=model,
        )


def test_sync_planner_retries_once_with_precise_validation_feedback():
    invalid = _valid_plan_payload()
    invalid["steps"][2]["depends_on"] = ["missing"]
    model = _Model([invalid, _valid_plan_payload()])

    plan = plan_task_from_assessment(
        "Compare my project with current options.",
        _multi_assessment(),
        model=model,
    )

    assert plan.steps[-1].id == "synthesize"
    assert len(model.prompts) == 2
    assert "depends on unknown step missing" in model.prompts[1]


def test_sync_planner_exhaustion_is_bounded():
    invalid = _valid_plan_payload()
    invalid["steps"][2]["depends_on"] = ["missing"]
    model = _Model([invalid, invalid])

    with pytest.raises(
        ValueError,
        match="depends on unknown step missing",
    ):
        plan_task_from_assessment(
            "Compare my project with current options.",
            _multi_assessment(),
            model=model,
        )

    assert len(model.prompts) == 2


def test_planner_does_not_import_execution_owners():
    import src.agent_runtime.task_planner as module

    source_path = module.__file__
    text = open(
        source_path,
        "r",
        encoding="utf-8",
    ).read()

    forbidden = (
        "worker_dispatcher",
        "WorkerHandoff",
        "WorkerResult",
        "legacy_runner",
        "capability_registry",
        "SupervisorDecision",
    )

    for marker in forbidden:
        assert marker not in text


def test_arbitrary_future_domains_survive_planning_contract():
    import json

    payload = _valid_plan_payload()
    payload["steps"][0]["work_domains"] = [
        "future_legacy_only_surface"
    ]

    plan = parse_task_plan_response(
        json.dumps(payload)
    )

    assert plan.steps[0].work_domains == (
        "future_legacy_only_surface",
    )
