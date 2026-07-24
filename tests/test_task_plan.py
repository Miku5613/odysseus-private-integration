from __future__ import annotations

import pytest

from src.agent_runtime.task_plan import (
    PlanStep,
    TaskPlan,
    blocked_step_ids,
    ready_step_ids,
    topological_step_ids,
    validate_task_plan_payload,
)


def _multi_plan_payload():
    return {
        "objective": "Compare private project state with current research.",
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
                "objective": "Combine both inputs into the requested result.",
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
            "Two independent acquisition surfaces feed one synthesis step."
        ),
    }


def test_validate_dependency_aware_backend_independent_plan():
    plan = validate_task_plan_payload(
        _multi_plan_payload()
    )

    assert isinstance(plan, TaskPlan)
    assert len(plan.steps) == 3
    assert plan.steps[0] == PlanStep(
        id="project_state",
        objective="Retrieve relevant project state.",
        depends_on=(),
        work_domains=("user_project_state",),
        requires_external_state=True,
        requires_mutation=False,
    )
    assert plan.steps[2].depends_on == (
        "project_state",
        "current_research",
    )
    assert plan.steps[2].uses_dependency_state is True


def test_dependency_state_usage_requires_dependency():
    payload = _multi_plan_payload()
    payload["steps"][0]["uses_dependency_state"] = True

    with pytest.raises(
        ValueError,
        match="uses_dependency_state=true requires at least one depends_on",
    ):
        validate_task_plan_payload(payload)


def test_backend_execution_fields_are_rejected():
    payload = _multi_plan_payload()
    payload["steps"][0]["backend"] = "legacy"

    with pytest.raises(
        ValueError,
        match="unsupported fields: backend",
    ):
        validate_task_plan_payload(payload)


def test_capability_selection_is_rejected():
    payload = _multi_plan_payload()
    payload["steps"][0]["capability"] = "knowledge_search"

    with pytest.raises(
        ValueError,
        match="unsupported fields: capability",
    ):
        validate_task_plan_payload(payload)


def test_duplicate_step_ids_are_rejected():
    payload = _multi_plan_payload()
    payload["steps"][1]["id"] = "project_state"

    with pytest.raises(
        ValueError,
        match="step ids must be unique",
    ):
        validate_task_plan_payload(payload)


def test_missing_dependency_is_rejected():
    payload = _multi_plan_payload()
    payload["steps"][2]["depends_on"].append(
        "missing_step"
    )

    with pytest.raises(
        ValueError,
        match="depends on unknown step missing_step",
    ):
        validate_task_plan_payload(payload)


def test_self_dependency_is_rejected():
    payload = _multi_plan_payload()
    payload["steps"][0]["depends_on"] = [
        "project_state"
    ]

    with pytest.raises(
        ValueError,
        match="cannot depend on itself",
    ):
        validate_task_plan_payload(payload)


def test_cycle_is_rejected():
    payload = _multi_plan_payload()
    payload["steps"][0]["depends_on"] = [
        "synthesize"
    ]

    with pytest.raises(
        ValueError,
        match="must be acyclic",
    ):
        validate_task_plan_payload(payload)


def test_topological_order_preserves_stable_parallel_order():
    plan = validate_task_plan_payload(
        _multi_plan_payload()
    )

    assert topological_step_ids(plan) == (
        "project_state",
        "current_research",
        "synthesize",
    )


def test_ready_steps_expose_parallel_frontier():
    plan = validate_task_plan_payload(
        _multi_plan_payload()
    )

    assert ready_step_ids(plan) == (
        "project_state",
        "current_research",
    )

    assert ready_step_ids(
        plan,
        completed_step_ids=("project_state",),
    ) == (
        "current_research",
    )

    assert ready_step_ids(
        plan,
        completed_step_ids=(
            "project_state",
            "current_research",
        ),
    ) == (
        "synthesize",
    )


def test_failed_dependency_blocks_downstream_transitively():
    payload = _multi_plan_payload()
    payload["steps"].append(
        {
            "id": "finalize",
            "objective": "Finalize the answer.",
            "depends_on": ["synthesize"],
            "work_domains": ["final_synthesis"],
            "requires_external_state": False,
            "requires_mutation": False,
        }
    )
    plan = validate_task_plan_payload(payload)

    assert blocked_step_ids(
        plan,
        failed_step_ids=("project_state",),
    ) == (
        "synthesize",
        "finalize",
    )

    assert ready_step_ids(
        plan,
        failed_step_ids=("project_state",),
    ) == (
        "current_research",
    )


def test_unknown_runtime_state_step_is_rejected():
    plan = validate_task_plan_payload(
        _multi_plan_payload()
    )

    with pytest.raises(
        ValueError,
        match="references unknown step missing",
    ):
        ready_step_ids(
            plan,
            completed_step_ids=("missing",),
        )


def test_step_cannot_be_completed_and_failed():
    plan = validate_task_plan_payload(
        _multi_plan_payload()
    )

    with pytest.raises(
        ValueError,
        match="both completed and failed",
    ):
        ready_step_ids(
            plan,
            completed_step_ids=("project_state",),
            failed_step_ids=("project_state",),
        )


def test_arbitrary_future_work_domains_are_allowed():
    payload = _multi_plan_payload()
    payload["steps"][0]["work_domains"] = [
        "future_legacy_only_surface",
        "calendar_conflict_state",
    ]

    plan = validate_task_plan_payload(payload)

    assert plan.steps[0].work_domains == (
        "future_legacy_only_surface",
        "calendar_conflict_state",
    )


def test_single_step_plan_is_valid_for_replanning_convergence():
    plan = validate_task_plan_payload(
        {
            "objective": "Finish the remaining bounded work.",
            "steps": [
                {
                    "id": "finish",
                    "objective": "Complete the remaining task.",
                    "depends_on": [],
                    "work_domains": ["remaining_work"],
                    "requires_external_state": False,
                    "requires_mutation": False,
                }
            ],
            "reasoning_summary": "",
        }
    )

    assert topological_step_ids(plan) == ("finish",)
    assert ready_step_ids(plan) == ("finish",)
