from __future__ import annotations

from dataclasses import replace

import pytest

from src.agent_runtime.plan_execution import (
    begin_replanned_execution,
    complete_step,
    fail_step,
    initialize_plan_execution,
    partial_step,
    request_replan,
    start_step,
    step_execution,
    task_plan_fingerprint,
)
from src.agent_runtime.task_plan import (
    PlanStep,
    TaskPlan,
    validate_task_plan_payload,
)


def _plan():
    return validate_task_plan_payload(
        {
            "objective": (
                "Compare project state with current research."
            ),
            "steps": [
                {
                    "id": "project_state",
                    "objective": (
                        "Retrieve relevant project state."
                    ),
                    "depends_on": [],
                    "work_domains": [
                        "user_project_state"
                    ],
                    "requires_external_state": True,
                    "requires_mutation": False,
                },
                {
                    "id": "current_research",
                    "objective": (
                        "Acquire current external research."
                    ),
                    "depends_on": [],
                    "work_domains": [
                        "current_public_research"
                    ],
                    "requires_external_state": True,
                    "requires_mutation": False,
                },
                {
                    "id": "synthesize",
                    "objective": (
                        "Combine the acquired evidence."
                    ),
                    "depends_on": [
                        "project_state",
                        "current_research",
                    ],
                    "work_domains": [
                        "cross_domain_synthesis"
                    ],
                    "uses_dependency_state": True,
                    "requires_external_state": False,
                    "requires_mutation": False,
                },
                {
                    "id": "finalize",
                    "objective": (
                        "Finalize the user-facing result."
                    ),
                    "depends_on": [
                        "synthesize"
                    ],
                    "work_domains": [
                        "final_synthesis"
                    ],
                    "requires_external_state": False,
                    "requires_mutation": False,
                },
            ],
            "reasoning_summary": (
                "Two independent acquisitions feed synthesis."
            ),
        }
    )


def test_initial_state_exposes_parallel_ready_frontier():
    state = initialize_plan_execution(_plan())

    assert step_execution(
        state,
        "project_state",
    ).status == "ready"

    assert step_execution(
        state,
        "current_research",
    ).status == "ready"

    assert step_execution(
        state,
        "synthesize",
    ).status == "pending"

    assert step_execution(
        state,
        "finalize",
    ).status == "pending"


def test_start_requires_ready_status():
    state = initialize_plan_execution(_plan())

    running = start_step(
        state,
        "project_state",
    )

    assert step_execution(
        running,
        "project_state",
    ).status == "running"

    with pytest.raises(
        ValueError,
        match="cannot start from status pending",
    ):
        start_step(
            state,
            "synthesize",
        )


def test_completion_records_output_and_artifacts():
    state = initialize_plan_execution(_plan())
    state = start_step(
        state,
        "project_state",
    )
    state = complete_step(
        state,
        "project_state",
        output={
            "summary": "project evidence",
        },
        artifact_refs=(
            "artifact://project/1",
            "artifact://project/1",
            "artifact://project/2",
        ),
    )

    completed = step_execution(
        state,
        "project_state",
    )

    assert completed.status == "completed"
    assert completed.output == {
        "summary": "project evidence",
    }
    assert completed.artifact_refs == (
        "artifact://project/1",
        "artifact://project/2",
    )


def test_dependents_unlock_only_after_all_prerequisites_complete():
    state = initialize_plan_execution(_plan())

    state = start_step(
        state,
        "project_state",
    )
    state = complete_step(
        state,
        "project_state",
    )

    assert step_execution(
        state,
        "synthesize",
    ).status == "pending"

    state = start_step(
        state,
        "current_research",
    )
    state = complete_step(
        state,
        "current_research",
    )

    assert step_execution(
        state,
        "synthesize",
    ).status == "ready"


def test_failure_blocks_dependents_transitively_but_not_independent_work():
    state = initialize_plan_execution(_plan())
    state = start_step(
        state,
        "project_state",
    )
    state = fail_step(
        state,
        "project_state",
        error="project source unavailable",
    )

    assert step_execution(
        state,
        "project_state",
    ).status == "failed"

    assert step_execution(
        state,
        "project_state",
    ).error == "project source unavailable"

    assert step_execution(
        state,
        "synthesize",
    ).status == "blocked"

    assert step_execution(
        state,
        "finalize",
    ).status == "blocked"

    assert step_execution(
        state,
        "current_research",
    ).status == "ready"


def test_only_running_step_can_complete_or_fail():
    state = initialize_plan_execution(_plan())

    with pytest.raises(
        ValueError,
        match="cannot complete from status ready",
    ):
        complete_step(
            state,
            "project_state",
        )

    with pytest.raises(
        ValueError,
        match="cannot fail from status ready",
    ):
        fail_step(
            state,
            "project_state",
            error="nope",
        )


def test_replan_request_captures_failure_context_and_revision():
    state = initialize_plan_execution(_plan())
    state = start_step(
        state,
        "project_state",
    )
    state = fail_step(
        state,
        "project_state",
        error="source unavailable",
    )

    request = request_replan(
        state,
        reason=(
            "Required source is unavailable; "
            "find another valid route."
        ),
        trigger_step_id="project_state",
    )

    assert request.from_revision == 1
    assert request.trigger_step_id == "project_state"
    assert request.failed_step_ids == (
        "project_state",
    )
    assert request.blocked_step_ids == (
        "synthesize",
        "finalize",
    )
    assert request.plan_fingerprint == (
        task_plan_fingerprint(state.plan)
    )


def test_partial_step_preserves_output_and_blocks_only_dependents():
    state = initialize_plan_execution(_plan())
    state = start_step(state, "project_state")
    state = partial_step(
        state,
        "project_state",
        output={"evidence": "useful partial"},
        artifact_refs=("artifact://partial",),
    )

    partial = step_execution(state, "project_state")
    assert partial.status == "partial"
    assert partial.output == {"evidence": "useful partial"}
    assert partial.artifact_refs == ("artifact://partial",)
    assert step_execution(state, "current_research").status == "ready"
    assert step_execution(state, "synthesize").status == "blocked"

    request = request_replan(
        state,
        reason="A separate step needs genuine clarification.",
        trigger_step_id="current_research",
    )
    assert request.partial_step_ids == ("project_state",)


def test_replanned_execution_increments_revision_and_lineage():
    state = initialize_plan_execution(_plan())

    revised = begin_replanned_execution(
        state,
        _plan(),
        reason="Use a revised orchestration strategy.",
    )

    assert revised.revision == 2
    assert revised.prior_plan_fingerprints == (
        task_plan_fingerprint(state.plan),
    )
    assert revised.replan_reason == (
        "Use a revised orchestration strategy."
    )


def test_replan_never_reuses_completed_output_implicitly():
    state = initialize_plan_execution(_plan())
    state = start_step(
        state,
        "project_state",
    )
    state = complete_step(
        state,
        "project_state",
        output="important evidence",
        artifact_refs=(
            "artifact://project/1",
        ),
    )

    revised = begin_replanned_execution(
        state,
        _plan(),
        reason="Re-evaluate the plan.",
    )

    new_step = step_execution(
        revised,
        "project_state",
    )

    assert new_step.status == "ready"
    assert new_step.output is None
    assert new_step.artifact_refs == ()


def test_explicit_reuse_preserves_only_completed_identical_step():
    state = initialize_plan_execution(_plan())
    state = start_step(
        state,
        "project_state",
    )
    state = complete_step(
        state,
        "project_state",
        output="important evidence",
        artifact_refs=(
            "artifact://project/1",
        ),
    )

    revised = begin_replanned_execution(
        state,
        _plan(),
        reason="Re-evaluate remaining work.",
        reusable_step_ids=(
            "project_state",
        ),
    )

    reused = step_execution(
        revised,
        "project_state",
    )

    assert reused.status == "completed"
    assert reused.output == "important evidence"
    assert reused.artifact_refs == (
        "artifact://project/1",
    )


def test_reuse_rejects_incomplete_step():
    state = initialize_plan_execution(_plan())

    with pytest.raises(
        ValueError,
        match="must be completed",
    ):
        begin_replanned_execution(
            state,
            _plan(),
            reason="Replan.",
            reusable_step_ids=(
                "project_state",
            ),
        )


def test_reuse_rejects_changed_semantic_step_definition():
    state = initialize_plan_execution(_plan())
    state = start_step(
        state,
        "project_state",
    )
    state = complete_step(
        state,
        "project_state",
        output="old evidence",
    )

    plan = _plan()
    changed_step = replace(
        plan.step_by_id("project_state"),
        objective="Retrieve different project state.",
    )
    revised_plan = TaskPlan(
        objective=plan.objective,
        steps=tuple(
            changed_step
            if step.id == "project_state"
            else step
            for step in plan.steps
        ),
        reasoning_summary=plan.reasoning_summary,
    )

    with pytest.raises(
        ValueError,
        match="changed semantic definition",
    ):
        begin_replanned_execution(
            state,
            revised_plan,
            reason="Requirements changed.",
            reusable_step_ids=(
                "project_state",
            ),
        )


def test_reused_completed_prerequisite_can_unlock_revised_frontier():
    state = initialize_plan_execution(_plan())
    state = start_step(
        state,
        "project_state",
    )
    state = complete_step(
        state,
        "project_state",
        output="project evidence",
    )

    revised = begin_replanned_execution(
        state,
        _plan(),
        reason="Continue with a revised revision.",
        reusable_step_ids=(
            "project_state",
        ),
    )

    assert step_execution(
        revised,
        "current_research",
    ).status == "ready"

    assert step_execution(
        revised,
        "synthesize",
    ).status == "pending"


def test_plan_fingerprint_changes_when_semantics_change():
    plan = _plan()
    changed = TaskPlan(
        objective="Different objective",
        steps=plan.steps,
        reasoning_summary=plan.reasoning_summary,
    )

    assert task_plan_fingerprint(plan) != (
        task_plan_fingerprint(changed)
    )


def test_plan_fingerprint_tracks_dependency_state_semantics():
    plan = _plan()
    changed_step = replace(
        plan.step_by_id("synthesize"),
        uses_dependency_state=False,
    )
    changed = TaskPlan(
        objective=plan.objective,
        steps=tuple(
            changed_step
            if step.id == "synthesize"
            else step
            for step in plan.steps
        ),
        reasoning_summary=plan.reasoning_summary,
    )

    assert task_plan_fingerprint(plan) != (
        task_plan_fingerprint(changed)
    )


def test_execution_state_module_is_backend_independent():
    import src.agent_runtime.plan_execution as module

    text = open(
        module.__file__,
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
        "builtin_worker_runners",
        "mcp_worker",
        "research_worker",
        "knowledge_worker",
        "code_worker",
    )

    for marker in forbidden:
        assert marker not in text
