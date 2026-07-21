from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.supervisor_decision import SupervisorDecision
from src.agent_runtime.worker_dispatcher import (
    build_worker_handoff,
    builtin_worker_runners,
    dispatch_supervisor_decision,
)


def test_builtin_dispatcher_registers_code_worker():
    runners = builtin_worker_runners()

    assert "code_inspection" in runners


def test_build_handoff_preserves_generic_worker_context():
    handoff = build_worker_handoff(
        "code_inspection",
        "Find the implementation.",
        supervisor_state={
            "worker_context": {
                "workspace": "D:/Projects/example",
                "target": "example_function",
            },
            "missing_information": [],
        },
    )

    assert handoff.capability == "code_inspection"
    assert handoff.objective == "Find the implementation."
    assert handoff.context["workspace"] == "D:/Projects/example"
    assert handoff.context["target"] == "example_function"


def test_dispatcher_executes_selected_code_worker():
    captured = {}

    def fake_code_worker(handoff, session_id=None):
        captured["handoff"] = handoff
        captured["session_id"] = session_id

        return WorkerResult(
            capability="code_inspection",
            status="completed",
            summary="Implementation found.",
        )

    decision = SupervisorDecision(
        selected_capabilities=("code_inspection",),
    )

    result = dispatch_supervisor_decision(
        decision,
        "Find where the route is implemented.",
        supervisor_state={
            "worker_context": {
                "workspace": "D:/Projects/example",
            }
        },
        worker_runners={
            "code_inspection": fake_code_worker,
        },
        session_id="session-42",
    )

    assert result.requested_capabilities == ("code_inspection",)
    assert result.completed_capabilities == ("code_inspection",)
    assert result.undispatched_capabilities == ()
    assert result.worker_results[0].summary == "Implementation found."
    assert captured["session_id"] == "session-42"
    assert captured["handoff"].context["workspace"] == "D:/Projects/example"


def test_dispatcher_supports_multiple_selected_capabilities():
    def fake_runner(handoff, session_id=None):
        return WorkerResult(
            capability=handoff.capability,
            status="completed",
            summary=f"{handoff.capability} complete",
        )

    decision = SupervisorDecision(
        selected_capabilities=(
            "knowledge_search",
            "deep_research",
            "code_inspection",
        ),
    )

    result = dispatch_supervisor_decision(
        decision,
        "Compare notes, research, and implementation.",
        worker_runners={
            "knowledge_search": fake_runner,
            "deep_research": fake_runner,
            "code_inspection": fake_runner,
        },
    )

    assert result.completed_capabilities == (
        "knowledge_search",
        "deep_research",
        "code_inspection",
    )
    assert len(result.worker_results) == 3


def test_unimplemented_worker_is_reported_not_silently_rerouted():
    decision = SupervisorDecision(
        selected_capabilities=("knowledge_search",),
    )

    result = dispatch_supervisor_decision(
        decision,
        "Search my notes.",
        worker_runners={},
    )

    assert result.worker_results == ()
    assert result.completed_capabilities == ()
    assert result.undispatched_capabilities == ("knowledge_search",)


def test_failed_worker_is_not_marked_completed():
    def fake_failed_worker(handoff, session_id=None):
        return WorkerResult(
            capability=handoff.capability,
            status="failed",
            summary="Worker failed.",
            error="boom",
        )

    decision = SupervisorDecision(
        selected_capabilities=("code_inspection",),
    )

    result = dispatch_supervisor_decision(
        decision,
        "Inspect this code.",
        worker_runners={
            "code_inspection": fake_failed_worker,
        },
    )

    assert result.completed_capabilities == ()
    assert result.worker_results[0].status == "failed"


def test_dispatcher_does_not_collapse_multi_capability_decision_to_one_worker():
    calls = []

    def fake_runner(handoff, session_id=None):
        calls.append(handoff.capability)
        return WorkerResult(
            capability=handoff.capability,
            status="completed",
            summary="done",
        )

    decision = SupervisorDecision(
        selected_capabilities=(
            "knowledge_search",
            "code_inspection",
        ),
    )

    dispatch_supervisor_decision(
        decision,
        "Compare my notes against the implementation.",
        worker_runners={
            "knowledge_search": fake_runner,
            "code_inspection": fake_runner,
        },
    )

    assert calls == [
        "knowledge_search",
        "code_inspection",
    ]
