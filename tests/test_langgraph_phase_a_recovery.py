from __future__ import annotations

import asyncio

import pytest

from src.agent_runtime.durable_checkpoint import DurableRunCheckpointStore
from src.agent_runtime.run_lifecycle import LangGraphRunLifecycle
from src.agent_runtime.step_resume import resume_interrupted_step
from src.agent_runtime.supervisor_contracts import WorkerResult


def _lifecycle(store, *, run_id="run-a", session_id="session-a", owner="alice"):
    lifecycle = LangGraphRunLifecycle.create(
        session_id=session_id,
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        checkpoint_ns=f"odysseus/test/{run_id}",
        owner=owner,
        checkpoint_store=store,
    )
    lifecycle.event({"type": "runtime_run", "data": {"status": "RUNNING"}})
    return lifecycle


def test_process_reopen_reconciles_running_to_interrupted(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    first = DurableRunCheckpointStore(path)
    lifecycle = _lifecycle(first)
    state = first.recover_latest(run_id="run-a")
    assert state["status"] == "RUNNING"
    assert first.reconcile_state(state, executor_alive=True)["status"] == "RUNNING"
    first.close()

    reopened = DurableRunCheckpointStore(path)
    state = reopened.recover_latest(run_id="run-a")
    reconciled = reopened.reconcile_state(state, executor_alive=False)
    assert reconciled["status"] == "INTERRUPTED"
    assert reopened.run_record("run-a")["status"] == "INTERRUPTED"
    assert lifecycle.terminal is None
    reopened.close()


def test_terminal_state_is_immutable_during_reconciliation(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    first = DurableRunCheckpointStore(path)
    lifecycle = _lifecycle(first)
    lifecycle.finish("COMPLETED", stage="test")
    first.close()

    reopened = DurableRunCheckpointStore(path)
    state = reopened.recover_latest(run_id="run-a")
    assert reopened.reconcile_state(state)["status"] == "COMPLETED"
    assert reopened.set_run_status("run-a", "INTERRUPTED") == "COMPLETED"
    reopened.close()


def test_pending_decision_survives_reopen_and_preserves_wait_state(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    first = DurableRunCheckpointStore(path)
    _lifecycle(first)
    decision = first.create_decision(
        kind="approval",
        run_id="run-a",
        session_id="session-a",
        owner="alice",
        step_id="step-1",
        attempt_id="attempt-1",
        action_text="send_email(to=alice@example.com)",
        fingerprint="fingerprint-a",
        side_effect_risk="external_write",
    )
    first.close()

    reopened = DurableRunCheckpointStore(path)
    state = reopened.recover_latest(run_id="run-a")
    assert reopened.reconcile_state(state)["status"] == "AWAITING_APPROVAL"
    with pytest.raises(PermissionError):
        reopened.resolve_decision(
            record_id=decision["record_id"],
            owner="bob",
            session_id="session-a",
            run_id="run-a",
            response={"approved": True},
            state="approved",
        )
    approved = reopened.resolve_decision(
        record_id=decision["record_id"],
        owner="alice",
        session_id="session-a",
        run_id="run-a",
        response={"approved": True},
        state="approved",
    )
    duplicate = reopened.resolve_decision(
        record_id=decision["record_id"],
        owner="alice",
        session_id="session-a",
        run_id="run-a",
        response={"approved": True},
        state="approved",
    )
    assert approved["record_id"] == duplicate["record_id"]
    reopened.close()


def test_missing_or_expired_decision_cannot_hold_wait_state(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "runtime.sqlite3")
    lifecycle = _lifecycle(store)
    lifecycle.suspend(
        "AWAITING_CLARIFICATION",
        stage="test",
        reason="Need detail",
    )
    state = store.recover_latest(run_id="run-a")
    assert store.reconcile_state(state)["status"] == "INTERRUPTED"
    store.close()


def test_step_attempt_resume_outcomes_and_history_survive_reopen(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    first = DurableRunCheckpointStore(path)
    _lifecycle(first)
    read = first.create_step_attempt(
        run_id="run-a",
        plan_id="plan-a",
        plan_revision=1,
        step_id="read-step",
        worker_kind="general",
        input_refs={"objective": "Read status", "dependency_outputs": {}},
        side_effect_classification="read_only",
    )
    write = first.create_step_attempt(
        run_id="run-a",
        plan_id="plan-a",
        plan_revision=1,
        step_id="write-step",
        worker_kind="code",
        input_refs={"objective": "Write file", "dependency_outputs": {}},
        side_effect_classification="external_effect",
    )
    assert read["resumability"] == "SAFE_TO_RESTART"
    assert write["resumability"] == "REQUIRES_VERIFICATION"
    first.close()

    reopened = DurableRunCheckpointStore(path)
    assert reopened.latest_step_attempt(
        run_id="run-a", step_id="read-step"
    )["attempt_id"] == read["attempt_id"]
    completed = reopened.complete_step_attempt(
        read["attempt_id"],
        status="COMPLETED",
        result={"summary": "done"},
    )
    assert completed["resumability"] == "ALREADY_COMPLETED"
    reopened.close()


@pytest.mark.asyncio
async def test_safe_step_resume_creates_new_attempt_and_never_restarts_query(
    tmp_path,
):
    path = tmp_path / "runtime.sqlite3"
    first = DurableRunCheckpointStore(path)
    _lifecycle(first)
    original = first.create_step_attempt(
        run_id="run-a",
        plan_id="plan-a",
        plan_revision=1,
        step_id="step-1",
        worker_kind="general",
        input_refs={
            "objective": "Read the current status",
            "dependency_outputs": {},
            "session_id": "session-a",
        },
        side_effect_classification="read_only",
    )
    first.close()

    reopened = DurableRunCheckpointStore(path)
    state = reopened.recover_latest(run_id="run-a")
    state = reopened.reconcile_state(state)
    calls = []

    async def runner(handoff, *, session_id=None):
        calls.append((handoff.objective, session_id))
        return WorkerResult(
            capability="general",
            status="completed",
            summary="Recovered read completed.",
            data={"value": "ok"},
        )

    result = await resume_interrupted_step(
        checkpoint_store=reopened,
        checkpoint_state=state,
        owner="alice",
        worker_runners={"general": runner},
    )
    assert result["outcome"] == "SAFE_TO_RESTART"
    assert result["previous_attempt_id"] == original["attempt_id"]
    assert result["attempt_id"] != original["attempt_id"]
    assert calls == [("Read the current status", "session-a")]
    recovered = reopened.recover_latest(run_id="run-a")
    assert "step-1" in recovered["completed_steps"]
    assert recovered["status"] == "SUSPENDED"
    reopened.close()


@pytest.mark.asyncio
async def test_unknown_write_outcome_is_not_executed_on_resume(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "runtime.sqlite3")
    lifecycle = _lifecycle(store)
    store.create_step_attempt(
        run_id="run-a",
        plan_id="plan-a",
        plan_revision=1,
        step_id="write-step",
        worker_kind="code",
        input_refs={"objective": "Write a file"},
        side_effect_classification="external_effect",
    )
    lifecycle.suspend("INTERRUPTED", stage="test")
    state = store.recover_latest(run_id="run-a")
    called = False

    async def runner(*args, **kwargs):
        nonlocal called
        called = True

    result = await resume_interrupted_step(
        checkpoint_store=store,
        checkpoint_state=state,
        owner="alice",
        worker_runners={"code": runner},
    )
    assert result["outcome"] == "REQUIRES_VERIFICATION"
    assert called is False
    store.close()
