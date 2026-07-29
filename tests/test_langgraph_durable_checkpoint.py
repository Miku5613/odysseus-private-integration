from __future__ import annotations

import asyncio

from routes.chat_routes import _replay_durable_langgraph_run
from src.agent_runtime.durable_checkpoint import (
    DurableRunCheckpointStore,
)
from src.agent_runtime.run_lifecycle import LangGraphRunLifecycle


def _lifecycle(store, run_id, session_id):
    return LangGraphRunLifecycle.create(
        session_id=session_id,
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        owner="owner-1",
        checkpoint_store=store,
    )


def test_checkpoint_survives_store_restart_and_recovers_active_plan(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    first_store = DurableRunCheckpointStore(path)
    lifecycle = _lifecycle(first_store, "run-1", "session-1")
    lifecycle.event({
        "type": "runtime_run",
        "data": {"status": "RUNNING"},
    })
    lifecycle.event({
        "type": "plan_update",
        "data": {
            "status": "planned",
            "steps": [{"id": "gather"}, {"id": "synthesize"}],
        },
    })
    thread_id = lifecycle.identity.thread_id
    checkpoint_ns = lifecycle.identity.checkpoint_ns
    first_store.close()

    restarted_store = DurableRunCheckpointStore(path)
    recovered = LangGraphRunLifecycle.recover(
        checkpoint_store=restarted_store,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
    )

    assert recovered is not None
    assert recovered.identity.run_id == "run-1"
    assert recovered.status == "RUNNING"
    assert recovered.current_plan["steps"][0]["id"] == "gather"
    assert recovered.event_cursor == 2
    continued = recovered.event({
        "type": "worker_update",
        "data": {"step_id": "gather"},
    })
    assert continued["data"]["event_cursor"] == 3
    restarted_store.close()


def test_terminal_is_idempotent_after_recovery(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    first_store = DurableRunCheckpointStore(path)
    lifecycle = _lifecycle(first_store, "run-1", "session-1")
    lifecycle.finish("COMPLETED", stage="coordinator")
    identity = lifecycle.identity
    first_store.close()

    restarted_store = DurableRunCheckpointStore(path)
    recovered = LangGraphRunLifecycle.recover(
        checkpoint_store=restarted_store,
        thread_id=identity.thread_id,
        checkpoint_ns=identity.checkpoint_ns,
    )

    assert recovered is not None
    assert recovered.finish("FAILED", stage="late-error") is None
    assert recovered.status == "COMPLETED"
    assert len([
        event
        for event in recovered.event_log
        if event.get("type") == "runtime_terminal"
    ]) == 1
    restarted_store.close()


def test_concurrent_runs_and_recovery_cancellation_are_isolated(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "checkpoints.sqlite3")
    run_a = _lifecycle(store, "run-a", "session-a")
    run_b = _lifecycle(store, "run-b", "session-b")
    run_a.event({"type": "runtime_run", "data": {}})
    run_b.event({"type": "runtime_run", "data": {}})
    identities = (run_a.identity, run_b.identity)
    store.close()

    restarted = DurableRunCheckpointStore(tmp_path / "checkpoints.sqlite3")
    recovered_a = LangGraphRunLifecycle.recover(
        checkpoint_store=restarted,
        thread_id=identities[0].thread_id,
        checkpoint_ns=identities[0].checkpoint_ns,
    )
    recovered_b = LangGraphRunLifecycle.recover(
        checkpoint_store=restarted,
        thread_id=identities[1].thread_id,
        checkpoint_ns=identities[1].checkpoint_ns,
    )

    assert recovered_a is not None
    assert recovered_b is not None
    recovered_a.cancellation.cancel()
    assert recovered_a.cancellation.cancelled is True
    assert recovered_b.cancellation.cancelled is False
    assert recovered_a.identity.checkpoint_ns != recovered_b.identity.checkpoint_ns
    restarted.close()


def test_latest_run_can_be_found_by_session_or_run_id(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "checkpoints.sqlite3")
    lifecycle = _lifecycle(store, "run-find", "session-find")
    lifecycle.event({"type": "runtime_run", "data": {}})

    by_session = store.recover_latest(session_id="session-find")
    by_run = store.recover_latest(run_id="run-find")

    assert by_session["thread_id"] == lifecycle.identity.thread_id
    assert (
        by_run["checkpoint_namespace"]
        == lifecycle.identity.checkpoint_ns
    )
    store.close()


def test_replay_continues_after_ordered_cursor():
    state = {
        "event_log": [
            {"type": "runtime_run", "data": {"event_cursor": 1}},
            {"type": "plan_update", "data": {"event_cursor": 2}},
            {"type": "runtime_terminal", "data": {"event_cursor": 3}},
        ]
    }

    async def collect():
        return [
            frame
            async for frame in _replay_durable_langgraph_run(
                state,
                after_cursor=1,
            )
        ]

    frames = asyncio.run(collect())

    assert '"event_cursor": 1' not in "".join(frames)
    assert '"event_cursor": 2' in frames[0]
    assert '"event_cursor": 3' in frames[1]
    assert frames[-1] == "data: [DONE]\n\n"
