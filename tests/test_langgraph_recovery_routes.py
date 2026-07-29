from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import routes.chat_routes as chat_routes
from src.agent_runtime.durable_checkpoint import DurableRunCheckpointStore
from src.agent_runtime.run_lifecycle import LangGraphRunLifecycle


class _AgentRuns:
    def is_active(self, session_id):
        return False

    def stop(self, session_id):
        return False


def _router(monkeypatch, store, *, owner="alice"):
    monkeypatch.setattr(
        chat_routes,
        "_durable_checkpoint_store_or_503",
        lambda: store,
    )
    monkeypatch.setattr(chat_routes, "agent_runs", _AgentRuns())

    def verify(request, session_id):
        if request.headers.get("x-user", owner) != owner:
            raise HTTPException(403, "owner mismatch")

    monkeypatch.setattr(chat_routes, "_verify_session_owner", verify)
    monkeypatch.setattr(
        chat_routes,
        "effective_user",
        lambda request: request.headers.get("x-user", owner),
    )
    values = [object() for _ in range(6)]
    app = FastAPI()
    app.include_router(chat_routes.setup_chat_routes(*values))
    return TestClient(app)


def _running(store, *, run_id="run-route", session_id="session-route"):
    lifecycle = LangGraphRunLifecycle.create(
        session_id=session_id,
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        checkpoint_ns=f"odysseus/test/{run_id}",
        owner="alice",
        checkpoint_store=store,
    )
    lifecycle.event({"type": "runtime_run", "data": {"status": "RUNNING"}})
    lifecycle.event({"type": "plan_update", "data": {"revision": 1}})
    return lifecycle


def test_status_and_replay_agree_after_process_style_reopen(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "routes.sqlite3"
    first = DurableRunCheckpointStore(path)
    _running(first)
    first.close()
    reopened = DurableRunCheckpointStore(path)
    client = _router(monkeypatch, reopened)

    status = client.get(
        "/api/chat/stream_status/session-route",
        headers={"x-user": "alice"},
    )
    assert status.status_code == 200
    assert status.json()["status"] == "interrupted"
    assert status.json()["terminal"] is None

    replay = client.get(
        "/api/chat/resume/session-route?after_cursor=1",
        headers={"x-user": "alice"},
    )
    assert replay.status_code == 200
    assert '"type": "plan_update"' in replay.text
    assert '"type": "runtime_replay_complete"' in replay.text
    assert '"status": "INTERRUPTED"' in replay.text
    assert '"run_completed": false' in replay.text
    assert replay.text.count("data: [DONE]") == 1

    duplicate = client.get(
        "/api/chat/resume/session-route?after_cursor=1",
        headers={"x-user": "alice"},
    )
    assert duplicate.text == replay.text
    reopened.close()


def test_recovered_stop_is_scoped_and_terminal_once(tmp_path, monkeypatch):
    path = tmp_path / "routes.sqlite3"
    first = DurableRunCheckpointStore(path)
    _running(first)
    first.close()
    reopened = DurableRunCheckpointStore(path)
    client = _router(monkeypatch, reopened)

    response = client.post(
        "/api/chat/stop/session-route",
        headers={"x-user": "alice"},
    )
    assert response.status_code == 200
    assert response.json() == {"stopped": True}
    state = reopened.recover_latest(run_id="run-route")
    terminals = [
        event
        for event in state["event_log"]
        if event.get("type") == "runtime_terminal"
    ]
    assert state["status"] == "CANCELLED"
    assert len(terminals) == 1
    assert client.post(
        "/api/chat/stop/session-route",
        headers={"x-user": "alice"},
    ).json() == {"stopped": False}
    reopened.close()


def test_route_blocks_unknown_write_resume_and_other_user(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "routes.sqlite3"
    store = DurableRunCheckpointStore(path)
    lifecycle = _running(store)
    store.create_step_attempt(
        run_id="run-route",
        plan_id="plan-a",
        plan_revision=1,
        step_id="write-step",
        worker_kind="code",
        input_refs={"objective": "Write file"},
        side_effect_classification="external_effect",
    )
    lifecycle.suspend("INTERRUPTED", stage="test")
    client = _router(monkeypatch, store)

    blocked = client.post(
        "/api/chat/resume_step/session-route",
        headers={"x-user": "alice"},
        json={"step_id": "write-step"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["outcome"] == "REQUIRES_VERIFICATION"

    forbidden = client.get(
        "/api/chat/stream_status/session-route",
        headers={"x-user": "bob"},
    )
    assert forbidden.status_code == 403
    store.close()
