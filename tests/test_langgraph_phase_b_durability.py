from __future__ import annotations

import sqlite3
import time

import pytest

from src.agent_runtime.durable_checkpoint import (
    CURRENT_RUNTIME_SCHEMA_VERSION,
    CheckpointPersistenceError,
    CheckpointSchemaError,
    DurableRunCheckpointStore,
)
from src.agent_runtime.run_lifecycle import LangGraphRunLifecycle


def _run(store, index: int, *, owner: str = "alice"):
    return LangGraphRunLifecycle.create(
        session_id=f"session-{index}",
        run_id=f"run-{index}",
        owner=owner,
        checkpoint_store=store,
    )


def test_indexed_lookup_survives_more_than_500_unrelated_checkpoints(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "registry.sqlite3")
    target = _run(store, 0)
    target.event({"type": "plan_update", "data": {"plan_id": "oldest"}})
    for index in range(1, 510):
        lifecycle = _run(store, index)
        lifecycle.event({"type": "runtime_snapshot", "data": {"index": index}})

    recovered = store.recover_latest(
        session_id="session-0",
        owner="alice",
    )
    assert recovered is not None
    assert recovered["run_id"] == "run-0"
    assert store.recover_latest(session_id="session-0", owner="bob") is None
    store.close()


def test_event_log_is_compacted_and_replay_is_bounded(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "events.sqlite3")
    lifecycle = _run(store, 1)
    for index in range(725):
        lifecycle.event({"type": "worker_update", "data": {"index": index}})

    recovered = store.recover_latest(run_id="run-1")
    assert recovered is not None
    assert len(recovered["event_log"]) <= 200
    first_page = store.events_after(run_id="run-1", limit=10_000)
    assert len(first_page) == 500
    cursor = first_page[-1]["data"]["event_cursor"]
    second_page = store.events_after(run_id="run-1", after_cursor=cursor)
    assert len(second_page) == 225
    assert second_page[0]["data"]["event_cursor"] == cursor + 1
    store.close()


def test_retention_deletes_only_old_terminal_runs(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "retention.sqlite3")
    old_terminal = _run(store, 1)
    old_terminal.event({"type": "worker_update", "data": {}})
    old_terminal.finish("COMPLETED", stage="done")
    active = _run(store, 2)
    active.own_research_job("research-live", "Keep this job", "running")
    active.event({"type": "worker_update", "data": {}})
    old_interrupted = _run(store, 3)
    old_interrupted.event({"type": "worker_update", "data": {}})
    old_interrupted.suspend("INTERRUPTED", stage="restart")
    store._connection.execute(
        "UPDATE odysseus_runtime_runs SET updated_at=?",
        (time.time() - 90 * 86400,),
    )
    store._connection.commit()

    assert store.cleanup_completed(retention_days=30) == 1
    assert store.run_record("run-1") is None
    assert store.run_record("run-2") is not None
    assert store.run_record("run-3") is not None
    store.close()


def test_retention_uses_typed_runtime_setting_when_not_explicit(
    tmp_path,
    monkeypatch,
):
    import src.agent_runtime.runtime_settings as runtime_settings

    store = DurableRunCheckpointStore(tmp_path / "configured-retention.sqlite3")
    old_terminal = _run(store, 1)
    old_terminal.finish("COMPLETED", stage="done")
    store._connection.execute(
        "UPDATE odysseus_runtime_runs SET updated_at=? WHERE run_id='run-1'",
        (time.time() - 8 * 86400,),
    )
    store._connection.commit()
    configured = dict(runtime_settings.DEFAULT_LANGGRAPH_RUNTIME_SETTINGS)
    configured["completed_run_retention"] = 7
    monkeypatch.setattr(
        runtime_settings,
        "get_langgraph_runtime_settings",
        lambda: configured,
    )
    assert store.cleanup_completed() == 1
    store.close()


def test_schema_migration_and_controlled_rejection(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "schema.sqlite3")
    old = {
        "run_id": "old",
        "thread_id": "thread-old",
        "checkpoint_namespace": "namespace-old",
        "status": "INTERRUPTED",
    }
    migrated = store._migrate_state(old)
    assert migrated["schema_version"] == CURRENT_RUNTIME_SCHEMA_VERSION
    assert migrated["owner"] is None

    with pytest.raises(CheckpointSchemaError, match="newer than supported"):
        store._migrate_state({**old, "schema_version": 999})
    with pytest.raises(CheckpointSchemaError, match="malformed"):
        store._migrate_state({**old, "schema_version": True})
    with pytest.raises(CheckpointSchemaError, match="missing required"):
        store._migrate_state({"schema_version": 1})
    store.close()


def test_closed_connection_surfaces_typed_persistence_failure(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "closed.sqlite3")
    lifecycle = _run(store, 1)
    lifecycle.event({"type": "worker_update", "data": {}})
    store.close()
    with pytest.raises(CheckpointPersistenceError, match="checkpoint save failed"):
        lifecycle.event({"type": "worker_update", "data": {}})


def test_sqlite_busy_timeout_is_configured(tmp_path):
    store = DurableRunCheckpointStore(tmp_path / "busy.sqlite3")
    assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    store.close()
