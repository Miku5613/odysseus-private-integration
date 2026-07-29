from __future__ import annotations

import atexit
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class CheckpointUnavailableError(RuntimeError):
    pass


class RunCheckpointState(TypedDict, total=False):
    run_id: str
    session_id: str | None
    thread_id: str
    checkpoint_namespace: str
    runtime_mode: str
    status: str
    admission_result: str | None
    fallback_policy: str | None
    cancellation_scope: str
    current_graph: str | None
    current_worker: str | None
    active_step: str | None
    completed_steps: list[str]
    dependency_results: dict[str, Any]
    artifact_references: list[dict[str, Any]]
    failure: dict[str, Any] | None
    event_cursor: int
    event_log: list[dict[str, Any]]
    current_plan: dict[str, Any] | None
    terminal: dict[str, Any] | None


def default_checkpoint_path() -> Path:
    configured = os.getenv("ODYSSEUS_LANGGRAPH_CHECKPOINT_DB", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "langgraph" / "checkpoints.sqlite3"


class DurableRunCheckpointStore:
    """SQLite-backed LangGraph graph containing authoritative run snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            serializer = JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=None,
            )
            self.checkpointer = SqliteSaver(
                self._connection,
                serde=serializer,
            )
            self.checkpointer.setup()
        except Exception as exc:
            try:
                self._connection.close()
            except Exception:
                pass
            raise CheckpointUnavailableError(
                f"Could not initialize LangGraph SQLite checkpointing at "
                f"{self.path}: {type(exc).__name__}: {exc}"
            ) from exc

        builder = StateGraph(RunCheckpointState)
        builder.add_node("persist", lambda state: {})
        builder.add_edge(START, "persist")
        builder.add_edge("persist", END)
        self.graph = builder.compile(checkpointer=self.checkpointer)
        self._closed = False

    @staticmethod
    def config(
        *,
        thread_id: str,
        checkpoint_ns: str,
    ) -> dict[str, Any]:
        storage_thread_id = f"{checkpoint_ns}:{thread_id}"
        return {
            "configurable": {
                "thread_id": storage_thread_id,
                # Non-empty checkpoint_ns is reserved by LangGraph for actual
                # compiled subgraphs. Odysseus keeps its stable logical
                # namespace in state and folds it into the storage thread ID.
                "checkpoint_ns": "",
            }
        }

    def save(self, state: RunCheckpointState) -> RunCheckpointState:
        config = self.config(
            thread_id=state["thread_id"],
            checkpoint_ns=state["checkpoint_namespace"],
        )
        result = self.graph.invoke(dict(state), config=config)
        return dict(result)

    def recover(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
    ) -> RunCheckpointState | None:
        snapshot = self.graph.get_state(
            self.config(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            )
        )
        values = dict(snapshot.values or {})
        return values or None

    def history(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
    ) -> list[RunCheckpointState]:
        return [
            dict(snapshot.values or {})
            for snapshot in self.graph.get_state_history(
                self.config(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                )
            )
            if snapshot.values
        ]

    def recover_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> RunCheckpointState | None:
        for checkpoint_tuple in self.checkpointer.list(
            None,
            limit=500,
        ):
            values = dict(
                checkpoint_tuple.checkpoint.get(
                    "channel_values",
                    {}
                )
            )
            if (
                session_id is not None
                and values.get("session_id") != session_id
            ):
                continue
            if (
                run_id is not None
                and values.get("run_id") != run_id
            ):
                continue
            if (
                values.get("thread_id")
                and values.get("checkpoint_namespace")
            ):
                return values
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()


_DEFAULT_STORE: DurableRunCheckpointStore | None = None
_DEFAULT_STORE_LOCK = threading.Lock()


def get_default_checkpoint_store() -> DurableRunCheckpointStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is not None:
        return _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = DurableRunCheckpointStore(
                default_checkpoint_path()
            )
            atexit.register(_DEFAULT_STORE.close)
    return _DEFAULT_STORE


__all__ = [
    "CheckpointUnavailableError",
    "DurableRunCheckpointStore",
    "RunCheckpointState",
    "default_checkpoint_path",
    "get_default_checkpoint_store",
]
