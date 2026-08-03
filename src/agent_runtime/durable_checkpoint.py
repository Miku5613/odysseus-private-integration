from __future__ import annotations

import atexit
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict

try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    _LANGGRAPH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # Optional runtime: Legacy must still import and start.
    JsonPlusSerializer = None  # type: ignore[assignment,misc]
    SqliteSaver = None  # type: ignore[assignment,misc]
    StateGraph = None  # type: ignore[assignment,misc]
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc
from src.agent_runtime.runtime_budgets import (
    RuntimeArtifactStore,
    apply_checkpoint_budgets,
)


class CheckpointUnavailableError(RuntimeError):
    pass


class CheckpointPersistenceError(RuntimeError):
    """An authoritative runtime state operation failed after initialization."""


class CheckpointSchemaError(RuntimeError):
    """A persisted runtime snapshot cannot be migrated safely."""


CURRENT_RUNTIME_SCHEMA_VERSION = 2
RUNTIME_EVENT_TAIL_LIMIT = 200
RUNTIME_REPLAY_PAGE_LIMIT = 500


class RunCheckpointState(TypedDict, total=False):
    schema_version: int
    run_id: str
    owner: str | None
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
    owned_research_jobs: dict[str, dict[str, Any]]
    owned_children: dict[str, dict[str, Any]]
    failure: dict[str, Any] | None
    event_cursor: int
    event_log: list[dict[str, Any]]
    current_plan: dict[str, Any] | None
    terminal: dict[str, Any] | None


ACTIVE_RUN_STATUSES = frozenset({
    "PENDING",
    "RUNNING",
    "AWAITING_APPROVAL",
    "AWAITING_CLARIFICATION",
    "SUSPENDED",
    "INTERRUPTED",
})
TERMINAL_RUN_STATUSES = frozenset({
    "COMPLETED",
    "FAILED",
    "CANCELLED",
})
RESUME_OUTCOMES = frozenset({
    "SAFE_TO_RESTART",
    "ALREADY_COMPLETED",
    "REQUIRES_VERIFICATION",
    "REQUIRES_APPROVAL",
    "NOT_RESUMABLE",
})
SIDE_EFFECT_RECEIPTS = frozenset({
    "no_side_effect",
    "planned_not_started",
    "started_unknown",
    "confirmed_completed",
    "confirmed_failed",
    "externally_verifiable",
    "not_safely_repeatable",
})


def default_checkpoint_path() -> Path:
    configured = os.getenv("ODYSSEUS_LANGGRAPH_CHECKPOINT_DB", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "langgraph" / "checkpoints.sqlite3"


class DurableRunCheckpointStore:
    """SQLite-backed LangGraph graph containing authoritative run snapshots."""

    def __init__(self, path: str | Path) -> None:
        if _LANGGRAPH_IMPORT_ERROR is not None:
            raise CheckpointUnavailableError(
                "LangGraph runtime dependencies are not installed. Run "
                "Express Installation.bat and approve the optional LangGraph "
                f"runtime. ({type(_LANGGRAPH_IMPORT_ERROR).__name__}: "
                f"{_LANGGRAPH_IMPORT_ERROR})"
            ) from _LANGGRAPH_IMPORT_ERROR
        self.path = Path(path).expanduser().resolve()
        artifact_root = os.getenv(
            "ODYSSEUS_LANGGRAPH_ARTIFACT_DIR",
            str(self.path.parent / "artifacts"),
        )
        self.artifact_store = RuntimeArtifactStore(artifact_root)
        self.process_id = f"process_{uuid.uuid4().hex}"
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            serializer = JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=None,
            )
            self.checkpointer = SqliteSaver(
                self._connection,
                serde=serializer,
            )
            self.checkpointer.setup()
            self._setup_runtime_tables()
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

    def _setup_runtime_tables(self) -> None:
        """Create Odysseus-owned indexes beside the LangGraph checkpoint tables."""

        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS odysseus_runtime_runs (
                run_id TEXT PRIMARY KEY,
                owner TEXT,
                session_id TEXT,
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                status TEXT NOT NULL,
                executor_id TEXT,
                executor_process_id TEXT,
                lease_expires_at REAL,
                updated_at REAL NOT NULL,
                UNIQUE(thread_id, checkpoint_ns)
            );
            CREATE INDEX IF NOT EXISTS idx_odysseus_runs_session_owner
                ON odysseus_runtime_runs(session_id, owner, updated_at DESC);
            CREATE TABLE IF NOT EXISTS odysseus_step_attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                plan_revision INTEGER NOT NULL,
                step_id TEXT NOT NULL,
                worker_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                input_refs_json TEXT NOT NULL,
                started_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                side_effect_classification TEXT NOT NULL,
                receipt_status TEXT NOT NULL,
                action_receipts_json TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                resumability TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_odysseus_attempts_run_step
                ON odysseus_step_attempts(run_id, step_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS odysseus_runtime_decisions (
                record_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                run_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                step_id TEXT,
                attempt_id TEXT,
                action_text TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                state TEXT NOT NULL,
                consumed_at REAL,
                response_json TEXT,
                capability TEXT,
                tool_metadata_json TEXT NOT NULL,
                side_effect_risk TEXT NOT NULL,
                UNIQUE(kind, session_id, fingerprint)
            );
            CREATE INDEX IF NOT EXISTS idx_odysseus_decisions_scope
                ON odysseus_runtime_decisions(
                    kind, session_id, owner, run_id, state, expires_at
                );
            CREATE TABLE IF NOT EXISTS odysseus_runtime_events (
                run_id TEXT NOT NULL,
                event_cursor INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY(run_id, event_cursor)
            );
            CREATE INDEX IF NOT EXISTS idx_odysseus_events_run_cursor
                ON odysseus_runtime_events(run_id, event_cursor);
            """
        )
        self._connection.commit()

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
        migrated = self._migrate_state(dict(state), for_write=True)
        migrated = apply_checkpoint_budgets(
            migrated,
            artifact_store=self.artifact_store,
        )
        config = self.config(
            thread_id=migrated["thread_id"],
            checkpoint_ns=migrated["checkpoint_namespace"],
        )
        with self._lock:
            try:
                self._persist_events(migrated)
                compacted = dict(migrated)
                compacted["event_log"] = list(
                    migrated.get("event_log") or []
                )[-RUNTIME_EVENT_TAIL_LIMIT:]
                result = self.graph.invoke(compacted, config=config)
                self._upsert_run(dict(result))
                return dict(result)
            except CheckpointSchemaError:
                raise
            except (sqlite3.Error, OSError, RuntimeError) as exc:
                try:
                    self._connection.rollback()
                except sqlite3.Error:
                    pass
                raise CheckpointPersistenceError(
                    "Authoritative LangGraph checkpoint save failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    def _migrate_state(
        self,
        state: dict[str, Any],
        *,
        for_write: bool = False,
    ) -> RunCheckpointState:
        if not isinstance(state, dict):
            raise CheckpointSchemaError("Runtime checkpoint payload is not an object.")
        raw_version = state.get("schema_version")
        if raw_version is None:
            version = 1
        elif isinstance(raw_version, bool):
            raise CheckpointSchemaError("Runtime checkpoint schema version is malformed.")
        else:
            try:
                version = int(raw_version)
            except (TypeError, ValueError) as exc:
                raise CheckpointSchemaError(
                    "Runtime checkpoint schema version is malformed."
                ) from exc
        if version > CURRENT_RUNTIME_SCHEMA_VERSION:
            raise CheckpointSchemaError(
                "Runtime checkpoint schema version "
                f"{version} is newer than supported version "
                f"{CURRENT_RUNTIME_SCHEMA_VERSION}."
            )
        if version < 1:
            raise CheckpointSchemaError(
                f"Unsupported runtime checkpoint schema version: {version}."
            )
        migrated = dict(state)
        required = ("run_id", "thread_id", "checkpoint_namespace")
        if not for_write and any(not migrated.get(key) for key in required):
            raise CheckpointSchemaError(
                "Runtime checkpoint is missing required identity fields."
            )
        if version == 1:
            migrated.setdefault("owner", None)
            migrated.setdefault("artifact_references", [])
            migrated.setdefault("owned_research_jobs", {})
            migrated.setdefault("owned_children", {})
            migrated.setdefault("dependency_results", {})
            migrated.setdefault("completed_steps", [])
            migrated.setdefault("event_log", [])
        migrated["schema_version"] = CURRENT_RUNTIME_SCHEMA_VERSION
        return migrated  # type: ignore[return-value]

    def _persist_events(self, state: RunCheckpointState) -> None:
        run_id = str(state["run_id"])
        rows: list[tuple[str, int, str, float]] = []
        now = time.time()
        for event in state.get("event_log") or []:
            if not isinstance(event, dict):
                continue
            cursor = int((event.get("data") or {}).get("event_cursor") or 0)
            if cursor <= 0:
                continue
            rows.append((run_id, cursor, json.dumps(event, default=str), now))
        if rows:
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO odysseus_runtime_events (
                    run_id, event_cursor, event_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def events_after(
        self,
        *,
        run_id: str,
        after_cursor: int = 0,
        limit: int = RUNTIME_REPLAY_PAGE_LIMIT,
    ) -> list[dict[str, Any]]:
        page_limit = max(1, min(int(limit), RUNTIME_REPLAY_PAGE_LIMIT))
        try:
            rows = self._connection.execute(
                """
                SELECT event_json FROM odysseus_runtime_events
                WHERE run_id=? AND event_cursor>?
                ORDER BY event_cursor ASC LIMIT ?
                """,
                (run_id, max(0, int(after_cursor)), page_limit),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise CheckpointPersistenceError(
                "Durable LangGraph event replay failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def cleanup_completed(self, *, retention_days: int | None = None) -> int:
        """Delete only terminal run data older than the configured retention."""

        if retention_days is None:
            from src.agent_runtime.runtime_settings import (
                get_langgraph_runtime_settings,
            )
            retention_days = int(
                get_langgraph_runtime_settings()["completed_run_retention"]
            )
        cutoff = time.time() - max(1, int(retention_days)) * 86400
        with self._lock:
            try:
                rows = self._connection.execute(
                    """
                    SELECT run_id FROM odysseus_runtime_runs
                    WHERE status IN ('COMPLETED','FAILED','CANCELLED')
                      AND updated_at<?
                    """,
                    (cutoff,),
                ).fetchall()
                run_ids = [row[0] for row in rows]
                for run_id in run_ids:
                    self._connection.execute(
                        "DELETE FROM odysseus_runtime_events WHERE run_id=?",
                        (run_id,),
                    )
                    self._connection.execute(
                        "DELETE FROM odysseus_step_attempts WHERE run_id=?",
                        (run_id,),
                    )
                    self._connection.execute(
                        "DELETE FROM odysseus_runtime_decisions WHERE run_id=?",
                        (run_id,),
                    )
                    self._connection.execute(
                        "DELETE FROM odysseus_runtime_runs WHERE run_id=?",
                        (run_id,),
                    )
                self._connection.commit()
                return len(run_ids)
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise CheckpointPersistenceError(
                    "Completed LangGraph retention cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    def _upsert_run(self, state: RunCheckpointState) -> None:
        now = time.time()
        run_id = str(state["run_id"])
        status = str(state.get("status") or "PENDING").upper()
        existing = self._connection.execute(
            "SELECT status, executor_id, executor_process_id, lease_expires_at "
            "FROM odysseus_runtime_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing and existing[0] in TERMINAL_RUN_STATUSES:
            status = existing[0]
        self._connection.execute(
            """
            INSERT INTO odysseus_runtime_runs (
                run_id, owner, session_id, thread_id, checkpoint_ns, status,
                executor_id, executor_process_id, lease_expires_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                owner=excluded.owner,
                session_id=excluded.session_id,
                thread_id=excluded.thread_id,
                checkpoint_ns=excluded.checkpoint_ns,
                status=CASE
                    WHEN odysseus_runtime_runs.status IN
                        ('COMPLETED','FAILED','CANCELLED')
                    THEN odysseus_runtime_runs.status
                    ELSE excluded.status
                END,
                updated_at=excluded.updated_at
            """,
            (
                run_id,
                state.get("owner"),
                state.get("session_id"),
                state["thread_id"],
                state["checkpoint_namespace"],
                status,
                existing[1] if existing else None,
                existing[2] if existing else None,
                existing[3] if existing else None,
                now,
            ),
        )
        self._connection.commit()

    def register_run_identity(
        self,
        *,
        run_id: str,
        owner: str | None,
        session_id: str | None,
        thread_id: str,
        checkpoint_ns: str,
        status: str = "PENDING",
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO odysseus_runtime_runs (
                    run_id, owner, session_id, thread_id, checkpoint_ns, status,
                    executor_id, executor_process_id, lease_expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    owner=excluded.owner,
                    session_id=excluded.session_id,
                    thread_id=excluded.thread_id,
                    checkpoint_ns=excluded.checkpoint_ns,
                    status=CASE
                        WHEN odysseus_runtime_runs.status IN
                            ('COMPLETED','FAILED','CANCELLED')
                        THEN odysseus_runtime_runs.status
                        ELSE excluded.status
                    END,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id, owner, session_id, thread_id, checkpoint_ns,
                    str(status).upper(), time.time(),
                ),
            )
            self._connection.commit()

    def claim_executor(
        self,
        *,
        run_id: str,
        executor_id: str,
        lease_seconds: float = 120.0,
    ) -> None:
        now = time.time()
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM odysseus_runtime_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row and row[0] in TERMINAL_RUN_STATUSES:
                return
            self._connection.execute(
                """
                UPDATE odysseus_runtime_runs
                SET executor_id=?, executor_process_id=?, lease_expires_at=?,
                    status='RUNNING', updated_at=?
                WHERE run_id=?
                """,
                (
                    executor_id,
                    self.process_id,
                    now + max(1.0, float(lease_seconds)),
                    now,
                    run_id,
                ),
            )
            self._connection.commit()

    def heartbeat_executor(
        self,
        *,
        run_id: str,
        executor_id: str,
        lease_seconds: float = 120.0,
    ) -> bool:
        now = time.time()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE odysseus_runtime_runs
                SET lease_expires_at=?, updated_at=?
                WHERE run_id=? AND executor_id=? AND executor_process_id=?
                    AND status NOT IN ('COMPLETED','FAILED','CANCELLED')
                """,
                (
                    now + max(1.0, float(lease_seconds)),
                    now,
                    run_id,
                    executor_id,
                    self.process_id,
                ),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def release_executor(
        self,
        *,
        run_id: str,
        executor_id: str | None = None,
    ) -> None:
        with self._lock:
            if executor_id is None:
                self._connection.execute(
                    """
                    UPDATE odysseus_runtime_runs
                    SET executor_id=NULL, executor_process_id=NULL,
                        lease_expires_at=NULL, updated_at=?
                    WHERE run_id=?
                    """,
                    (time.time(), run_id),
                )
            else:
                self._connection.execute(
                    """
                    UPDATE odysseus_runtime_runs
                    SET executor_id=NULL, executor_process_id=NULL,
                        lease_expires_at=NULL, updated_at=?
                    WHERE run_id=? AND executor_id=?
                    """,
                    (time.time(), run_id, executor_id),
                )
            self._connection.commit()

    def set_run_status(self, run_id: str, status: str) -> str:
        normalized = str(status).upper()
        if normalized not in ACTIVE_RUN_STATUSES | TERMINAL_RUN_STATUSES:
            raise ValueError(f"Unsupported runtime status: {status}")
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM odysseus_runtime_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown runtime run: {run_id}")
            if row[0] in TERMINAL_RUN_STATUSES:
                return row[0]
            self._connection.execute(
                """
                UPDATE odysseus_runtime_runs
                SET status=?, updated_at=?,
                    executor_id=CASE WHEN ? IN
                        ('COMPLETED','FAILED','CANCELLED','SUSPENDED','INTERRUPTED')
                        THEN NULL ELSE executor_id END,
                    executor_process_id=CASE WHEN ? IN
                        ('COMPLETED','FAILED','CANCELLED','SUSPENDED','INTERRUPTED')
                        THEN NULL ELSE executor_process_id END,
                    lease_expires_at=CASE WHEN ? IN
                        ('COMPLETED','FAILED','CANCELLED','SUSPENDED','INTERRUPTED')
                        THEN NULL ELSE lease_expires_at END
                WHERE run_id=?
                """,
                (normalized, time.time(), normalized, normalized, normalized, run_id),
            )
            self._connection.commit()
            return normalized

    def run_record(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT run_id, owner, session_id, thread_id, checkpoint_ns, status,
                   executor_id, executor_process_id, lease_expires_at, updated_at
            FROM odysseus_runtime_runs WHERE run_id=?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "run_id", "owner", "session_id", "thread_id", "checkpoint_ns",
            "status", "executor_id", "executor_process_id",
            "lease_expires_at", "updated_at",
        )
        return dict(zip(keys, row))

    def latest_run_record(
        self,
        *,
        session_id: str,
        owner: str | None = None,
    ) -> dict[str, Any] | None:
        if owner is None:
            row = self._connection.execute(
                """
                SELECT run_id FROM odysseus_runtime_runs
                WHERE session_id=? ORDER BY updated_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT run_id FROM odysseus_runtime_runs
                WHERE session_id=? AND owner=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (session_id, owner),
            ).fetchone()
        return self.run_record(row[0]) if row else None

    def reconcile_state(
        self,
        state: RunCheckpointState,
        *,
        executor_alive: bool = False,
    ) -> RunCheckpointState:
        """Return authoritative state after reconciling a persisted executor."""

        values = dict(state)
        run_id = str(values["run_id"])
        record = self.run_record(run_id)
        status = str(
            (record or {}).get("status") or values.get("status") or "PENDING"
        ).upper()
        if status in TERMINAL_RUN_STATUSES:
            values["status"] = status
            return values
        if executor_alive:
            values["status"] = "RUNNING"
            return values
        if status == "AWAITING_APPROVAL":
            if self.has_pending_decision(run_id=run_id, kind="approval"):
                values["status"] = status
                return values
        if status == "AWAITING_CLARIFICATION":
            if self.has_pending_decision(run_id=run_id, kind="clarification"):
                values["status"] = status
                return values
        now = time.time()
        same_process = bool(
            record
            and record.get("executor_process_id") == self.process_id
        )
        lease_live = bool(
            record
            and record.get("lease_expires_at")
            and float(record["lease_expires_at"]) > now
        )
        if status == "RUNNING" and same_process and lease_live:
            values["status"] = "RUNNING"
            return values
        if status in {
            "PENDING", "RUNNING", "AWAITING_APPROVAL",
            "AWAITING_CLARIFICATION",
        }:
            status = "INTERRUPTED"
            self.set_run_status(run_id, status)
        values["status"] = status
        values["executor_present"] = False
        return values

    def create_step_attempt(
        self,
        *,
        run_id: str,
        plan_id: str,
        plan_revision: int,
        step_id: str,
        worker_kind: str,
        input_refs: dict[str, Any],
        side_effect_classification: str,
    ) -> dict[str, Any]:
        now = time.time()
        attempt_id = f"attempt_{uuid.uuid4().hex}"
        classification = str(side_effect_classification)
        read_only = classification in {"none", "read_only"}
        receipt = "no_side_effect" if read_only else "started_unknown"
        resumability = (
            "SAFE_TO_RESTART" if read_only else "REQUIRES_VERIFICATION"
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO odysseus_step_attempts (
                    attempt_id, run_id, plan_id, plan_revision, step_id,
                    worker_kind, status, input_refs_json, started_at, updated_at,
                    completed_at, side_effect_classification, receipt_status,
                    action_receipts_json, result_json, error, resumability
                ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', ?, ?, ?, NULL, ?, ?, '[]',
                          NULL, NULL, ?)
                """,
                (
                    attempt_id, run_id, plan_id, int(plan_revision), step_id,
                    worker_kind, json.dumps(input_refs, default=str),
                    now, now, classification, receipt, resumability,
                ),
            )
            self._connection.commit()
        return self.step_attempt(attempt_id) or {}

    def complete_step_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        result: Any = None,
        error: str | None = None,
        action_receipts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized = str(status).upper()
        current = self.step_attempt(attempt_id)
        if current is None:
            raise KeyError(f"Unknown step attempt: {attempt_id}")
        if current["status"] == "COMPLETED":
            return current
        receipts = list(action_receipts or [])
        external_id = any(
            isinstance(item, dict) and item.get("external_id")
            for item in receipts
        )
        if normalized == "COMPLETED":
            receipt = "confirmed_completed"
            resumability = "ALREADY_COMPLETED"
        elif normalized == "PARTIAL":
            receipt = "confirmed_partial"
            resumability = "NOT_RESUMABLE"
        elif normalized == "FAILED":
            receipt = "confirmed_failed"
            resumability = (
                "SAFE_TO_RESTART"
                if current["side_effect_classification"] in {"none", "read_only"}
                else "REQUIRES_VERIFICATION"
            )
        elif external_id:
            receipt = "externally_verifiable"
            resumability = "REQUIRES_VERIFICATION"
        else:
            receipt = current["receipt_status"]
            resumability = current["resumability"]
        completed_at = (
            time.time()
            if normalized in {"COMPLETED", "PARTIAL", "FAILED"}
            else None
        )
        with self._lock:
            self._connection.execute(
                """
                UPDATE odysseus_step_attempts
                SET status=?, updated_at=?, completed_at=?, receipt_status=?,
                    action_receipts_json=?, result_json=?, error=?, resumability=?
                WHERE attempt_id=?
                """,
                (
                    normalized, time.time(), completed_at, receipt,
                    json.dumps(receipts, default=str),
                    json.dumps(result, default=str) if result is not None else None,
                    str(error) if error else None, resumability, attempt_id,
                ),
            )
            self._connection.commit()
        return self.step_attempt(attempt_id) or {}

    def step_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT attempt_id, run_id, plan_id, plan_revision, step_id,
                   worker_kind, status, input_refs_json, started_at, updated_at,
                   completed_at, side_effect_classification, receipt_status,
                   action_receipts_json, result_json, error, resumability
            FROM odysseus_step_attempts WHERE attempt_id=?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "attempt_id", "run_id", "plan_id", "plan_revision", "step_id",
            "worker_kind", "status", "input_refs_json", "started_at",
            "updated_at", "completed_at", "side_effect_classification",
            "receipt_status", "action_receipts_json", "result_json", "error",
            "resumability",
        )
        item = dict(zip(keys, row))
        for key in ("input_refs_json", "action_receipts_json", "result_json"):
            raw = item.pop(key)
            item[key.removesuffix("_json")] = (
                json.loads(raw) if raw is not None else None
            )
        return item

    def latest_step_attempt(
        self,
        *,
        run_id: str,
        step_id: str | None = None,
    ) -> dict[str, Any] | None:
        if step_id is None:
            row = self._connection.execute(
                """
                SELECT attempt_id FROM odysseus_step_attempts
                WHERE run_id=? ORDER BY updated_at DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT attempt_id FROM odysseus_step_attempts
                WHERE run_id=? AND step_id=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (run_id, step_id),
            ).fetchone()
        return self.step_attempt(row[0]) if row else None

    def create_decision(
        self,
        *,
        kind: str,
        run_id: str,
        session_id: str,
        owner: str,
        action_text: str,
        fingerprint: str,
        step_id: str | None = None,
        attempt_id: str | None = None,
        expires_in_seconds: float = 3600.0,
        capability: str | None = None,
        tool_metadata: dict[str, Any] | None = None,
        side_effect_risk: str = "guided",
    ) -> dict[str, Any]:
        normalized_kind = str(kind).lower()
        if normalized_kind not in {"approval", "clarification"}:
            raise ValueError(f"Unsupported decision kind: {kind}")
        now = time.time()
        record_id = f"decision_{uuid.uuid4().hex}"
        with self._lock:
            existing = self._connection.execute(
                """
                SELECT record_id FROM odysseus_runtime_decisions
                WHERE kind=? AND session_id=? AND fingerprint=?
                """,
                (normalized_kind, session_id, fingerprint),
            ).fetchone()
            if existing:
                return self.decision_record(existing[0]) or {}
            self._connection.execute(
                """
                INSERT INTO odysseus_runtime_decisions (
                    record_id, kind, run_id, session_id, owner, step_id,
                    attempt_id, action_text, fingerprint, created_at, expires_at,
                    state, consumed_at, response_json, capability,
                    tool_metadata_json, side_effect_risk
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL,
                          NULL, ?, ?, ?)
                """,
                (
                    record_id, normalized_kind, run_id, session_id, owner,
                    step_id, attempt_id, action_text, fingerprint, now,
                    now + max(1.0, float(expires_in_seconds)), capability,
                    json.dumps(tool_metadata or {}, default=str),
                    side_effect_risk,
                ),
            )
            self._connection.commit()
        if self.run_record(run_id) is not None:
            self.set_run_status(
                run_id,
                (
                    "AWAITING_APPROVAL"
                    if normalized_kind == "approval"
                    else "AWAITING_CLARIFICATION"
                ),
            )
        return self.decision_record(record_id) or {}

    def decision_record(self, record_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT record_id, kind, run_id, session_id, owner, step_id,
                   attempt_id, action_text, fingerprint, created_at, expires_at,
                   state, consumed_at, response_json, capability,
                   tool_metadata_json, side_effect_risk
            FROM odysseus_runtime_decisions WHERE record_id=?
            """,
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "record_id", "kind", "run_id", "session_id", "owner", "step_id",
            "attempt_id", "action_text", "fingerprint", "created_at",
            "expires_at", "state", "consumed_at", "response_json",
            "capability", "tool_metadata_json", "side_effect_risk",
        )
        item = dict(zip(keys, row))
        item["response"] = (
            json.loads(item.pop("response_json"))
            if item["response_json"] is not None
            else None
        )
        item["tool_metadata"] = json.loads(item.pop("tool_metadata_json"))
        return item

    def find_decision(
        self,
        *,
        kind: str,
        session_id: str,
        fingerprint: str | None = None,
        owner: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["kind=?", "session_id=?"]
        params: list[Any] = [str(kind).lower(), session_id]
        if fingerprint is not None:
            clauses.append("fingerprint=?")
            params.append(fingerprint)
        if owner is not None:
            clauses.append("owner=?")
            params.append(owner)
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(run_id)
        row = self._connection.execute(
            "SELECT record_id FROM odysseus_runtime_decisions WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT 1",
            tuple(params),
        ).fetchone()
        return self.decision_record(row[0]) if row else None

    def decision_for_tool_call(
        self,
        *,
        session_id: str,
        call_id: str,
    ) -> dict[str, Any] | None:
        rows = self._connection.execute(
            """
            SELECT record_id FROM odysseus_runtime_decisions
            WHERE kind='approval' AND session_id=?
            ORDER BY created_at DESC
            """,
            (session_id,),
        ).fetchall()
        for row in rows:
            record = self.decision_record(row[0])
            if (
                record is not None
                and str(record["tool_metadata"].get("call_id") or "") == call_id
            ):
                return record
        return None

    def delete_session_decisions(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> int:
        with self._lock:
            if session_id is None and kind is None:
                cursor = self._connection.execute(
                    "DELETE FROM odysseus_runtime_decisions"
                )
            elif session_id is not None and kind is None:
                cursor = self._connection.execute(
                    "DELETE FROM odysseus_runtime_decisions WHERE session_id=?",
                    (session_id,),
                )
            elif session_id is None:
                cursor = self._connection.execute(
                    "DELETE FROM odysseus_runtime_decisions WHERE kind=?",
                    (str(kind).lower(),),
                )
            else:
                cursor = self._connection.execute(
                    """
                    DELETE FROM odysseus_runtime_decisions
                    WHERE session_id=? AND kind=?
                    """,
                    (session_id, str(kind).lower()),
                )
            self._connection.commit()
            return int(cursor.rowcount)

    def has_pending_decision(self, *, run_id: str, kind: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM odysseus_runtime_decisions
            WHERE run_id=? AND kind=? AND state IN ('pending','approved')
                AND expires_at>? LIMIT 1
            """,
            (run_id, str(kind).lower(), time.time()),
        ).fetchone()
        return row is not None

    def resolve_decision(
        self,
        *,
        record_id: str,
        owner: str,
        session_id: str,
        run_id: str,
        response: Any,
        state: str,
    ) -> dict[str, Any]:
        normalized = str(state).lower()
        if normalized not in {
            "approved", "rejected", "answered", "cancelled", "consumed",
        }:
            raise ValueError(f"Unsupported decision state: {state}")
        with self._lock:
            record = self.decision_record(record_id)
            if record is None:
                raise KeyError("pending decision not found")
            if (
                record["owner"] != owner
                or record["session_id"] != session_id
                or record["run_id"] != run_id
            ):
                raise PermissionError("decision ownership mismatch")
            if record["expires_at"] <= time.time():
                self._connection.execute(
                    "UPDATE odysseus_runtime_decisions SET state='expired' "
                    "WHERE record_id=? AND state IN ('pending','approved')",
                    (record_id,),
                )
                self._connection.commit()
                raise ValueError("decision has expired")
            if record["state"] == normalized:
                return record
            if record["state"] not in {"pending", "approved"}:
                raise ValueError("decision has already been resolved")
            self._connection.execute(
                """
                UPDATE odysseus_runtime_decisions
                SET state=?, response_json=?, consumed_at=?
                WHERE record_id=?
                """,
                (
                    normalized,
                    json.dumps(response, default=str),
                    time.time(),
                    record_id,
                ),
            )
            self._connection.commit()
        if normalized in {"consumed", "answered"}:
            try:
                self.set_run_status(run_id, "SUSPENDED")
            except KeyError:
                pass
        elif normalized in {"rejected", "cancelled"}:
            try:
                self.set_run_status(run_id, "INTERRUPTED")
            except KeyError:
                pass
        return self.decision_record(record_id) or {}

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
        if not values:
            return None
        migrated = self._migrate_state(values)
        migrated["event_log"] = self.events_after(
            run_id=str(migrated["run_id"]),
            after_cursor=max(
                0,
                int(migrated.get("event_cursor") or 0)
                - RUNTIME_EVENT_TAIL_LIMIT,
            ),
            limit=RUNTIME_EVENT_TAIL_LIMIT,
        ) or list(migrated.get("event_log") or [])
        return migrated

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
        owner: str | None = None,
    ) -> RunCheckpointState | None:
        if run_id is not None:
            record = self.run_record(run_id)
            if record is None:
                return None
            if session_id is not None and record.get("session_id") != session_id:
                return None
            if owner is not None and record.get("owner") != owner:
                return None
        elif session_id is not None:
            record = self.latest_run_record(session_id=session_id, owner=owner)
        else:
            raise ValueError("run_id or session_id is required for checkpoint lookup")
        if record is None:
            return None
        recovered = self.recover(
            thread_id=str(record["thread_id"]),
            checkpoint_ns=str(record["checkpoint_ns"]),
        )
        if recovered is not None:
            recovered["status"] = str(record["status"])
        return recovered

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


def reset_default_checkpoint_store_for_tests() -> None:
    """Close and forget the singleton so tests can simulate process reopen."""

    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is not None:
            _DEFAULT_STORE.close()
        _DEFAULT_STORE = None


__all__ = [
    "CURRENT_RUNTIME_SCHEMA_VERSION",
    "CheckpointPersistenceError",
    "CheckpointSchemaError",
    "CheckpointUnavailableError",
    "DurableRunCheckpointStore",
    "RunCheckpointState",
    "default_checkpoint_path",
    "get_default_checkpoint_store",
    "reset_default_checkpoint_store_for_tests",
]
