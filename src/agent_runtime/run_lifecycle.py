from __future__ import annotations

import asyncio
import hashlib
import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


FINAL_STATUSES = frozenset({
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "AWAITING_APPROVAL",
})


@dataclass(frozen=True)
class LangGraphRunIdentity:
    run_id: str
    session_id: str | None
    thread_id: str
    checkpoint_ns: str
    runtime_mode: str = "multi_graph"

    @classmethod
    def create(
        cls,
        *,
        session_id: str | None,
        run_id: str | None = None,
        thread_id: str | None = None,
        checkpoint_ns: str | None = None,
        owner: str | None = None,
        runtime_mode: str = "multi_graph",
    ) -> "LangGraphRunIdentity":
        stable_run_id = str(run_id or f"lg_{uuid.uuid4().hex}").strip()
        if not stable_run_id:
            raise ValueError("run_id must not be empty")
        stable_session_id = (
            str(session_id).strip()
            if session_id is not None
            else None
        )
        stable_thread_id = str(
            thread_id
            or (
                f"odysseus:{stable_session_id or 'anonymous'}:"
                f"{stable_run_id}"
            )
        ).strip()
        if not stable_thread_id:
            raise ValueError("thread_id must not be empty")
        if checkpoint_ns:
            stable_checkpoint_ns = str(checkpoint_ns).strip()
        else:
            isolation_key = "|".join((
                str(owner or "anonymous"),
                stable_session_id or "anonymous",
                stable_run_id,
            ))
            isolation_hash = hashlib.sha256(
                isolation_key.encode("utf-8")
            ).hexdigest()
            stable_checkpoint_ns = (
                f"odysseus/{runtime_mode}/{isolation_hash}"
            )
        if not stable_checkpoint_ns:
            raise ValueError("checkpoint_ns must not be empty")
        return cls(
            run_id=stable_run_id,
            session_id=stable_session_id,
            thread_id=stable_thread_id,
            checkpoint_ns=stable_checkpoint_ns,
            runtime_mode=runtime_mode,
        )


class LangGraphCancellationScope:
    """Own only tasks and cancellation hooks created for one LangGraph run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._cancelled = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._callbacks: dict[str, Callable[[], Any]] = {}

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def own_task(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        if self._cancelled:
            task.cancel()
            return task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def own_cancellation(
        self,
        owned_id: str,
        callback: Callable[[], Any],
    ) -> None:
        key = str(owned_id).strip()
        if not key:
            raise ValueError("owned cancellation ID must not be empty")
        if self._cancelled:
            result = callback()
            if inspect.isawaitable(result):
                asyncio.create_task(result)
            return
        self._callbacks[key] = callback

    def release(self, owned_id: str) -> None:
        self._callbacks.pop(str(owned_id), None)

    def cancel(self) -> bool:
        if self._cancelled:
            return False
        self._cancelled = True

        for task in tuple(self._tasks):
            if not task.done():
                task.cancel()

        callbacks = tuple(self._callbacks.values())
        self._callbacks.clear()
        for callback in callbacks:
            result = callback()
            if inspect.isawaitable(result):
                asyncio.create_task(result)
        return True


@dataclass
class LangGraphRunLifecycle:
    identity: LangGraphRunIdentity
    cancellation: LangGraphCancellationScope
    status: str = "RUNNING"
    event_cursor: int = 0
    terminal: dict[str, Any] | None = field(default=None, init=False)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    current_plan: dict[str, Any] | None = None
    completed_steps: list[str] = field(default_factory=list)
    dependency_results: dict[str, Any] = field(default_factory=dict)
    artifact_references: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_store: Any = None

    @classmethod
    def create(
        cls,
        *,
        session_id: str | None,
        run_id: str | None = None,
        thread_id: str | None = None,
        checkpoint_ns: str | None = None,
        owner: str | None = None,
        runtime_mode: str = "multi_graph",
        cancellation: LangGraphCancellationScope | None = None,
        checkpoint_store: Any = None,
    ) -> "LangGraphRunLifecycle":
        identity = LangGraphRunIdentity.create(
            session_id=session_id,
            run_id=run_id,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            owner=owner,
            runtime_mode=runtime_mode,
        )
        scope = cancellation or LangGraphCancellationScope(identity.run_id)
        if scope.run_id != identity.run_id:
            raise ValueError(
                "cancellation scope run_id must match lifecycle run_id"
            )
        return cls(
            identity=identity,
            cancellation=scope,
            checkpoint_store=checkpoint_store,
        )

    def _decorate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.event_cursor += 1
        framed = dict(payload)
        data = dict(framed.get("data") or {})
        data.update({
            "run_id": self.identity.run_id,
            "session_id": self.identity.session_id,
            "thread_id": self.identity.thread_id,
            "checkpoint_ns": self.identity.checkpoint_ns,
            "runtime_mode": self.identity.runtime_mode,
            "event_cursor": self.event_cursor,
        })
        framed["data"] = data
        return framed

    def _checkpoint_state(self) -> dict[str, Any]:
        terminal_data = (
            dict(self.terminal.get("data") or {})
            if self.terminal is not None
            else None
        )
        return {
            "run_id": self.identity.run_id,
            "session_id": self.identity.session_id,
            "thread_id": self.identity.thread_id,
            "checkpoint_namespace": self.identity.checkpoint_ns,
            "runtime_mode": self.identity.runtime_mode,
            "status": self.status,
            "cancellation_scope": self.cancellation.run_id,
            "completed_steps": list(self.completed_steps),
            "dependency_results": dict(self.dependency_results),
            "artifact_references": list(self.artifact_references),
            "failure": (
                {
                    "stage": terminal_data.get("stage"),
                    "reason": terminal_data.get("reason"),
                }
                if terminal_data
                and terminal_data.get("status") == "FAILED"
                else None
            ),
            "event_cursor": self.event_cursor,
            "event_log": list(self.event_log),
            "current_plan": (
                dict(self.current_plan)
                if self.current_plan is not None
                else None
            ),
            "terminal": terminal_data,
        }

    def _persist(self) -> None:
        if self.checkpoint_store is not None:
            self.checkpoint_store.save(self._checkpoint_state())

    def event(self, payload: dict[str, Any]) -> dict[str, Any]:
        framed = self._decorate(payload)
        self.event_log.append(framed)
        if framed.get("type") == "plan_update":
            self.current_plan = dict(framed.get("data") or {})
        self._persist()
        return framed

    def update_runtime_state(
        self,
        *,
        completed_steps: list[str],
        dependency_results: dict[str, Any],
        artifact_references: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.completed_steps = list(completed_steps)
        self.dependency_results = dict(dependency_results)
        self.artifact_references = list(artifact_references or [])
        return self.event({
            "type": "runtime_snapshot",
            "data": {
                "status": self.status,
                "completed_steps": self.completed_steps,
                "dependency_results": self.dependency_results,
                "artifact_references": self.artifact_references,
            },
        })

    def finish(
        self,
        status: str,
        *,
        stage: str,
        reason: str | None = None,
        **details: Any,
    ) -> dict[str, Any] | None:
        normalized = str(status).upper()
        if normalized not in FINAL_STATUSES:
            raise ValueError(f"Unsupported terminal status: {status}")
        if self.terminal is not None:
            return None
        self.status = normalized
        self.terminal = self._decorate({
            "type": "runtime_terminal",
            "data": {
                "status": normalized,
                "stage": stage,
                "reason": reason,
                **details,
            },
        })
        self.event_log.append(self.terminal)
        self._persist()
        return self.terminal

    @classmethod
    def recover(
        cls,
        *,
        checkpoint_store: Any,
        thread_id: str,
        checkpoint_ns: str,
    ) -> "LangGraphRunLifecycle | None":
        state = checkpoint_store.recover(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
        )
        if not state:
            return None
        lifecycle = cls.create(
            session_id=state.get("session_id"),
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            checkpoint_ns=state["checkpoint_namespace"],
            runtime_mode=state.get("runtime_mode") or "multi_graph",
            checkpoint_store=checkpoint_store,
        )
        lifecycle.status = state.get("status") or "RUNNING"
        lifecycle.event_cursor = int(state.get("event_cursor") or 0)
        lifecycle.event_log = list(state.get("event_log") or [])
        lifecycle.current_plan = state.get("current_plan")
        lifecycle.completed_steps = list(
            state.get("completed_steps") or []
        )
        lifecycle.dependency_results = dict(
            state.get("dependency_results") or {}
        )
        lifecycle.artifact_references = list(
            state.get("artifact_references") or []
        )
        terminal = state.get("terminal")
        if terminal:
            lifecycle.terminal = {
                "type": "runtime_terminal",
                "data": dict(terminal),
            }
        if lifecycle.status == "CANCELLED":
            lifecycle.cancellation.cancel()
        return lifecycle


__all__ = [
    "FINAL_STATUSES",
    "LangGraphCancellationScope",
    "LangGraphRunIdentity",
    "LangGraphRunLifecycle",
]
