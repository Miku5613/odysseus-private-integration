from __future__ import annotations

import asyncio
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
    runtime_mode: str = "multi_graph"

    @classmethod
    def create(
        cls,
        *,
        session_id: str | None,
        run_id: str | None = None,
        thread_id: str | None = None,
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
        return cls(
            run_id=stable_run_id,
            session_id=stable_session_id,
            thread_id=stable_thread_id,
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

    @classmethod
    def create(
        cls,
        *,
        session_id: str | None,
        run_id: str | None = None,
        thread_id: str | None = None,
        runtime_mode: str = "multi_graph",
        cancellation: LangGraphCancellationScope | None = None,
    ) -> "LangGraphRunLifecycle":
        identity = LangGraphRunIdentity.create(
            session_id=session_id,
            run_id=run_id,
            thread_id=thread_id,
            runtime_mode=runtime_mode,
        )
        scope = cancellation or LangGraphCancellationScope(identity.run_id)
        if scope.run_id != identity.run_id:
            raise ValueError(
                "cancellation scope run_id must match lifecycle run_id"
            )
        return cls(identity=identity, cancellation=scope)

    def event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.event_cursor += 1
        framed = dict(payload)
        data = dict(framed.get("data") or {})
        data.update({
            "run_id": self.identity.run_id,
            "session_id": self.identity.session_id,
            "thread_id": self.identity.thread_id,
            "runtime_mode": self.identity.runtime_mode,
            "event_cursor": self.event_cursor,
        })
        framed["data"] = data
        return framed

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
        self.terminal = self.event({
            "type": "runtime_terminal",
            "data": {
                "status": normalized,
                "stage": stage,
                "reason": reason,
                **details,
            },
        })
        return self.terminal


__all__ = [
    "FINAL_STATUSES",
    "LangGraphCancellationScope",
    "LangGraphRunIdentity",
    "LangGraphRunLifecycle",
]
