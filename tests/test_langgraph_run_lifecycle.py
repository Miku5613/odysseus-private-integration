from __future__ import annotations

import asyncio

from src.agent_runtime.run_lifecycle import (
    LangGraphCancellationScope,
    LangGraphRunLifecycle,
)


def test_run_identity_and_event_cursor_are_stable():
    lifecycle = LangGraphRunLifecycle.create(
        session_id="session-1",
        run_id="run-1",
        thread_id="thread-1",
    )

    first = lifecycle.event({"type": "plan_update", "data": {}})
    second = lifecycle.event({"type": "worker_update", "data": {}})

    assert first["data"]["run_id"] == "run-1"
    assert first["data"]["session_id"] == "session-1"
    assert first["data"]["thread_id"] == "thread-1"
    assert first["data"]["event_cursor"] == 1
    assert second["data"]["event_cursor"] == 2


def test_terminal_transition_is_exactly_once():
    lifecycle = LangGraphRunLifecycle.create(
        session_id="session-1",
        run_id="run-1",
    )

    first = lifecycle.finish(
        "FAILED",
        stage="planner",
        reason="invalid plan",
    )
    duplicate = lifecycle.finish(
        "COMPLETED",
        stage="coordinator",
    )

    assert first is not None
    assert first["data"]["status"] == "FAILED"
    assert duplicate is None
    assert lifecycle.status == "FAILED"


def test_cancelling_one_run_does_not_cancel_another_run():
    async def scenario():
        release = asyncio.Event()

        async def child():
            await release.wait()

        scope_a = LangGraphCancellationScope("run-a")
        scope_b = LangGraphCancellationScope("run-b")
        task_a = scope_a.own_task(asyncio.create_task(child()))
        task_b = scope_b.own_task(asyncio.create_task(child()))

        assert scope_a.cancel() is True
        assert scope_a.cancel() is False
        await asyncio.sleep(0)

        assert task_a.cancelled()
        assert not task_b.cancelled()
        assert not task_b.done()

        release.set()
        await task_b

    asyncio.run(scenario())


def test_cancellation_calls_only_hooks_owned_by_that_run():
    calls = []
    scope_a = LangGraphCancellationScope("run-a")
    scope_b = LangGraphCancellationScope("run-b")
    scope_a.own_cancellation("research-a", lambda: calls.append("a"))
    scope_b.own_cancellation("research-b", lambda: calls.append("b"))

    scope_a.cancel()

    assert calls == ["a"]
    assert scope_b.cancelled is False


def test_async_cancellation_is_drained_and_failures_are_recorded():
    async def scenario():
        scope = LangGraphCancellationScope("run-drain")
        calls = []

        async def successful():
            await asyncio.sleep(0)
            calls.append("completed")

        async def failed():
            raise RuntimeError("remote cancellation failed")

        scope.own_cancellation("research:one", successful)
        scope.own_cancellation("mcp:two", failed)
        results = await scope.cancel_and_wait(timeout_s=1)

        assert calls == ["completed"]
        assert results["research:one"] == "cancelled"
        assert results["mcp:two"] == "failed:RuntimeError"

    asyncio.run(scenario())


def test_async_cancellation_timeout_is_bounded():
    async def scenario():
        scope = LangGraphCancellationScope("run-timeout")
        never = asyncio.Event()

        async def blocked():
            await never.wait()

        scope.own_cancellation("tool:blocked", blocked)
        results = await scope.cancel_and_wait(timeout_s=0)
        assert results["tool:blocked"] == "timeout"

    asyncio.run(scenario())
