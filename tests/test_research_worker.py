import asyncio
import json

import pytest

from src.agent_runtime.research_worker import (
    cancel_owned_research_jobs,
    run_research_worker,
)
from src.agent_runtime.run_lifecycle import LangGraphCancellationScope
from src.agent_runtime.supervisor_contracts import WorkerHandoff


@pytest.mark.asyncio
async def test_research_trigger_is_pending_until_final_report_is_observed():
    captured = {}
    statuses = iter(["queued", "running", "done"])

    async def fake_trigger(content, owner=None):
        captured["content"] = json.loads(content)
        captured["owner"] = owner
        return {
            "output": "Research started.",
            "task_id": "research-1",
            "exit_code": 0,
        }

    async def fake_status(job_id, owner=None):
        assert job_id == "research-1"
        return {"status": next(statuses), "exit_code": 0}

    report = (
        "LangGraph supervisor patterns use explicit coordinator ownership, "
        "durable state, bounded worker delegation, and isolated cancellation. "
        "This detailed report explains how those patterns support reliable "
        "multi-step execution and recovery."
    )

    async def fake_result(job_id, owner=None):
        return {
            "result": report,
            "sources": [{"url": "https://example.test/langgraph"}],
            "raw_findings": [{"text": "Coordinator evidence"}],
            "exit_code": 0,
        }

    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Investigate LangGraph supervisor patterns.",
        context={
            "owner": "user-1",
            "research_poll_interval_s": 0,
        },
    )

    result = await run_research_worker(
        handoff,
        trigger_backend=fake_trigger,
        status_backend=fake_status,
        result_backend=fake_result,
    )

    assert captured["content"]["topic"] == handoff.objective
    assert captured["owner"] == "user-1"
    assert result.status == "completed"
    assert result.data["research_job_id"] == "research-1"
    assert result.data["research_status"] == "completed"
    assert result.data["observed_states"] == [
        "queued",
        "running",
        "completed",
    ]
    assert result.data["semantic_complete"] is True
    assert result.data["dependency_state"]["report"] == report


@pytest.mark.asyncio
async def test_research_worker_reads_existing_report():
    captured = {}
    cancellation_calls = []
    scope = LangGraphCancellationScope("run-read")

    async def fake_manage(content, owner=None):
        captured.update(json.loads(content))
        return {"output": "Saved report text.", "exit_code": 0}

    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Read the saved report.",
        context={
            "action": "read",
            "research_id": "rp-123",
            "cancellation_scope": scope,
        },
    )

    async def forbidden_cancel(job_id, owner=None):
        cancellation_calls.append(job_id)
        return {"cancelled": True, "exit_code": 0}

    result = await run_research_worker(
        handoff,
        manage_backend=fake_manage,
        cancel_backend=forbidden_cancel,
        semantic_validator=lambda objective, report: (True, "accepted"),
    )
    scope.cancel()
    await asyncio.sleep(0)

    assert captured == {"action": "read", "id": "rp-123"}
    assert result.status == "completed"
    assert result.summary == "Saved report text."
    assert result.data["semantic_complete"] is True
    assert cancellation_calls == []


@pytest.mark.asyncio
async def test_research_worker_resolves_latest_report_when_read_has_no_id():
    requests = []

    async def fake_manage(content, owner=None):
        request = json.loads(content)
        requests.append(request)

        if request == {"action": "list"}:
            return {
                "output": (
                    "[Latest report](#research-rp-latest)"
                ),
                "exit_code": 0,
            }

        if request == {
            "action": "read",
            "id": "rp-latest",
        }:
            return {
                "output": "Latest saved report text.",
                "exit_code": 0,
            }

        raise AssertionError(
            f"Unexpected research request: {request}"
        )

    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Read the latest saved report.",
        context={"action": "read"},
    )

    result = await run_research_worker(
        handoff,
        manage_backend=fake_manage,
        semantic_validator=lambda objective, report: (True, "accepted"),
    )

    assert requests == [
        {"action": "list"},
        {
            "action": "read",
            "id": "rp-latest",
        },
    ]
    assert result.status == "completed"
    assert result.summary == "Latest saved report text."
    assert result.data["semantic_complete"] is True


@pytest.mark.asyncio
async def test_research_trigger_without_stable_job_id_fails_closed():
    async def fake_trigger(content, owner=None):
        return {
            "output": "Research accepted.",
            "exit_code": 0,
        }

    result = await run_research_worker(
        WorkerHandoff(
            capability="deep_research",
            objective="Investigate runtime durability.",
        ),
        trigger_backend=fake_trigger,
    )

    assert result.status == "failed"
    assert result.data["semantic_complete"] is False
    assert "stable job ID" in (result.error or "")


@pytest.mark.asyncio
async def test_research_worker_clarifies_when_no_completed_report_can_be_resolved():
    async def fake_manage(content, owner=None):
        assert json.loads(content) == {
            "action": "list",
        }
        return {
            "output": "No completed reports.",
            "exit_code": 0,
        }

    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Read the latest saved report.",
        context={"action": "read"},
    )

    result = await run_research_worker(
        handoff,
        manage_backend=fake_manage,
    )

    assert result.status == "needs_clarification"
    assert result.missing_information == (
        "completed research report",
    )


@pytest.mark.asyncio
async def test_research_worker_normalizes_backend_failure():
    async def broken_trigger(content, owner=None):
        return {"error": "Research unavailable", "exit_code": 1}

    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Research this topic.",
    )

    result = await run_research_worker(
        handoff,
        trigger_backend=broken_trigger,
    )

    assert result.status == "failed"
    assert result.error == "Research unavailable"


@pytest.mark.asyncio
async def test_owned_job_resume_prevents_duplicate_trigger():
    trigger_calls = []

    async def forbidden_trigger(content, owner=None):
        trigger_calls.append(content)
        raise AssertionError("owned job was triggered twice")

    async def completed_status(job_id, owner=None):
        return {"status": "completed", "exit_code": 0}

    report = (
        "Durable checkpoint recovery keeps each LangGraph run isolated by "
        "thread and run identity. The completed analysis includes sufficient "
        "checkpoint evidence and explains recovery after process restart."
    )

    async def final_result(job_id, owner=None):
        return {"result": report, "sources": [], "exit_code": 0}

    result = await run_research_worker(
        WorkerHandoff(
            capability="deep_research",
            objective="Analyze durable checkpoint recovery.",
            context={
                "action": "start",
                "research_job_id": "owned-1",
                "research_job_owned": True,
            },
        ),
        trigger_backend=forbidden_trigger,
        status_backend=completed_status,
        result_backend=final_result,
    )

    assert trigger_calls == []
    assert result.status == "completed"
    assert result.data["research_job_id"] == "owned-1"


@pytest.mark.asyncio
async def test_running_research_remains_nonterminal_with_job_id():
    async def trigger(content, owner=None):
        return {"research_session_id": "running-1", "exit_code": 0}

    async def running_status(job_id, owner=None):
        return {"status": "running", "exit_code": 0}

    result = await run_research_worker(
        WorkerHandoff(
            capability="deep_research",
            objective="Research durable worker state.",
            context={"research_timeout_s": 0},
        ),
        trigger_backend=trigger,
        status_backend=running_status,
    )

    assert result.status == "partial"
    assert result.data["research_job_id"] == "running-1"
    assert result.data["research_status"] == "running"
    assert result.data["semantic_complete"] is False


@pytest.mark.asyncio
async def test_research_failure_is_typed_and_terminal():
    async def trigger(content, owner=None):
        return {"research_session_id": "failed-1", "exit_code": 0}

    async def failed_status(job_id, owner=None):
        return {
            "status": "error",
            "error": "provider exhausted",
            "exit_code": 0,
        }

    result = await run_research_worker(
        WorkerHandoff(
            capability="deep_research",
            objective="Research provider behavior.",
        ),
        trigger_backend=trigger,
        status_backend=failed_status,
    )

    assert result.status == "failed"
    assert result.data["research_status"] == "failed"
    assert result.error == "provider exhausted"


@pytest.mark.asyncio
async def test_insufficient_final_report_fails_controlled():
    async def trigger(content, owner=None):
        return {"research_session_id": "short-1", "exit_code": 0}

    async def completed_status(job_id, owner=None):
        return {"status": "done", "exit_code": 0}

    async def short_result(job_id, owner=None):
        return {"result": "Too short.", "sources": [], "exit_code": 0}

    result = await run_research_worker(
        WorkerHandoff(
            capability="deep_research",
            objective="Research LangGraph cancellation isolation.",
        ),
        trigger_backend=trigger,
        status_backend=completed_status,
        result_backend=short_result,
    )

    assert result.status == "failed"
    assert result.data["semantic_complete"] is False
    assert "too short" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_cancellation_stops_only_owned_research_job():
    status_started = asyncio.Event()
    never_complete = asyncio.Event()
    cancelled = []
    scope = LangGraphCancellationScope("run-owned")

    async def trigger(content, owner=None):
        return {"research_session_id": "owned-job", "exit_code": 0}

    async def running_status(job_id, owner=None):
        status_started.set()
        await never_complete.wait()
        return {"status": "running", "exit_code": 0}

    async def cancel(job_id, owner=None):
        cancelled.append(job_id)
        return {"cancelled": True, "exit_code": 0}

    task = asyncio.create_task(
        run_research_worker(
            WorkerHandoff(
                capability="deep_research",
                objective="Research owned cancellation behavior.",
                context={"cancellation_scope": scope},
            ),
            trigger_backend=trigger,
            status_backend=running_status,
            cancel_backend=cancel,
        )
    )
    await status_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert cancelled == ["owned-job"]


@pytest.mark.asyncio
async def test_recovered_cancellation_ignores_unowned_and_completed_jobs():
    calls = []

    async def cancel(job_id, owner=None):
        calls.append((job_id, owner))
        return {"cancelled": True, "exit_code": 0}

    cancelled = await cancel_owned_research_jobs(
        {
            "owned_research_jobs": {
                "running-owned": {
                    "owned": True,
                    "status": "running",
                },
                "completed-owned": {
                    "owned": True,
                    "status": "completed",
                },
                "existing-report": {
                    "owned": False,
                    "status": "completed",
                },
            }
        },
        owner="owner-1",
        cancel_backend=cancel,
    )

    assert cancelled == ("running-owned",)
    assert calls == [("running-owned", "owner-1")]
