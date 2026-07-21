import json

import pytest

from src.agent_runtime.research_worker import run_research_worker
from src.agent_runtime.supervisor_contracts import WorkerHandoff


@pytest.mark.asyncio
async def test_research_worker_starts_existing_research_runtime():
    captured = {}

    async def fake_trigger(content, owner=None):
        captured["content"] = json.loads(content)
        captured["owner"] = owner
        return {
            "output": "Research started.",
            "task_id": "research-1",
            "exit_code": 0,
        }

    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Investigate LangGraph supervisor patterns.",
        context={"owner": "user-1"},
    )

    result = await run_research_worker(
        handoff,
        trigger_backend=fake_trigger,
    )

    assert captured["content"]["topic"] == handoff.objective
    assert captured["owner"] == "user-1"
    assert result.status == "completed"
    assert result.data["payload"]["task_id"] == "research-1"


@pytest.mark.asyncio
async def test_research_worker_reads_existing_report():
    captured = {}

    async def fake_manage(content, owner=None):
        captured.update(json.loads(content))
        return {"output": "Saved report text.", "exit_code": 0}

    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Read the saved report.",
        context={
            "action": "read",
            "research_id": "rp-123",
        },
    )

    result = await run_research_worker(
        handoff,
        manage_backend=fake_manage,
    )

    assert captured == {"action": "read", "id": "rp-123"}
    assert result.status == "completed"
    assert result.summary == "Saved report text."


@pytest.mark.asyncio
async def test_research_worker_requires_id_for_read():
    handoff = WorkerHandoff(
        capability="deep_research",
        objective="Read that report.",
        context={"action": "read"},
    )

    result = await run_research_worker(handoff)

    assert result.status == "needs_clarification"
    assert result.missing_information == ("research_id",)


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
