import pytest

from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.supervisor_decision import SupervisorDecision
from src.agent_runtime.worker_dispatcher import dispatch_supervisor_decision_async


@pytest.mark.asyncio
async def test_async_dispatcher_runs_async_research_worker():
    async def fake_research_worker(handoff, session_id=None):
        return WorkerResult(
            capability=handoff.capability,
            status="completed",
            summary="Research started.",
        )

    decision = SupervisorDecision(
        selected_capabilities=("deep_research",),
    )

    result = await dispatch_supervisor_decision_async(
        decision,
        "Research LangGraph.",
        worker_runners={
            "deep_research": fake_research_worker,
        },
    )

    assert result.completed_capabilities == ("deep_research",)
    assert result.worker_results[0].summary == "Research started."


@pytest.mark.asyncio
async def test_async_dispatcher_also_runs_sync_worker():
    def fake_sync_worker(handoff, session_id=None):
        return WorkerResult(
            capability=handoff.capability,
            status="completed",
            summary="Sync worker complete.",
        )

    decision = SupervisorDecision(
        selected_capabilities=("code_inspection",),
    )

    result = await dispatch_supervisor_decision_async(
        decision,
        "Inspect code.",
        worker_runners={
            "code_inspection": fake_sync_worker,
        },
    )

    assert result.completed_capabilities == ("code_inspection",)
