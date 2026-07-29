import json

import pytest

from src.agent_runtime.live_supervisor import stream_general_supervisor_sse
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.worker_dispatcher import DispatchResult


def _payload(frame):
    assert frame.startswith("data: ")
    return json.loads(frame[len("data: "):].strip())


@pytest.mark.asyncio
async def test_live_supervisor_reasons_dispatches_and_finishes():
    calls = []

    async def fake_llm(endpoint_url, model, messages, **kwargs):
        calls.append(messages[0]["content"])
        if len(calls) == 1:
            return json.dumps({
                "selected_capabilities": ["knowledge_search"],
                "missing_information": [],
                "clarification_needed": False,
                "reasoning_summary": "Search indexed knowledge.",
            })
        return "I found the relevant indexed project notes."

    async def fake_dispatch(decision, objective, **kwargs):
        assert decision.selected_capabilities == ("knowledge_search",)
        assert objective == "Summarize my Odysseus notes."
        return DispatchResult(
            requested_capabilities=("knowledge_search",),
            completed_capabilities=("knowledge_search",),
            worker_results=(
                WorkerResult(
                    capability="knowledge_search",
                    status="completed",
                    summary="Retrieved indexed notes.",
                    evidence=({"path": "Projects/Odysseus.md"},),
                ),
            ),
        )

    frames = [
        frame
        async for frame in stream_general_supervisor_sse(
            "http://localhost",
            "test-model",
            [{"role": "user", "content": "Summarize my Odysseus notes."}],
            llm_caller=fake_llm,
            dispatcher=fake_dispatch,
        )
    ]

    assert _payload(frames[0])["tool"] == "knowledge_search"
    assert _payload(frames[0])["type"] == "tool_start"
    assert _payload(frames[1])["type"] == "tool_output"
    assert _payload(frames[2])["delta"] == "I found the relevant indexed project notes."
    assert _payload(frames[3])["data"]["runtime"] == "general-supervisor-v1"
    assert _payload(frames[3])["data"]["presenting_model"] == "test-model"
    assert _payload(frames[3])["data"]["response_origin"] == "selected_model"
    assert frames[-1] == "data: [DONE]\n\n"
    assert len(calls) == 2
    assert calls[1].startswith(
        "You are the final-response stage"
    )


@pytest.mark.asyncio
async def test_live_supervisor_clarifies_without_dispatching():
    async def fake_llm(endpoint_url, model, messages, **kwargs):
        return json.dumps({
            "selected_capabilities": [],
            "missing_information": ["which saved report you mean"],
            "clarification_needed": True,
            "reasoning_summary": "The target report is ambiguous.",
        })

    async def forbidden_dispatch(*args, **kwargs):
        raise AssertionError("dispatcher must not run")

    frames = [
        frame
        async for frame in stream_general_supervisor_sse(
            "http://localhost",
            "test-model",
            [{"role": "user", "content": "Read that report."}],
            llm_caller=fake_llm,
            dispatcher=forbidden_dispatch,
        )
    ]

    assert "which saved report" in _payload(frames[0])["delta"]
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_live_supervisor_raises_before_output_on_reasoner_failure():
    async def broken_llm(*args, **kwargs):
        raise RuntimeError("endpoint unavailable")

    generator = stream_general_supervisor_sse(
        "http://localhost",
        "test-model",
        [{"role": "user", "content": "Inspect the project."}],
        llm_caller=broken_llm,
    )

    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        await anext(generator)


@pytest.mark.asyncio
async def test_live_supervisor_retries_empty_synthesis_with_compact_evidence():
    calls = []

    async def fake_llm(endpoint_url, model, messages, **kwargs):
        calls.append((messages[0]["content"], kwargs))
        if len(calls) == 1:
            return json.dumps({
                "selected_capabilities": ["code_inspection"],
                "capability_actions": {"code_inspection": "inspect"},
                "missing_information": [],
                "clarification_needed": False,
                "reasoning_summary": "Inspect the named file.",
            })
        if len(calls) == 2:
            return ""
        return "Dependencies include web, data, security, and test tooling."

    async def fake_dispatch(*args, **kwargs):
        return DispatchResult(
            requested_capabilities=("code_inspection",),
            completed_capabilities=("code_inspection",),
            worker_results=(
                WorkerResult(
                    capability="code_inspection",
                    status="completed",
                    summary="requirements.txt contains fastapi and pytest.",
                ),
            ),
        )

    frames = [
        frame
        async for frame in stream_general_supervisor_sse(
            "http://localhost",
            "test-model",
            [{"role": "user", "content": "Read requirements.txt."}],
            llm_caller=fake_llm,
            dispatcher=fake_dispatch,
        )
    ]

    assert len(calls) == 3
    assert "WORKER SUMMARIES:" in calls[2][0]
    assert calls[2][1]["temperature"] == 0.3
    assert _payload(frames[2])["delta"].startswith("Dependencies include")
    assert _payload(frames[3])["data"]["presenting_model"] == "test-model"


def test_chat_route_uses_live_general_supervisor():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = (root / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "stream_general_supervisor_sse" in chat
    assert "stream_langgraph_agent_legacy_sse" not in chat
    assert "_stream_langgraph_first_with_legacy_fallback" in chat
