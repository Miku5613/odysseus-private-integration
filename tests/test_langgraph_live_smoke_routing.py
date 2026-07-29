from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.agent_runtime import agent_admission
from src.agent_runtime.general_graph import plan_task
from src.agent_runtime.knowledge_semantic_chat_bridge import (
    should_intercept_workspace_knowledge,
)
from src.agent_runtime.live_supervisor import stream_general_supervisor_sse
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.tool_policy import infer_tool_scope
from src.agent_runtime.worker_dispatcher import DispatchResult


SKY = "Explain in two paragraphs why the sky appears blue."
README = (
    "Inspect the Odysseus project README and summarize its main purpose.\n"
    "Do not modify any files."
)
ARCHITECTURE = (
    "Inspect the current Odysseus LangGraph runtime documentation and relevant "
    "source files, compare the intended architecture with the implementation, "
    "and summarize any differences. Do not edit anything."
)
WRITE = (
    "Create a new text file named langgraph-live-smoke.txt in the current "
    'workspace containing the words "live smoke test". Ask for approval before '
    "writing."
)


def _payload(frame: str):
    assert frame.startswith("data: ")
    return json.loads(frame[6:].strip())


@pytest.mark.asyncio
async def test_exact_sky_prompt_uses_agent_zero_graph_direct_llm(monkeypatch):
    calls = []

    async def conversation(*args, **kwargs):
        calls.append(args[0])
        return type("Assessment", (), {"execution_class": "conversation"})()

    monkeypatch.setattr(agent_admission, "assess_intent_complexity", conversation)
    decision = await agent_admission.assess_explicit_agent_admission(
        SKY,
        langgraph_enabled=True,
        llm_candidates=[("endpoint", "qwen3:8b", {})],
    )

    assert decision.use_normal_chat
    assert calls == [SKY]
    assert should_intercept_workspace_knowledge(SKY) is False


@pytest.mark.parametrize("prompt", [README, ARCHITECTURE])
def test_exact_readonly_prompts_do_not_acquire_mutation_scope(prompt):
    assert infer_tool_scope(prompt) == "file_inspection"
    assert plan_task(prompt).is_readonly is True
    assert plan_task(prompt).reason != (
        "Request appears to involve write/edit/execute behavior."
    )


@pytest.mark.asyncio
async def test_readme_worker_reads_evidence_and_selected_model_synthesizes():
    calls = []
    project_readme = Path(__file__).resolve().parents[1] / "README.md"

    async def fake_llm(endpoint_url, model, messages, **kwargs):
        calls.append(messages[0]["content"])
        if len(calls) == 1:
            return json.dumps({
                "selected_capabilities": ["code_inspection"],
                "missing_information": [],
                "clarification_needed": False,
                "reasoning_summary": "Read the requested project file.",
            })
        return "The selected model synthesized the README evidence."

    async def read_dispatch(decision, objective, **kwargs):
        text = project_readme.read_text(encoding="utf-8")
        return DispatchResult(
            requested_capabilities=("code_inspection",),
            completed_capabilities=("code_inspection",),
            worker_results=(
                WorkerResult(
                    capability="code_inspection",
                    status="completed",
                    summary=text[:500],
                    evidence=({"path": str(project_readme)},),
                ),
            ),
        )

    frames = [
        frame
        async for frame in stream_general_supervisor_sse(
            "endpoint",
            "qwen3:8b",
            [{"role": "user", "content": README}],
            llm_caller=fake_llm,
            dispatcher=read_dispatch,
        )
    ]
    metrics = _payload(frames[-2])["data"]
    assert len(calls) == 2
    assert "WORKER RESULTS:" in calls[1]
    assert "README.md" in calls[1]
    assert _payload(frames[-3])["delta"].startswith("The selected model")
    assert metrics["presenting_model"] == "qwen3:8b"
    assert metrics["response_origin"] == "selected_model"
    assert metrics["fallback_used"] is False


@pytest.mark.asyncio
async def test_exact_write_prompt_projects_durable_confirmation_without_write():
    calls = []

    async def fake_llm(endpoint_url, model, messages, **kwargs):
        calls.append(messages[0]["content"])
        if len(calls) == 1:
            return json.dumps({
                "selected_capabilities": ["code_inspection"],
                "missing_information": [],
                "clarification_needed": False,
                "reasoning_summary": "Prepare the exact workspace write.",
                "capability_actions": {
                    "code_inspection": "prepare_code_output",
                },
            })
        return "I prepared the exact write and am waiting for your approval."

    confirmation = {
        "type": "tool_confirmation_required",
        "name": "write_file",
        "reason": "Workspace mutation requires confirmation.",
        "call_id": "write-1",
        "session_id": "session-1",
        "fingerprint": "exact-fingerprint",
        "confirmation_available": True,
    }

    async def approval_dispatch(decision, objective, **kwargs):
        return DispatchResult(
            requested_capabilities=("code_inspection",),
            completed_capabilities=(),
            worker_results=(
                WorkerResult(
                    capability="code_inspection",
                    status="partial",
                    summary="No file was written before approval.",
                    data={"runtime_events": [confirmation]},
                ),
            ),
        )

    frames = [
        frame
        async for frame in stream_general_supervisor_sse(
            "endpoint",
            "qwen3:8b",
            [{"role": "user", "content": WRITE}],
            session_id="session-1",
            owner="alice",
            workspace=str(Path(__file__).resolve().parents[1]),
            llm_caller=fake_llm,
            dispatcher=approval_dispatch,
        )
    ]
    payloads = [_payload(frame) for frame in frames[:-1]]
    approval = next(item for item in payloads if item.get("type") == "ask_user")
    assert infer_tool_scope(WRITE) == "workspace_mutation"
    assert approval["data"]["interaction_mode"] == "tool_confirmation"
    assert approval["data"]["fingerprint"] == "exact-fingerprint"
    assert len(calls) == 2


def test_route_has_no_semantically_admitted_deterministic_or_legacy_widening():
    source = (
        Path(__file__).resolve().parents[1] / "routes" / "chat_routes.py"
    ).read_text(encoding="utf-8")

    assert "falling back to Legacy agent loop" not in source
    assert "failing closed without whole-query Legacy fallback" in source
    assert (
        '_runtime_decision.intent == "write_mutation"\n'
        "                        and not (\n"
        "                            _explicit_agent_single_graph"
    ) in source
