from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.agent_runtime import agent_admission
from src.agent_runtime.agent_admission import AgentAdmissionDecision
from src.agent_runtime.general_graph import plan_task
from src.agent_runtime.knowledge_semantic_chat_bridge import (
    should_intercept_workspace_knowledge,
)
from src.agent_runtime.live_supervisor import stream_general_supervisor_sse
from src.agent_runtime.supervisor_contracts import WorkerResult
from src.agent_runtime.tool_policy import infer_tool_scope
from src.agent_runtime.worker_dispatcher import DispatchResult
import routes.chat_routes as chat_routes


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


class _RouteSession:
    def __init__(self):
        self.id = "live-smoke-session"
        self.name = "Live smoke"
        self.model = "qwen3:8b"
        self.endpoint_url = "http://model.test"
        self.headers = {}
        self.history = []

    def add_message(self, message):
        self.history.append(message)


class _RouteSessions:
    def __init__(self, session):
        self.session = session

    def get_session(self, session_id):
        assert session_id == self.session.id
        return self.session

    def save_sessions(self):
        return None


def _live_route_client(monkeypatch, tmp_path, *, admission_path, stream):
    session = _RouteSession()
    manager = _RouteSessions(session)
    saved = []

    async def admission(*args, **kwargs):
        return AgentAdmissionDecision(
            path=admission_path,
            assessment=SimpleNamespace(execution_class=admission_path),
        )

    async def context(*args, **kwargs):
        message = kwargs["message"]
        return SimpleNamespace(
            messages=[{"role": "user", "content": message}],
            preface=[],
            preset=SimpleNamespace(
                temperature=0.0,
                max_tokens=512,
                character_name=None,
            ),
            context_length=4096,
            context_trimmed=False,
            was_compacted=False,
            context_messages_before_trim=1,
            context_messages_after_trim=1,
            context_tokens_before_trim=20,
            context_tokens_after_trim=20,
            auto_opened_docs=[],
            uploaded_files=[],
            web_sources=[],
            rag_sources=[],
            used_memories=[],
            uprefs=None,
            user="alice",
            preprocessed=SimpleNamespace(
                attachment_meta=[],
                youtube_transcripts=[],
            ),
            get=lambda *_args, **_kwargs: None,
        )

    monkeypatch.setattr(chat_routes, "assess_explicit_agent_admission", admission)
    monkeypatch.setattr(chat_routes, "build_chat_context", context)
    monkeypatch.setattr(chat_routes, "_verify_session_owner", lambda *_: None)
    monkeypatch.setattr(chat_routes, "_enforce_chat_privileges", lambda *_: None)
    monkeypatch.setattr(chat_routes, "_clear_orphaned_session_endpoint", lambda *_, **__: False)
    monkeypatch.setattr(chat_routes, "_recover_empty_session_model", lambda *_, **__: None)
    monkeypatch.setattr(chat_routes, "resolve_session_auth", lambda *_, **__: None)
    monkeypatch.setattr(chat_routes, "effective_user", lambda *_: "alice")
    monkeypatch.setattr(chat_routes, "get_current_user", lambda *_: "alice")
    monkeypatch.setattr(chat_routes, "get_session_mode", lambda *_: "agent")
    monkeypatch.setattr(chat_routes, "set_session_mode", lambda *_: None)
    monkeypatch.setattr(chat_routes, "_agent_langgraph_architecture_enabled_for_request", lambda *_: True)
    monkeypatch.setattr(chat_routes, "_resolve_request_workspace", lambda request, raw: (str(tmp_path), False))
    monkeypatch.setattr(chat_routes, "_resolve_workspace_from_message_path", lambda *_: ("", False))
    monkeypatch.setattr(chat_routes, "_is_image_generation_session", lambda *_, **__: False)
    monkeypatch.setattr(chat_routes, "run_post_response_tasks", lambda *_, **__: None)

    def save(*args, **kwargs):
        saved.append((args, kwargs))
        return "saved-live-smoke"

    monkeypatch.setattr(chat_routes, "save_assistant_response", save)
    if admission_path == "normal_chat":
        monkeypatch.setattr(chat_routes, "stream_llm_with_fallback", stream)
    else:
        monkeypatch.setattr(chat_routes, "stream_general_supervisor_sse", stream)

    app = FastAPI()
    app.include_router(
        chat_routes.setup_chat_routes(
            manager,
            object(),
            object(),
            object(),
            object(),
            object(),
        )
    )
    return TestClient(app), session, saved


def test_real_chat_stream_route_uses_zero_graph_selected_llm(
    monkeypatch,
    tmp_path,
):
    calls = []

    async def direct(candidates, messages, **kwargs):
        calls.append(messages)
        yield 'data: {"delta":"Paragraph one.\\n\\nParagraph two."}\n\n'
        yield (
            'data: {"type":"metrics","data":{"model":"qwen3:8b",'
            '"selected_model":"qwen3:8b","presenting_model":"qwen3:8b",'
            '"response_origin":"selected_model"}}\n\n'
        )
        yield "data: [DONE]\n\n"

    client, _, saved = _live_route_client(
        monkeypatch,
        tmp_path,
        admission_path="normal_chat",
        stream=direct,
    )
    response = client.post(
        "/api/chat_stream",
        data={
            "message": SKY,
            "session": "live-smoke-session",
            "mode": "agent",
            "workspace": str(tmp_path),
        },
    )

    assert response.status_code == 200
    assert "Paragraph one" in response.text
    assert "RUNTIME_DECISION" not in response.text
    assert len(calls) == 1
    assert len(saved) == 1


def test_real_chat_stream_route_bypasses_deterministic_readonly_bridge(
    monkeypatch,
    tmp_path,
):
    calls = []

    async def one_graph(*args, **kwargs):
        calls.append((args, kwargs))
        yield 'data: {"type":"tool_start","tool":"code_inspection"}\n\n'
        yield 'data: {"type":"tool_output","tool":"code_inspection","output":"README evidence"}\n\n'
        yield 'data: {"delta":"Selected-model README synthesis."}\n\n'
        yield (
            'data: {"type":"metrics","data":{"runtime":"general-supervisor-v1",'
            '"model":"qwen3:8b","selected_model":"qwen3:8b",'
            '"presenting_model":"qwen3:8b","response_origin":"selected_model",'
            '"langgraph_runtime":true}}\n\n'
        )
        yield "data: [DONE]\n\n"

    client, _, saved = _live_route_client(
        monkeypatch,
        tmp_path,
        admission_path="single_graph",
        stream=one_graph,
    )
    response = client.post(
        "/api/chat_stream",
        data={
            "message": README,
            "session": "live-smoke-session",
            "mode": "agent",
            "workspace": str(tmp_path),
        },
    )

    assert response.status_code == 200
    assert "Selected-model README synthesis." in response.text
    assert "deterministic-general" not in response.text
    assert len(calls) == 1
    assert len(saved) == 1
