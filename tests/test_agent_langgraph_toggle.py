from __future__ import annotations

import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_agent_langgraph_toggle_is_exposed_in_agent_tools_ui() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert "set-agentLangGraphEnabled" in html
    assert "Use LangGraph for Agent mode" in html
    assert "legacy runtime" in html


def test_agent_langgraph_toggle_is_saved_by_settings_js() -> None:
    js = (ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")

    assert "set-agentLangGraphEnabled" in js
    assert "agent_langgraph_enabled" in js
    assert "LangGraph on" in js
    assert "LangGraph off" in js


def test_chat_route_skips_runtime_router_when_agent_langgraph_disabled() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert 'get_setting("agent_langgraph_enabled", True)' in chat
    assert "_runtime_decision = _route_agent_runtime_for_request(" in chat
    assert "_agent_langgraph_architecture_enabled" in chat
    assert "_runtime_decision is not None" in chat
    assert "_agent_langgraph_enabled_for_knowledge" not in chat


def test_agent_langgraph_toggle_is_request_level_bypass() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "def _agent_langgraph_architecture_enabled_for_request" in chat
    assert "AGENT_LANGGRAPH_REQUEST_BYPASS_V1_START" in chat
    assert "_agent_langgraph_architecture_enabled = _agent_langgraph_architecture_enabled_for_request(chat_mode)" in chat
    assert "if user_requested_agent:" in chat
    assert "assess_explicit_agent_admission(" in chat
    assert "_agent_admission.use_normal_chat" in chat
    assert "_clear_tool_or_runtime_intent" not in chat
    assert "_runtime_decision = _route_agent_runtime_for_request(" in chat
    assert "_agent_langgraph_enabled_for_knowledge" not in chat


def test_agent_langgraph_toggle_gates_direct_knowledge_runtime_branch() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "_knowledge_intercept = (" in chat
    assert "should_intercept_workspace_knowledge(message)" in chat
    assert "and _knowledge_runtime_enabled" in chat
    assert "and _knowledge_intercept" in chat
    assert "_explicit_agent_single_graph" in chat
    assert "_explicit_agent_multi_graph" in chat
    assert "and not (" in chat
    assert 'if chat_mode == "agent" and _knowledge_runtime_enabled and _knowledge_intercept:' not in chat


def test_explicit_single_graph_admission_owns_native_supervisor_boundary() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "P3_C5_EXPLICIT_GRAPH_ROUTE_START" in chat
    assert "_explicit_agent_single_graph = bool(" in chat
    assert "_agent_admission.use_single_graph" in chat
    assert "_explicit_agent_single_graph\n                            or (" in chat
    assert "and not (\n                                _explicit_agent_single_graph" in chat
    assert "or _explicit_agent_multi_graph" in chat
    assert "_stream_langgraph_first_with_legacy_fallback" in chat


def test_agent_langgraph_toggle_has_default_setting() -> None:
    settings_py = (ROOT / "src" / "settings.py").read_text(encoding="utf-8")

    assert '"agent_langgraph_enabled": True' in settings_py


def test_agent_langgraph_toggle_is_accepted_by_settings_endpoint() -> None:
    auth_routes = (ROOT / "routes" / "auth_routes.py").read_text(encoding="utf-8")

    assert '"agent_langgraph_enabled"' in auth_routes
    assert "_coerce_bool_setting" in auth_routes
    assert 'current["agent_langgraph_enabled"] = _coerce_bool_setting(body.get("agent_langgraph_enabled"))' in auth_routes


def test_model_backed_explicit_agent_admission_preserves_upstream_auto_escalation() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "_clear_tool_or_runtime_intent" not in chat
    assert "assess_explicit_agent_admission(" in chat
    assert "if user_requested_agent:" in chat

    assert "# Intent auto-escalation:" in chat
    assert 'if _agent_langgraph_architecture_enabled and chat_mode == "chat" and _tool_intent and _tool_intent.needs_tools:' in chat
    assert 'chat_mode = "agent"' in chat
    assert "auto_escalated = True" in chat
    assert "chat→agent auto-escalation:" in chat
    assert "if auto_escalated:" in chat
    assert '"bash", "python", "read_file", "write_file", "builtin_browser"' in chat


def test_langgraph_toggle_off_allows_none_runtime_decision_without_router_crash() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "if _runtime_decision is not None:" in chat
    assert "[odysseus-router] skipped because Agent LangGraph architecture is disabled" in chat
    assert "and _runtime_decision is not None\n                        and _runtime_decision.target_runtime == \"contract\"" in chat
    assert "and _runtime_decision is not None\n                        and _runtime_decision.target_runtime == \"general\"" in chat


def test_langgraph_request_bypass_never_calls_runtime_router(monkeypatch) -> None:
    from routes import chat_routes

    calls = []

    def forbidden_router(message):
        calls.append(message)
        raise AssertionError("LangGraph router leaked into Legacy request")

    monkeypatch.setattr(chat_routes, "route_agent_runtime", forbidden_router)

    assert chat_routes._route_agent_runtime_for_request("Find doc_stream", False) is None
    assert calls == []


def test_explicit_multi_graph_admission_owns_coordinator_boundary() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "stream_multi_graph_coordinator_sse" in chat
    assert "def _stream_multi_graph_fail_closed" in chat
    assert "_explicit_agent_multi_graph = bool(" in chat
    assert "_agent_admission.use_multi_graph" in chat
    assert "admission_assessment=(" in chat
    assert "_agent_admission.assessment" in chat
    assert "not _explicit_agent_multi_graph\n                        and contract_runtime_enabled()" in chat
    assert "not _explicit_agent_multi_graph\n                        and general_runtime_enabled()" in chat
    assert "not _explicit_agent_multi_graph\n                        and _router_agent_runtime_requested" in chat
    assert "_stream_multi_graph_fail_closed" in chat
    assert "whole-query Legacy fallback" in chat
    assert "legacy_kwargs" not in chat
    assert '"runtime_run"' in chat
    assert '"runtime_terminal"' in chat


def test_multi_graph_startup_failure_never_invokes_whole_query_legacy(
    monkeypatch,
) -> None:
    from routes import chat_routes

    async def failing_coordinator():
        if False:
            yield ""
        raise RuntimeError("coordinator unavailable")

    legacy_calls = []

    async def forbidden_legacy(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        yield "data: forbidden\n\n"

    monkeypatch.setattr(
        chat_routes,
        "stream_multi_graph_coordinator_sse",
        failing_coordinator,
    )
    monkeypatch.setattr(
        chat_routes,
        "stream_agent_loop",
        forbidden_legacy,
    )

    async def collect():
        return [
            frame
            async for frame in chat_routes._stream_multi_graph_fail_closed()
        ]

    frames = asyncio.run(collect())
    payloads = [
        json.loads(frame[6:])
        for frame in frames
        if frame.startswith("data: ")
        and frame != "data: [DONE]\n\n"
    ]
    terminals = [
        item
        for item in payloads
        if item.get("type") == "runtime_terminal"
    ]

    assert legacy_calls == []
    assert frames.count("data: [DONE]\n\n") == 1
    assert len(terminals) == 1
    assert terminals[0]["data"]["status"] == "FAILED"
    assert terminals[0]["data"]["whole_query_legacy_fallback"] is False


def test_durable_checkpoint_initialization_failure_is_controlled(
    monkeypatch,
) -> None:
    from routes import chat_routes
    from src.agent_runtime.durable_checkpoint import (
        CheckpointUnavailableError,
    )

    def unavailable():
        raise CheckpointUnavailableError("database is read-only")

    coordinator_calls = []

    async def forbidden_coordinator(*args, **kwargs):
        coordinator_calls.append((args, kwargs))
        yield "data: forbidden\n\n"

    monkeypatch.setattr(
        chat_routes,
        "get_default_checkpoint_store",
        unavailable,
    )
    monkeypatch.setattr(
        chat_routes,
        "stream_multi_graph_coordinator_sse",
        forbidden_coordinator,
    )

    async def collect():
        return [
            frame
            async for frame in chat_routes._stream_multi_graph_fail_closed()
        ]

    frames = asyncio.run(collect())
    payloads = [
        json.loads(frame[6:])
        for frame in frames
        if frame.startswith("data: ")
        and frame != "data: [DONE]\n\n"
    ]
    terminal = next(
        item
        for item in payloads
        if item.get("type") == "runtime_terminal"
    )

    assert coordinator_calls == []
    assert terminal["data"]["status"] == "FAILED"
    assert "CheckpointUnavailableError" in terminal["data"]["reason"]
    assert frames.count("data: [DONE]\n\n") == 1
