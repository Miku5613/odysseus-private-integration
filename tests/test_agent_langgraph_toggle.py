from __future__ import annotations

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
    assert "_knowledge_intent_requested = (" in chat
    assert "_agent_langgraph_architecture_enabled\n                and isinstance(message, str)" in chat
    assert "_f3_runtime_probe = _route_agent_runtime_for_request(" in chat
    assert "_runtime_decision = _route_agent_runtime_for_request(" in chat
    assert "_agent_langgraph_enabled_for_knowledge" not in chat


def test_agent_langgraph_toggle_gates_direct_knowledge_runtime_branch() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "_knowledge_intercept = (" in chat
    assert "should_intercept_workspace_knowledge(message)" in chat
    assert 'if _agent_langgraph_architecture_enabled and chat_mode == "agent" and _knowledge_runtime_enabled and _knowledge_intercept:' in chat
    assert 'if chat_mode == "agent" and _knowledge_runtime_enabled and _knowledge_intercept:' not in chat


def test_agent_langgraph_toggle_has_default_setting() -> None:
    settings_py = (ROOT / "src" / "settings.py").read_text(encoding="utf-8")

    assert '"agent_langgraph_enabled": True' in settings_py


def test_agent_langgraph_toggle_is_accepted_by_settings_endpoint() -> None:
    auth_routes = (ROOT / "routes" / "auth_routes.py").read_text(encoding="utf-8")

    assert '"agent_langgraph_enabled"' in auth_routes
    assert "_coerce_bool_setting" in auth_routes
    assert 'current["agent_langgraph_enabled"] = _coerce_bool_setting(body.get("agent_langgraph_enabled"))' in auth_routes


def test_langgraph_toggle_quarantines_daily_driver_downgrade_and_auto_escalation() -> None:
    chat = (ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert 'if str(chat_mode or "").strip().lower() != "agent":\n        return True' not in chat

    assert 'if (\n            _agent_langgraph_architecture_enabled\n            and chat_mode == "agent"\n            and not _clear_tool_or_runtime_intent' in chat

    assert 'if _agent_langgraph_architecture_enabled and chat_mode == "chat" and _tool_intent and _tool_intent.needs_tools:' in chat
    assert 'if chat_mode == "chat" and _tool_intent and _tool_intent.needs_tools:' not in chat


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
