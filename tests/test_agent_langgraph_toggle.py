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
    assert "_runtime_decision = route_agent_runtime(message) if _agent_langgraph_enabled else None" in chat
    assert "_agent_langgraph_enabled" in chat
    assert "_runtime_decision is not None" in chat
