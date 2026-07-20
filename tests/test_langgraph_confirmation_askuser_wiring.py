from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

graph = (ROOT / "src/agent_runtime/graph.py").read_text(
    encoding="utf-8",
    errors="replace",
)
streaming = (ROOT / "src/agent_runtime/streaming.py").read_text(
    encoding="utf-8",
    errors="replace",
)
runner = (ROOT / "src/agent_runtime/legacy_runner.py").read_text(
    encoding="utf-8",
    errors="replace",
)
adapter = (
    ROOT / "src/agent_runtime/legacy_sse_adapter.py"
).read_text(encoding="utf-8", errors="replace")
routes = (ROOT / "routes/chat_routes.py").read_text(
    encoding="utf-8",
    errors="replace",
)
chat = (ROOT / "static/js/chat.js").read_text(
    encoding="utf-8",
    errors="replace",
)
renderer = (
    ROOT / "static/js/chatRenderer.js"
).read_text(encoding="utf-8", errors="replace")
legacy = (ROOT / "src/agent_loop.py").read_text(
    encoding="utf-8",
    errors="replace",
)


def test_session_identity_reaches_live_graph():
    assert "session_id: str | None = None" in streaming
    assert (
        "build_graph(scope=scope, session_id=session_id)"
        in streaming
    )
    assert "session_id=session_id" in runner


def test_both_confirm_gates_register_pending_call():
    assert graph.count(
        "register_pending_confirmation("
    ) == 2
    assert "confirmation_available=(" in graph
    assert "fingerprint=(" in graph


def test_internal_auto_read_is_allow_only_gated():
    start = graph.index('name = "read_file_around"')
    gate = graph.index(
        "_auto_risk_decision = "
        "decide_tool_call_policy_action(",
        start,
    )
    invoke = graph.index(
        "content = tool_obj.invoke(args)",
        start,
    )

    assert start < gate < invoke
    assert (
        '_auto_risk_decision.action != "allow"'
        in graph[gate:invoke]
    )


def test_confirmation_reuses_ask_user_event_shape():
    assert (
        'event_type == "tool_confirmation_required"'
        in adapter
    )
    assert '"type": "ask_user"' in adapter
    assert (
        '"interaction_mode": "tool_confirmation"'
        in adapter
    )
    assert '"label": "Yes"' in adapter
    assert '"label": "No"' in adapter


def test_renderer_supports_secure_submit_override():
    assert "renderOptions.onSubmit" in renderer
    assert "renderOptions.allowOther === false" in renderer


def test_chat_confirmation_does_not_send_yes_as_user_message():
    assert (
        "_askPayload.interaction_mode === "
        "'tool_confirmation'"
        in chat
    )
    assert "/api/chat/tool_confirmation/" in chat
    assert "approved," in chat


def test_owner_checked_confirmation_route_exists():
    marker = (
        '"/api/chat/tool_confirmation/{session_id}"'
    )

    start = routes.index(marker)
    owner = routes.index(
        "_verify_session_owner(request, session_id)",
        start,
    )
    resolver = routes.index(
        "resolve_and_execute_tool_confirmation(",
        owner,
    )

    assert start < owner < resolver


def test_legacy_agent_loop_not_modified_for_confirmation():
    assert "register_pending_confirmation(" not in legacy
    assert (
        "resolve_and_execute_tool_confirmation"
        not in legacy
    )
