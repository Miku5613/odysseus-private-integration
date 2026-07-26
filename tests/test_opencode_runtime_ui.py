from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runtime_source() -> str:
    return (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")


def test_runtime_control_exposes_three_peer_runtime_states_and_toggles():
    source = runtime_source()

    assert "new Set(['legacy', 'langgraph', 'opencode'])" in source
    assert "Enable LangGraph" in source
    assert "Enable OpenCode" in source
    assert 'role="switch"' in source
    assert "runtime-toggle-track" in source
    assert "Legacy remains available from the composer runtime control." in source


def test_runtime_selector_remains_mutually_exclusive():
    source = runtime_source()

    assert "if (openCode && langGraph.checked) openCode.checked = false" in source
    assert "if (langGraph && openCode.checked) langGraph.checked = false" in source
    assert "await setCurrentRuntime('legacy')" in source


def test_runtime_panel_is_inserted_next_to_indexes():
    source = runtime_source()

    assert "findIndexesTab" in source
    assert "indexesTab.parentNode.insertBefore(runtimeTab, indexesTab)" in source
    assert "indexesPanel.parentNode.insertBefore(runtimePanel, indexesPanel)" in source
    assert "panel.className = 'hidden admin-only runtime-settings-panel'" in source


def test_runtime_indicator_lives_beside_agent_chat_toggle_and_cycles_all_states():
    source = runtime_source()

    assert "document.querySelector('.chat-input-right')" in source
    assert "right.insertBefore(button, modeToggle)" in source
    assert "const order = ['legacy', 'langgraph', 'opencode']" in source
    assert "const selectorEnabled = getDefaultRuntime() !== 'legacy'" in source
    assert "const visible = isAgentMode() && selectorEnabled" in source
    assert "runtime-indicator-x" not in source


def test_opencode_model_workspace_and_restart_controls_exist():
    source = runtime_source()

    assert "runtime-opencode-model" in source
    assert "runtime-opencode-workspace" in source
    assert "Use active chat model" in source
    assert "Use active workspace" in source
    assert "Apply &amp; restart OpenCode" in source
    assert "/api/opencode/runtime/server/restart" in source


def test_opencode_server_and_session_controls_exist():
    source = runtime_source()

    assert "Restart current session" in source
    assert "Close OpenCode" in source
    assert "Open log terminal" in source
    assert "/api/opencode/runtime/server/stop" in source
    assert "/api/opencode/runtime/server/log-terminal" in source
    assert "/api/opencode/runtime/conversation/" in source


def test_opencode_ui_uses_native_bridge_without_fallback():
    source = runtime_source()

    assert "/api/opencode/runtime/stream" in source
    assert "/api/opencode/runtime/cancel/" in source
    assert "/api/opencode/runtime/health" in source
    assert "never falls back" in source
    assert "/api/chat/stream" not in source


def test_chat_submit_branches_to_opencode_and_stop_targets_active_run():
    source = (
        ROOT / "static" / "js" / "chat.js"
    ).read_text(encoding="utf-8")

    assert "import runtimeControl from './runtimeControl.js'" in source
    assert "runtimeControl.isOpenCodeStreaming()" in source
    assert "runtimeControl.stopActiveOpenCodeRun()" in source
    assert "runtimeControl.streamOpenCodeTurn({" in source
    assert "await runtimeControl.prepareNativeRuntime(_selectedRuntime)" in source


def test_runtime_ui_styles_include_switches_and_right_side_indicator():
    source = (
        ROOT / "static" / "style.css"
    ).read_text(encoding="utf-8")

    assert ".runtime-toggle-track" in source
    assert ".runtime-toggle-thumb" in source
    assert ".runtime-config-grid" in source
    assert ".runtime-indicator-btn.runtime-legacy" in source
    assert ".agent-thread-content" in source
    assert ".opencode-agent-thread" not in source


def test_backend_routes_expose_managed_runtime_controls():
    source = (
        ROOT / "routes" / "opencode_runtime_routes.py"
    ).read_text(encoding="utf-8")

    assert '@router.post("/server/restart")' in source
    assert '@router.post("/server/stop")' in source
    assert '@router.post("/server/log-terminal")' in source
    assert "OpenCodeRestartRequest" in source


def test_log_terminal_targets_current_odysseus_session():
    runtime_source = (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")
    route_source = (
        ROOT / "routes" / "opencode_runtime_routes.py"
    ).read_text(encoding="utf-8")

    assert "sessionModule.getCurrentSessionId()" in runtime_source
    assert "odysseus_session_id: sessionId" in runtime_source
    assert "payload.scope === 'session'" in runtime_source
    assert "class OpenCodeLogTerminalRequest" in route_source
    assert "request.odysseus_session_id if request else" in route_source


def test_runtime_logs_are_project_relative_and_split_by_scope():
    manager_source = (
        ROOT / "src" / "opencode_runtime" / "process_manager.py"
    ).read_text(encoding="utf-8")

    assert 'Path(__file__).resolve().parents[2]' in manager_source
    assert 'self._project_root / "logs" / "OpenCode"' in manager_source
    assert '/ "server"' in manager_source
    assert '/ "sessions"' in manager_source
    assert '"Exclusions"' not in manager_source
    assert '"OpenCodeRuntime"' not in manager_source


def test_opencode_tools_reuse_native_odysseus_tool_thread_ui():
    runtime_source = (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")
    chat_source = (
        ROOT / "static" / "js" / "chat.js"
    ).read_text(encoding="utf-8")

    # OpenCode keeps its own event transport but renders through the native
    # Odysseus tool-thread DOM contract. No parallel bubble component.
    assert "thread.className = 'agent-thread streaming'" in runtime_source
    assert "node.className = 'agent-thread-node running'" in runtime_source
    assert "details.className = 'agent-thread-content'" in runtime_source
    assert "summary.className = 'agent-thread-cmd'" in runtime_source
    assert "result.className = 'agent-tool-output'" in runtime_source

    assert "agent-thread-details" not in runtime_source
    assert "opencode-tool-node" not in runtime_source
    assert "opencode-tool-summary" not in runtime_source
    assert "opencode-tool-result" not in runtime_source

    # Reuse the already-existing delegated fold handler from chat.js.
    assert "e.target.closest('.agent-thread-header')" in chat_source
    assert "header.closest('.agent-thread-node')" in chat_source
    assert "node.classList.toggle('open')" in chat_source


def test_permission_handler_settings_and_session_routes_exist():
    runtime_source = (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")
    route_source = (
        ROOT / "routes" / "opencode_runtime_routes.py"
    ).read_text(encoding="utf-8")

    assert "Ask for Approval" in runtime_source
    assert "Allow safe workspace file handling" in runtime_source
    assert "runtime-opencode-permission-mode" in runtime_source
    assert "runtime-clear-permission-approvals-btn" in runtime_source
    assert "permission_mode: permissionMode()" in runtime_source

    assert '@router.get("/permissions/{odysseus_session_id}")' in route_source
    assert '@router.put("/permissions/{odysseus_session_id}/mode")' in route_source
    assert '@router.post("/permissions/reply")' in route_source
    assert (
        '@router.delete("/permissions/{odysseus_session_id}/approvals")'
        in route_source
    )


def test_permission_prompt_reuses_native_tool_thread_ui():
    runtime_source = (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")

    assert "function createPermissionNode(context, event)" in runtime_source
    assert "node.className = 'agent-thread-node open waiting'" in runtime_source
    assert "content.className = 'agent-thread-content'" in runtime_source
    assert "summary.className = 'agent-thread-cmd'" in runtime_source
    assert "Allow once" in runtime_source
    assert "Allow for this chat" in runtime_source
    assert "Reject" in runtime_source
    assert "runtime.permission_requested" in runtime_source
    assert "runtime.permission_resolved" in runtime_source


def test_permission_handler_does_not_use_opencode_always_reply():
    service_source = (
        ROOT / "src" / "opencode_runtime" / "service.py"
    ).read_text(encoding="utf-8")
    transport_source = (
        ROOT / "src" / "opencode_runtime" / "transport.py"
    ).read_text(encoding="utf-8")

    assert '"user_session_approval"' in service_source
    assert 'reply = "reject" if normalized == "reject" else "once"' in service_source
    assert '"odysseus_session_approval"' in service_source
    assert 'f"/permission/{encoded}/reply{suffix}"' in transport_source


def test_workspace_policy_keeps_risky_actions_gated():
    service_source = (
        ROOT / "src" / "opencode_runtime" / "service.py"
    ).read_text(encoding="utf-8")

    assert '"external_directory"' not in (
        service_source[
            service_source.index("WORKSPACE_SAFE_ACTIONS"):
            service_source.index("SENSITIVE_PATH_MARKERS")
        ]
    )
    assert '"bash"' not in (
        service_source[
            service_source.index("WORKSPACE_SAFE_ACTIONS"):
            service_source.index("SENSITIVE_PATH_MARKERS")
        ]
    )
    assert "self._resource_path_within_workspace" in service_source
    assert "self._resource_is_sensitive" in service_source


def test_runtime_console_redesign_preserves_controls_and_ids():
    source = runtime_source()

    for marker in (
        "runtime-selector-card",
        "runtime-console-card",
        "runtime-default-pill",
        "runtime-health-pill",
        "runtime-console-layout",
        "runtime-environment-section",
        "runtime-permission-section",
        "runtime-behavior-section",
        "runtime-action-dock",
        "runtime-primary-action",
    ):
        assert marker in source

    for control_id in (
        "set-runtime-langgraph",
        "set-runtime-opencode",
        "runtime-opencode-model",
        "runtime-opencode-workspace",
        "runtime-opencode-permission-mode",
        "set-opencode-reuse-session",
        "set-opencode-reset-on-switch",
        "runtime-health-btn",
        "runtime-reset-session-btn",
        "runtime-stop-server-btn",
        "runtime-log-terminal-btn",
    ):
        assert control_id in source


def test_composer_has_quick_opencode_session_reset():
    source = runtime_source()

    assert "runtime-quick-reset-btn" in source
    assert "createRuntimeQuickReset()" in source
    assert "Start a fresh OpenCode session" in source
    assert "await resetOpenCodeSession(id)" in source
    assert "resetVisible = visible && runtime === 'opencode'" in source
    assert "_quickResetButton.disabled = !!_activeRun" in source


def test_runtime_console_styles_are_native_single_block():
    style = (ROOT / "static" / "style.css").read_text(
        encoding="utf-8"
    )

    assert style.count(
        "/* ── OpenCode / Agent runtime control v3"
    ) == 1
    assert style.count(
        "/* ── End OpenCode / Agent runtime control v3"
    ) == 1
    assert ".runtime-console-layout" in style
    assert ".runtime-action-dock" in style
    assert ".runtime-quick-reset-btn" in style
    assert "@keyframes runtime-spin" in style

def test_opencode_terminal_logs_use_active_chat_and_descriptive_title():
    runtime_source = (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")
    manager_source = (
        ROOT / "src" / "opencode_runtime" / "process_manager.py"
    ).read_text(encoding="utf-8")

    assert "odysseus_session_id: sessionId" in runtime_source
    assert "active chat" in runtime_source
    assert "Following {scope_label} log for {model_label}" in manager_source
    assert "Odysseus OpenCode — {scope_label} — {model_label}" in manager_source
