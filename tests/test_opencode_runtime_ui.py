from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runtime_source() -> str:
    return (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")






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




def test_composer_has_quick_opencode_session_reset():
    source = runtime_source()

    assert "runtime-quick-reset-btn" in source
    assert "createRuntimeQuickReset()" in source
    assert "Start OpenCode or open OpenCode Session Manager" in source
    assert "await resetOpenCodeSession(id)" in source
    assert "managerVisible = visible && runtime === 'opencode'" in source
    assert "!!_activeRun || _openCodeSessionManagerBusy" in source



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

def test_live_session_manager_controls_are_native_sized_and_single_instance():
    runtime_js = runtime_source()
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    manager_source = (
        ROOT / "src" / "opencode_runtime" / "process_manager.py"
    ).read_text(encoding="utf-8")

    assert "_quickResetButton.classList.toggle('active', ready)" in runtime_js
    assert "_indicator.classList.toggle(" in runtime_js
    assert "height: 32px !important" in style
    assert "_focus_existing_session_manager" in manager_source
    assert '"existing-window"' in manager_source

def test_prepare_materializes_pending_chat_before_reading_session_id():
    source = runtime_source()
    pending_index = source.index("sessionModule.hasPendingChat")
    current_index = source.index("sessionModule.getCurrentSessionId", pending_index)
    assert pending_index < current_index
    assert "await sessionModule.materializePendingSession()" in source


def test_model_selection_accepts_unique_registry_suffix_match():
    source = runtime_source()
    assert "normalizeOpenCodeModelId" in source
    assert "modelId.endsWith(`/${visibleModel}`)" in source
    assert "visibleModel.endsWith(`/${modelId}`)" in source
    assert "matches.length === 1" in source

def test_session_manager_control_routes_bypass_outer_auth_middleware():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    app_path = root / "app.py"
    app_source = app_path.read_text(encoding="utf-8-sig")
    app_tree = ast.parse(app_source, filename=str(app_path))

    prefixes = None
    dispatch_source = ""
    for node in ast.walk(app_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AUTH_EXEMPT_PREFIXES":
                    prefixes = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and node.name == "AuthMiddleware":
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "dispatch":
                    dispatch_source = ast.get_source_segment(app_source, item) or ""

    assert "/api/opencode/runtime/session-manager-control" in (prefixes or [])
    assert "_is_auth_exempt(path)" in dispatch_source
    assert "request.cookies.get(SESSION_COOKIE)" in dispatch_source
    assert dispatch_source.index("_is_auth_exempt(path)") < dispatch_source.index(
        "request.cookies.get(SESSION_COOKIE)"
    )

    routes_path = root / "routes" / "opencode_runtime_routes.py"
    routes_source = routes_path.read_text(encoding="utf-8-sig")
    routes_tree = ast.parse(routes_source, filename=str(routes_path))
    helper_source = ""
    for node in ast.walk(routes_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_session_manager_control":
            helper_source = ast.get_source_segment(routes_source, node) or ""
            break

    lowered = helper_source.lower()
    assert "authorization" in lowered
    assert "scheme.lower()" in helper_source
    assert '"bearer"' in lowered
    assert "resolve_session_manager_control" in helper_source

def test_runtime_settings_ui_is_minimal_and_lifecycle_focused():
    source = runtime_source()
    panel_start = source.index("function createRuntimePanel() {")
    panel_end = source.index("\nfunction activateRuntimePanel()", panel_start)
    panel = source[panel_start:panel_end]

    assert 'id="set-runtime-default"' in panel
    assert '<option value="legacy">Legacy</option>' in panel
    assert '<option value="langgraph">LangGraph</option>' in panel
    assert '<option value="opencode">OpenCode</option>' in panel
    assert "normalizeRuntime(runtimeSelect.value, 'legacy')" in source
    assert "Conversation Lifecycle" in panel
    assert "Reuse conversation per chat" in panel
    assert "Reset after switching runtime" in panel
    assert "<h3>OpenCode Session Manager</h3>" in panel
    assert 'id="runtime-session-manager-settings-btn"' in panel

    for removed in (
        "set-runtime-langgraph",
        "set-runtime-opencode",
        "runtime-health-pill",
        "runtime-environment-section",
        "runtime-permission-section",
        "runtime-action-dock",
        "Apply changes &amp; restart",
    ):
        assert removed not in panel


def test_runtime_settings_minimal_styles_exist_once():
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    assert style.count("/* ── Runtime settings minimal layout v1") == 1
    assert ".runtime-select-row" in style
    assert ".runtime-lifecycle-panel" in style
    assert ".runtime-session-manager-copy h3" in style
    assert ".runtime-session-manager-launch" in style

def test_runtime_session_manager_control_is_visible_and_wired():
    runtime = runtime_source()
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    assert "<h3>OpenCode Session Manager</h3>" in runtime
    assert '<button id="runtime-session-manager-settings-btn"' in runtime
    assert "const sessionManagerButton = document.getElementById(" in runtime
    assert "sessionManagerButton.addEventListener('click'" in runtime
    assert "handleOpenCodeSessionManagerClick().catch(() => {});" in runtime
    assert ".runtime-session-manager-copy h3" in style
    assert "font-size: 14px;" in style
    assert "var(--accent-primary)" in style
    assert "background:" in style
    assert "border:" in style
