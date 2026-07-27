from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUI_PATH = ROOT / "src/opencode_runtime/session_manager_tui.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "odysseus_opencode_session_manager_tui",
        TUI_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tui_source_parses_and_uses_control_apis():
    source = TUI_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'sys.stdout.reconfigure(encoding="utf-8"' in source
    assert "OpenCode Session Manager" in source
    assert "/global/health" in source
    assert "/session/status" in source
    assert "/session/{encoded}/command" in source
    assert "/command" in source
    assert "/mcp" in source
    assert "/config" in source
    assert "tail -f" not in source
    assert "Get-Content -Wait" not in source


def test_slash_command_parser():
    module = load_module()
    assert module.parse_slash_command("/new") == ("new", "")
    assert module.parse_slash_command("/compact now") == ("compact", "now")


def test_terminal_layout_contains_required_panels():
    module = load_module()
    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    app = module.SessionManagerTUI(
        client,
        session_id="ses_test",
        model="qwen3.5:9b-q6",
    )
    app.snapshot.server_live = True
    app.snapshot.version = "1.18.5"
    rendered = app.render()
    assert "OpenCode Session Manager" in rendered
    for label in (
        "SESSION",
        "PERMISSIONS",
        "LIFECYCLE",
        "MCP",
        "PLUGINS",
        "EXTENSIONS",
        "COMMAND",
    ):
        assert label in rendered
    assert "Tab next" in rendered


def test_process_manager_launches_custom_terminal_client():
    source = (
        ROOT / "src/opencode_runtime/process_manager.py"
    ).read_text(encoding="utf-8")
    assert '"session_manager_tui.py"' in source
    assert '"OpenCode Session Manager"' in source
    assert 'shutil.which("wt.exe")' in source
    assert '"odysseus-custom-tui"' in source
    assert "sys.executable" in source

def test_command_history_and_autocomplete():
    module = load_module()
    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    app = module.SessionManagerTUI(
        client,
        session_id="ses_test",
        model="qwen3.5:9b-q6",
    )
    app.snapshot.commands = [
        {"name": "status"},
        {"name": "share"},
    ]

    assert app._autocomplete_command("sta") == "status"
    assert app._autocomplete_command("per") == "permissions"
    assert app._autocomplete_command("s") == "s"

    app._remember_command("/status")
    app._remember_command("/new Demo")
    assert app._history_command(-1) == "new Demo"
    assert app._history_command(-1) == "status"
    assert app._history_command(1) == "new Demo"
    assert app._history_command(1) == ""


def test_history_is_bounded_and_skips_adjacent_duplicates():
    module = load_module()
    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    app = module.SessionManagerTUI(
        client,
        session_id="ses_test",
        model="qwen3.5:9b-q6",
    )

    app._remember_command("/status")
    app._remember_command("/status")
    assert app.command_history == ["/status"]

    for index in range(120):
        app._remember_command(f"/command-{index}")
    assert len(app.command_history) == 100
    assert app.command_history[-1] == "/command-119"

def test_mapping_aware_control_commands_use_odysseus_bridge():
    module = load_module()

    class FakeControl:
        def __init__(self):
            self.calls = []

        def get(self, path):
            self.calls.append(("GET", path, None))
            return module.ApiResult(
                True,
                200,
                {
                    "odysseus_session_id": "chat-1",
                    "opencode_session_id": "ses-new",
                    "model_id": "qwen3.5:9b-q6",
                    "permission": {
                        "mode": "ask",
                        "pending": [],
                        "approvals": [],
                    },
                },
            )

        def post(self, path, payload=None):
            self.calls.append(("POST", path, payload))
            if path == "/new":
                return module.ApiResult(
                    True,
                    200,
                    {"opencode_session_id": "ses-new"},
                )
            if path == "/restart":
                return module.ApiResult(
                    True,
                    200,
                    {"opencode_session_id": "ses-restarted"},
                )
            return module.ApiResult(True, 200, {"ok": True})

        def put(self, path, payload=None):
            self.calls.append(("PUT", path, payload))
            return module.ApiResult(True, 200, {"ok": True})

        def delete(self, path):
            self.calls.append(("DELETE", path, None))
            return module.ApiResult(True, 200, {"ok": True})

    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    control = FakeControl()
    app = module.SessionManagerTUI(
        client,
        session_id="ses-old",
        model="qwen3.5:9b-q6",
        control_client=control,
    )
    app.snapshot.control = {"odysseus_session_id": "chat-1"}

    assert app.execute("/new Demo") is True
    assert app.session_id == "ses-new"
    assert ("POST", "/new", {"title": "Demo"}) in control.calls

    assert app.execute("/permission workspace") is True
    assert (
        "PUT",
        "/permissions/mode",
        {"mode": "workspace"},
    ) in control.calls

    assert app.execute("/restart") is True
    assert app.session_id == "ses-new"
    assert any(call[:2] == ("POST", "/restart") for call in control.calls)

    assert app.execute("/stop") is False
    assert app.exit_requested is True
    assert app.session_id == ""
    assert any(call[:2] == ("POST", "/stop") for call in control.calls)


def test_tui_control_token_is_not_a_command_line_argument():
    source = TUI_PATH.read_text(encoding="utf-8")
    assert "ODYSSEUS_SESSION_MANAGER_TOKEN" in source
    assert '--control-token' not in source
    assert "Bearer {self.token}" in source

def test_package_commands_use_the_secure_control_bridge():
    module = load_module()

    class FakeControl:
        def __init__(self):
            self.calls = []

        def post(self, path, payload=None):
            self.calls.append(("POST", path, payload))
            return module.ApiResult(True, 200, {"opened": True})

    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    control = FakeControl()
    app = module.SessionManagerTUI(
        client,
        session_id="ses-test",
        model="qwen3.5:9b-q6",
        control_client=control,
    )

    assert app.execute("/mcp-list") is True
    assert (
        "POST",
        "/packages/open",
        {"package": "mcp", "action": "list", "argument": ""},
    ) in control.calls

    assert app.execute("/mcp-auth github") is True
    assert (
        "POST",
        "/packages/open",
        {
            "package": "mcp",
            "action": "auth",
            "argument": "github",
        },
    ) in control.calls

    assert app.execute("/plugin-install opencode-example") is True
    assert (
        "POST",
        "/packages/open",
        {
            "package": "plugin",
            "action": "install",
            "argument": "opencode-example",
        },
    ) in control.calls


def test_package_panels_explain_supported_controls():
    module = load_module()
    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    app = module.SessionManagerTUI(
        client,
        session_id="ses-test",
        model="qwen3.5:9b-q6",
    )

    app.tab_index = module.TABS.index("MCP")
    assert "/mcp-add" in "\n".join(app.content_lines(90))

    app.tab_index = module.TABS.index("PLUGINS")
    assert "/plugin-install" in "\n".join(app.content_lines(90))

    app.tab_index = module.TABS.index("EXTENSIONS")
    assert "read-only" in "\n".join(app.content_lines(90)).lower()

def test_windows_prompt_imports_msvcrt_and_renders_an_input_row():
    source = TUI_PATH.read_text(encoding="utf-8")
    assert "import msvcrt" in source
    assert "self.command_buffer" in source
    assert "Press / to enter a command" in source
    assert "HOME + self.render()" in source

def test_lazygit_layout_uses_dashboard_panels_and_alternate_screen(tmp_path):
    module = load_module()
    log_path = tmp_path / "session.log"
    log_path.write_text("SERVER_STARTED\nSESSION_RESTORED\n", encoding="utf-8")
    client = module.OpenCodeClient("http://127.0.0.1:4096", str(ROOT))
    app = module.SessionManagerTUI(client, session_id="ses-test", model="qwen3.5:9b-q6", log_path=str(log_path))
    app._terminal_size_override = (120, 38)
    app.snapshot.server_live = True
    app._refresh_logs()
    rendered = app.render()
    for label in ("OpenCode Session Manager", "OPTIONS", "ACTIVE PANEL", "SERVER ACTIONS", "COMMAND", "TERMINAL LOGS", "SHORTCUTS", "Restart Server", "Close Server", "SERVER_STARTED"):
        assert label in rendered
    source = TUI_PATH.read_text(encoding="utf-8")
    for marker in ("ALT_ENTER", "ALT_EXIT", "MOUSE_ENABLE", "MOUSE_DISABLE", "RESIZE_DEFAULT", "HOME + self.render()"):
        assert marker in source


def test_command_input_is_integrated_into_command_panel():
    module = load_module()
    client = module.OpenCodeClient("http://127.0.0.1:4096", str(ROOT))
    app = module.SessionManagerTUI(client, session_id="ses-test", model="qwen3.5:9b-q6")
    app._terminal_size_override = (100, 32)
    assert app._handle_key("/") is True
    for key in "review":
        assert app._handle_key(key) is True
    rendered = app.render()
    assert "> /review" in rendered
    assert app.focus_name == "command"


def test_panel_scroll_offsets_are_independent():
    module = load_module()
    client = module.OpenCodeClient("http://127.0.0.1:4096", str(ROOT))
    app = module.SessionManagerTUI(client, session_id="ses-test", model="qwen3.5:9b-q6")
    app._scroll("active", 3)
    app._scroll("logs", 7)
    assert app.scroll_offsets["active"] == 3
    assert app.scroll_offsets["logs"] == 7

def test_short_viewport_frame_fits_without_terminal_edge_wrap():
    module = load_module()
    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    app = module.SessionManagerTUI(
        client,
        session_id="ses-test",
        model="qwen3.5:9b-q6",
    )
    app._terminal_size_override = (96, 24)
    rendered = module.ANSI_RE.sub("", app.render())
    lines = rendered.splitlines()

    assert len(lines) <= 23
    assert all(len(line) <= 95 for line in lines)
    for marker in (
        "OpenCode Session Manager",
        "OPTIONS",
        "ACTIVE PANEL",
        "SERVER ACTIONS",
        "COMMAND",
        "TERMINAL LOGS",
        "SHORTCUTS",
    ):
        assert marker in rendered


def test_frame_redraw_clears_stale_terminal_content():
    source = TUI_PATH.read_text(encoding="utf-8")
    assert 'CLEAR_TO_END = "\\x1b[J"' in source
    assert "HOME + self.render() + CLEAR_TO_END" in source

def test_launch_ticket_supplies_auth_and_is_deleted(tmp_path):
    module = load_module()
    ticket = tmp_path / "ticket.json"
    ticket.write_text(
        module.json.dumps({
            "username": "opencode",
            "password": "secret",
            "control_token": "bridge-token",
        }),
        encoding="utf-8",
    )

    value = module.load_launch_ticket(str(ticket))
    assert value["password"] == "secret"
    assert value["control_token"] == "bridge-token"
    assert not ticket.exists()


def test_confirmed_stop_exits_tui_and_stops_server():
    module = load_module()

    class FakeControl:
        def post(self, path, payload=None):
            assert path == "/stop"
            return module.ApiResult(
                True,
                200,
                {"ok": True, "stopped": True},
            )

    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    app = module.SessionManagerTUI(
        client,
        session_id="ses-test",
        model="qwen3.5:9b-q6",
        control_client=FakeControl(),
    )
    app._set_focus("actions")
    app.action_index = 1

    assert app._invoke_action("stop") is True
    assert app.confirm_action == "stop"
    assert app._handle_key("\r") is False
    assert app.exit_requested is True
    assert app.session_id == ""


def test_tui_uses_tmux_catppuccin_truecolor_palette():
    source = TUI_PATH.read_text(encoding="utf-8")
    for marker in (
        "\\x1b[48;2;30;30;46m",
        "\\x1b[48;2;36;36;52m",
        "\\x1b[38;2;137;180;250m",
        "\\x1b[38;2;166;227;161m",
        "PANEL_BG",
    ):
        assert marker in source


def test_ansi_aware_clip_preserves_visible_action_labels():
    module = load_module()
    value = module.YELLOW + " [ Restart Server ] " + module.RESET
    clipped = module.clip(value, 30)
    assert "Restart Server" in clipped
    assert len(module.ANSI_RE.sub("", clipped)) <= 30

def test_handoff_refreshes_auth_and_control(tmp_path):
    module = load_module()
    ticket = tmp_path / "ticket.json"
    ticket.write_text(
        module.json.dumps({
            "username": "opencode",
            "password": "secret",
            "control_token": "bridge-token",
        }),
        encoding="utf-8",
    )
    log_path = tmp_path / "session.log"
    log_path.write_text("READY\n", encoding="utf-8")
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        module.json.dumps({
            "launch_ticket": str(ticket),
            "server": "http://127.0.0.1:4096",
            "workspace": str(ROOT),
            "model": "qwen3.5:9b-q6",
            "session_id": "ses-refreshed",
            "control_url": "http://127.0.0.1:9000",
            "log_path": str(log_path),
        }),
        encoding="utf-8",
    )
    client = module.OpenCodeClient("http://127.0.0.1:1", str(ROOT))
    app = module.SessionManagerTUI(
        client,
        session_id="ses-old",
        model="old",
        handoff_path=str(handoff),
    )
    app.refresh = lambda: None
    assert app._consume_handoff() is True
    assert app.client.username == "opencode"
    assert app.client.password == "secret"
    assert app.session_id == "ses-refreshed"
    assert app.control_client.token == "bridge-token"
    assert not handoff.exists()
    assert not ticket.exists()


def test_logs_panel_can_be_maximized_and_restored(tmp_path):
    module = load_module()
    log_path = tmp_path / "session.log"
    log_path.write_text(
        "\n".join(f"line-{index}" for index in range(80)),
        encoding="utf-8",
    )
    client = module.OpenCodeClient("http://127.0.0.1:4096", str(ROOT))
    app = module.SessionManagerTUI(
        client,
        session_id="ses-test",
        model="qwen3.5:9b-q6",
        log_path=str(log_path),
    )
    app._terminal_size_override = (110, 34)
    app._refresh_logs()
    app._set_focus("logs")
    assert app._handle_key("z") is True
    assert app.zoom_panel == "logs"
    rendered = module.ANSI_RE.sub("", app.render())
    assert "TERMINAL LOGS" in rendered
    assert "Z / Esc restore dashboard" in rendered
    assert app._handle_key("z") is True
    assert app.zoom_panel == ""


def test_panel_reset_preserves_background():
    module = load_module()
    assert module.RESET.startswith(module.TERMINAL_RESET)
    assert module.PANEL_BG in module.RESET
    assert module.TEXT in module.RESET

def test_native_client_sends_basic_auth(
    monkeypatch,
):
    module = load_module()
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            traceback,
        ):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(
        request,
        timeout,
    ):
        captured.update({
            str(key).casefold(): str(value)
            for key, value
            in request.header_items()
        })
        return Response()

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
        username="opencode",
        password="secret",
    )
    result = client.get(
        "/global/health"
    )
    assert result.ok is True
    assert captured["authorization"] == (
        "Basic b3BlbmNvZGU6c2VjcmV0"
    )


def test_control_client_sends_bearer_auth(
    monkeypatch,
):
    module = load_module()
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            traceback,
        ):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(
        request,
        timeout,
    ):
        captured.update({
            str(key).casefold(): str(value)
            for key, value
            in request.header_items()
        })
        return Response()

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    client = (
        module.OdysseusControlClient(
            "http://127.0.0.1:8000",
            "control-secret",
        )
    )
    result = client.get("/state")
    assert result.ok is True
    assert captured["authorization"] == (
        "Bearer control-secret"
    )


def test_ctrl_arrow_keys_move_panel_focus():
    module = load_module()
    client = module.OpenCodeClient(
        "http://127.0.0.1:4096",
        str(ROOT),
    )
    app = module.SessionManagerTUI(
        client,
        session_id="ses-test",
        model="qwen3.5:9b-q6",
    )

    assert app.focus_name == "options"
    assert app._handle_key(
        "CTRL_RIGHT"
    ) is True
    assert app.focus_name == "active"
    assert app._handle_key(
        "CTRL_DOWN"
    ) is True
    assert app.focus_name == "command"
    assert app._handle_key(
        "CTRL_LEFT"
    ) is True
    assert app.focus_name == "active"


def test_windows_ctrl_arrow_extended_codes():
    module = load_module()
    app = module.SessionManagerTUI(
        module.OpenCodeClient(
            "http://127.0.0.1:4096",
            str(ROOT),
        ),
        session_id="ses-test",
        model="qwen3.5:9b-q6",
    )

    class FakeMsvcrt:
        def __init__(self, values):
            self.values = iter(values)

        def getwch(self):
            return next(self.values)

        def kbhit(self):
            return False

    cases = (
        (115, "CTRL_LEFT"),
        (116, "CTRL_RIGHT"),
        (141, "CTRL_UP"),
        (145, "CTRL_DOWN"),
    )
    for code, expected in cases:
        actual = app._read_windows_key(
            FakeMsvcrt([
                "\xe0",
                chr(code),
            ])
        )
        assert actual == expected




def test_no_stale_alt_arrow_contract():
    source = TUI_PATH.read_text(
        encoding="utf-8"
    )

    retired_runtime_markers = (
        "Alt+" + "Arrow",
        "Alt+" + "Arrows",
        "ALT_" + "LEFT",
        "ALT_" + "RIGHT",
        "_windows_" + "alt_pressed",
    )
    assert not any(
        marker in source
        for marker in retired_runtime_markers
    )

    test_source = Path(__file__).read_text(
        encoding="utf-8"
    )
    test_names = {
        line.strip()
        .removeprefix("def ")
        .split("(", 1)[0]
        for line in test_source.splitlines()
        if line.startswith("def test_")
    }
    retired_prefixes = (
        "test_" + "alt_arrow",
        "test_real_windows_" + "alt_arrow",
    )
    assert not any(
        name.startswith(retired_prefixes)
        for name in test_names
    )

def test_header_drag_watcher_does_not_depend_on_terminal_mouse_events():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "opencode_runtime" / "session_manager_tui.py"
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.get_source_segment(source, node) or ""

    watcher = functions["_start_window_drag_watcher"]
    mouse = functions["_handle_mouse"]
    for marker in (
        "GetAsyncKeyState",
        "GetForegroundWindow",
        "GetWindowRect",
        "GetCursorPos",
        "SetWindowPos",
        "CASCADIA",
        "threading.Thread",
        "vk_shift",
    ):
        assert marker.lower() in watcher.lower()
    assert "self._drag_window()" not in mouse
    assert "if button == 0 and y <= 4:" in mouse


def test_header_drag_watcher_uses_run_try_finally_lifecycle():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "opencode_runtime" / "session_manager_tui.py"
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    run_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_node = node
            break
    assert run_node is not None

    run_source = ast.get_source_segment(source, run_node) or ""
    assert "self._start_window_drag_watcher()" in run_source
    assert "self._stop_window_drag_watcher()" in run_source
    outer_try = [
        item
        for item in run_node.body
        if isinstance(item, ast.Try) and item.finalbody
    ]
    assert len(outer_try) == 1
    first_final = ast.get_source_segment(source, outer_try[0].finalbody[0]) or ""
    assert "self._stop_window_drag_watcher()" in first_final


def test_session_manager_terminal_profile_hides_scrollbar_and_matches_tui():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "opencode_runtime" / "process_manager.py"
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.get_source_segment(source, node) or ""

    profile = functions["_ensure_session_manager_terminal_profile"]
    launch = functions["open_session_manager_terminal"]
    assert "scrollbarState" in profile
    assert "\"hidden\"" in profile
    assert "historySize" in profile
    assert "Fragments" in profile
    assert 'Odysseus OpenCode Session Manager' in profile
    assert '121,37' in launch
    assert "\"--profile\"" in launch
    assert "_ensure_session_manager_terminal_profile()" in launch
