from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _class_method_names(relative: str, class_name: str) -> set[str]:
    tree = ast.parse(_source(relative))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} not found in {relative}")


def test_prepare_and_session_manager_routes_exist():
    source = _source("routes/opencode_runtime_routes.py")
    assert '@router.post("/prepare")' in source
    assert '@router.post("/server/session-manager-terminal")' in source
    assert "class OpenCodePrepareRequest" in source
    assert "class OpenCodeSessionManagerRequest" in source
    assert "_verify_opencode_session_owner(" in source


def test_service_owns_prepare_and_terminal_lifecycle():
    methods = _class_method_names(
        "src/opencode_runtime/service.py",
        "OpenCodeRuntimeService",
    )
    assert "prepare_runtime" in methods
    assert "open_session_manager_terminal" in methods

    source = _source("src/opencode_runtime/service.py")
    assert '"RUNTIME_PREPARED"' in source
    assert "config_matches" in source
    assert "await self.restart_server(" in source
    assert "await self._get_or_create_session(" in source
    assert "self.server_manager.open_session_manager_terminal" in source


def test_process_manager_launches_custom_terminal_tui():
    methods = _class_method_names(
        "src/opencode_runtime/process_manager.py",
        "OpenCodeServerManager",
    )
    assert "open_session_manager_terminal" in methods

    source = _source("src/opencode_runtime/process_manager.py")
    assert '"session_manager_tui.py"' in source
    assert '"OpenCode Session Manager"' in source
    assert 'shutil.which("wt.exe")' in source
    assert '"--session"' in source
    assert '"--workspace"' in source
    assert '"windows-terminal"' in source
    assert '"windows-console"' in source


def test_foundation_does_not_replace_log_terminal():
    source = _source("routes/opencode_runtime_routes.py")
    assert '@router.post("/server/log-terminal")' in source
    assert '@router.post("/server/session-manager-terminal")' in source

def test_secure_session_manager_control_bridge_contract():
    routes = _source("routes/opencode_runtime_routes.py")
    for route in (
        "/session-manager-control/state",
        "/session-manager-control/new",
        "/session-manager-control/conversation",
        "/session-manager-control/restart",
        "/session-manager-control/stop",
        "/session-manager-control/permissions/mode",
        "/session-manager-control/permissions/reply",
        "/session-manager-control/permissions/approvals",
        "/session-manager-control/release",
    ):
        assert route in routes
    assert 'scheme.lower() != "bearer"' in routes
    assert "control_base_url=control_base_url" in routes

    service = _source("src/opencode_runtime/service.py")
    for marker in (
        "SessionManagerControlGrant",
        "secrets.token_urlsafe(32)",
        "issue_session_manager_control",
        "resolve_session_manager_control",
        "release_session_manager_control",
        "replace_conversation_session",
        "restart_current_server",
    ):
        assert marker in service

    process = _source("src/opencode_runtime/process_manager.py")
    assert '"--control-url"' in process
    assert '"ODYSSEUS_SESSION_MANAGER_TOKEN"' in process
    assert '"--control-token"' not in process
    assert "loopback URL" in process

def test_package_control_bridge_uses_allowlisted_cli_actions():
    routes = _source("routes/opencode_runtime_routes.py")
    assert "/session-manager-control/packages/open" in routes
    assert "class OpenCodePackageControlRequest" in routes
    assert "await _service.open_package_terminal(" in routes

    service = _source("src/opencode_runtime/service.py")
    assert "async def open_package_terminal(" in service
    assert "self.server_manager.open_package_terminal" in service

    process = _source("src/opencode_runtime/process_manager.py")
    for marker in (
        '("mcp", "add")',
        '("mcp", "list")',
        '("mcp", "auth")',
        '("mcp", "logout")',
        '("mcp", "debug")',
        '("plugin", "install")',
        "subprocess.list2cmdline(cli_args)",
        '"PACKAGE_TERMINAL_OPENED"',
    ):
        assert marker in process
    assert "shell=True" not in process

def test_session_manager_focuses_existing_window_instead_of_spawning_duplicate():
    process = _source("src/opencode_runtime/process_manager.py")
    assert "_focus_existing_session_manager" in process
    assert "EnumWindows" in process
    assert "SetForegroundWindow" in process
    assert '"SESSION_MANAGER_TERMINAL_FOCUSED"' in process
    assert '"focused": True' in process

    service = _source("src/opencode_runtime/service.py")
    assert 'if result.get("focused") and control is not None:' not in service
    assert "_write_session_manager_handoff" in process

def test_prepare_session_owner_verification_retries_short_creation_race():
    source = _source("routes/opencode_runtime_routes.py")
    assert "import time" in source
    assert "for attempt in range(6):" in source
    assert "time.sleep(0.05)" in source
    assert "if attempt >= 5:" in source

def test_session_manager_receives_current_session_log_path():
    process = _source("src/opencode_runtime/process_manager.py")
    assert '"--log-path"' in process
    assert "self.session_log_path(log_session_id)" in process

def test_windows_terminal_launch_requests_usable_tui_dimensions():
    process = _source("src/opencode_runtime/process_manager.py")
    assert '"--size"' in process
    assert '"121,37"' in process
    assert process.index('"--size"') < process.index('"new-tab"')

def test_session_manager_uses_one_time_secret_launch_ticket():
    process = _source("src/opencode_runtime/process_manager.py")
    tui = _source("src/opencode_runtime/session_manager_tui.py")

    assert "_write_session_manager_launch_ticket" in process
    assert "os.O_EXCL" in process
    assert '"--launch-ticket"' in process
    assert '"password":' in process
    assert "load_launch_ticket" in tui
    assert "path.unlink(missing_ok=True)" in tui
    assert "--control-token" not in process


def test_session_manager_log_uses_odysseus_conversation_id():
    process = _source("src/opencode_runtime/process_manager.py")
    service = _source("src/opencode_runtime/service.py")

    assert "odysseus_session_id: str = \"\"" in process
    assert "self.session_log_path(log_session_id)" in process
    assert "log_path.touch(exist_ok=True)" in process
    assert '"odysseus_session_id": conversation_id' in service

def test_session_manager_launches_focus_mode_window():
    process = _source("src/opencode_runtime/process_manager.py")
    for marker in (
        '"--window"',
        '"new"',
        '"--focus"',
        '"--size"',
        '"121,37"',
        '"--suppressApplicationTitle"',
        '"--handoff-path"',
    ):
        assert marker in process


def test_existing_manager_uses_secure_handoff():
    process = _source("src/opencode_runtime/process_manager.py")
    service = _source("src/opencode_runtime/service.py")
    assert "_write_session_manager_handoff" in process
    assert "session-manager-handoff.json" in process
    assert "os.replace(" in process
    assert 'if result.get("focused") and control is not None:' not in service

def test_session_manager_control_is_cross_instance(
    tmp_path,
):
    import threading
    from types import SimpleNamespace

    import pytest

    from src.opencode_runtime.service import (
        OpenCodeRuntimeService,
    )

    def service_instance():
        service = object.__new__(
            OpenCodeRuntimeService
        )
        service.server_manager = (
            SimpleNamespace(
                log_root=tmp_path,
                owns_process=True,
            )
        )
        service._session_manager_controls = {}
        service._sessions = {}
        service._lock = threading.RLock()
        service._log = (
            lambda *args, **kwargs: None
        )
        return service

    issuer = service_instance()
    resolver = service_instance()
    grant = (
        issuer.issue_session_manager_control(
            "chat-file-backed",
            ttl_seconds=60,
        )
    )
    token = grant["token"]

    assert (
        resolver.resolve_session_manager_control(
            token
        )
        == "chat-file-backed"
    )
    assert (
        resolver.release_session_manager_control(
            token
        )
        is True
    )
    with pytest.raises(
        PermissionError
    ):
        issuer.resolve_session_manager_control(
            token
        )
