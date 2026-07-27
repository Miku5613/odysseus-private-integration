#!/usr/bin/env python3
# Terminal-native control surface for an Odysseus-managed OpenCode server.

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TERMINAL_RESET = "\x1b[0m"
INVERSE = "\x1b[7m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

BG = "\x1b[48;2;30;30;46m"
PANEL_BG = "\x1b[48;2;36;36;52m"
TEXT = "\x1b[38;2;205;214;244m"
RESET = TERMINAL_RESET + PANEL_BG + TEXT
GREEN = "\x1b[38;2;166;227;161m"
RED = "\x1b[38;2;243;139;168m"
BORDER = "\x1b[38;2;108;112;134m"
FOCUS_BORDER = "\x1b[38;2;137;180;250m"
TITLE = "\x1b[38;2;166;227;161m"
CYAN = "\x1b[38;2;137;220;235m"
YELLOW = "\x1b[38;2;249;226;175m"
MUTED = "\x1b[38;2;166;173;200m"
DANGER = "\x1b[38;2;243;139;168m"
SELECTED = "\x1b[48;2;137;180;250m\x1b[38;2;30;30;46m"

HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR = "\x1b[2J\x1b[H"
HOME = "\x1b[H"
CLEAR_TO_END = "\x1b[J"
ALT_ENTER = "\x1b[?1049h"
ALT_EXIT = "\x1b[?1049l"
MOUSE_ENABLE = "\x1b[?1000h\x1b[?1006h"
MOUSE_DISABLE = "\x1b[?1000l\x1b[?1006l"
RESIZE_DEFAULT = "\x1b[8;42;128t"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
MOUSE_RE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

TABS = (
    "SESSION",
    "PERMISSIONS",
    "LIFECYCLE",
    "MCP",
    "PLUGINS",
    "EXTENSIONS",
    "COMMAND",
)



@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def contains(self, x: int, y: int) -> bool:
        return (
            self.x <= int(x) < self.x + self.width
            and self.y <= int(y) < self.y + self.height
        )

@dataclass
class ApiResult:
    ok: bool
    status: int
    value: Any = None
    error: str = ""


class OpenCodeClient:
    def __init__(
        self,
        base_url: str,
        workspace: str,
        username: str = "",
        password: str = "",
        timeout: float = 5.0,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.workspace = str(Path(workspace).expanduser().resolve())
        self.username = str(username or "")
        self.password = str(password or "")
        self.timeout = float(timeout)

    def _url(self, path: str) -> str:
        route = "/" + str(path or "").lstrip("/")
        url = self.base_url + route
        if route.startswith("/global/"):
            return url
        separator = "&" if "?" in url else "?"
        return url + separator + urllib.parse.urlencode({
            "directory": self.workspace,
        })

    def request(self, method: str, path: str, payload: Any = None) -> ApiResult:
        body = None
        headers = {
            "Accept": "application/json",
            "User-Agent": "Odysseus-OpenCode-Session-Manager/1",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        if self.password:
            identity = f"{self.username or 'opencode'}:{self.password}"
            token = base64.b64encode(
                identity.encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"

        request = urllib.request.Request(
            self._url(path),
            data=body,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                value = None
                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    try:
                        value = json.loads(text)
                    except json.JSONDecodeError:
                        value = text
                return ApiResult(
                    ok=200 <= response.status < 300,
                    status=response.status,
                    value=value,
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ApiResult(False, exc.code, error=detail or str(exc))
        except Exception as exc:
            return ApiResult(False, 0, error=str(exc))

    def get(self, path: str) -> ApiResult:
        return self.request("GET", path)

    def post(self, path: str, payload: Any = None) -> ApiResult:
        return self.request("POST", path, payload)

    def delete(self, path: str) -> ApiResult:
        return self.request("DELETE", path)

class OdysseusControlClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: float = 8.0,
    ):
        self.base_url = str(base_url or "").rstrip("/")
        self.token = str(token or "").strip()
        self.timeout = float(timeout)

    def _url(self, path: str) -> str:
        return self.base_url + "/" + str(path or "").lstrip("/")

    def request(
        self,
        method: str,
        path: str,
        payload: Any = None,
    ) -> ApiResult:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "Odysseus-Session-Manager-Control/1",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self._url(path),
            data=body,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
                value = json.loads(raw) if raw.strip() else None
                if isinstance(value, dict):
                    rotated = str(value.get("control_token") or "").strip()
                    if rotated:
                        self.token = rotated
                return ApiResult(True, int(response.status), value)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            detail = raw
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    detail = str(parsed.get("detail") or parsed)
            except Exception:
                pass
            return ApiResult(False, int(exc.code), error=detail)
        except Exception as exc:
            return ApiResult(False, 0, error=str(exc))

    def get(self, path: str) -> ApiResult:
        return self.request("GET", path)

    def post(self, path: str, payload: Any = None) -> ApiResult:
        return self.request("POST", path, payload)

    def put(self, path: str, payload: Any = None) -> ApiResult:
        return self.request("PUT", path, payload)

    def delete(self, path: str) -> ApiResult:
        return self.request("DELETE", path)



@dataclass
class Snapshot:
    server_live: bool = False
    version: str = ""
    session: dict[str, Any] = field(default_factory=dict)
    session_status: Any = None
    commands: list[dict[str, Any]] = field(default_factory=list)
    mcp: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    control: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def load_launch_ticket(path_value: str) -> dict[str, str]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {}

    path = Path(path_text).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        return {
            str(key): str(item or "")
            for key, item in value.items()
        }
    except Exception:
        return {}
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

def load_session_manager_handoff(path_value: str) -> dict[str, str]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        return {
            str(key): str(item or "")
            for key, item in value.items()
        }
    except Exception:
        return {}
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

def parse_slash_command(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text.startswith("/"):
        raise ValueError("Commands must begin with /")
    body = text[1:].strip()
    if not body:
        raise ValueError("Enter an OpenCode command")
    command, _, arguments = body.partition(" ")
    return command.strip().lower(), arguments.strip()


def plain(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def clip(text: str, width: int) -> str:
    value = str(text)
    limit = max(0, int(width))
    if limit <= 0:
        return ""

    output: list[str] = []
    visible = 0
    index = 0

    while index < len(value) and visible < limit:
        if value[index] == "\x1b":
            match = ANSI_RE.match(value, index)
            if match is not None:
                output.append(match.group(0))
                index = match.end()
                continue

        output.append(value[index])
        visible += 1
        index += 1

    return "".join(output) + RESET


def pad(text: str, width: int) -> str:
    visible = ANSI_RE.sub("", str(text))
    spaces = " " * max(0, width - len(visible))
    return str(text) + PANEL_BG + TEXT + spaces


def mapping_rows(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [plain(value)]
    rows = [
        f"{str(key):<22} {plain(value[key])}"
        for key in sorted(value, key=lambda item: str(item).casefold())
    ]
    return rows or ["(none)"]


class SessionManagerTUI:
    def __init__(
        self,
        client: OpenCodeClient,
        session_id: str,
        model: str,
        control_client: OdysseusControlClient | None = None,
        log_path: str = "",
        handoff_path: str = "",
    ):
        self.client = client
        self.control_client = control_client
        self.session_id = str(session_id or "").strip()
        self.model = str(model or "").strip()
        self.log_path = str(log_path or "").strip()
        self.handoff_path = str(handoff_path or "").strip()
        self.tab_index = 0
        self.message = "ready"
        self.snapshot = Snapshot()
        self.command_history: list[str] = []
        self.history_index: int | None = None
        self.command_buffer = ""
        self.command_mode = False
        self.focus_order = (
            "options",
            "active",
            "command",
            "logs",
            "actions",
        )
        self.focus_index = 0
        self.scroll_offsets = {"active": 0, "logs": 0}
        self.action_index = 0
        self.confirm_action = ""
        self.confirm_deadline = 0.0
        self.exit_requested = False
        self.zoom_panel = ""
        self.log_lines: list[str] = []
        self.log_follow = True
        self._hitboxes: dict[str, Rect] = {}
        self._option_hitboxes: list[Rect] = []
        self._action_hitboxes: list[Rect] = []
        self._terminal_size_override: tuple[int, int] | None = None

    @property
    def tab(self) -> str:
        return TABS[self.tab_index]

    @property
    def focus_name(self) -> str:
        return self.focus_order[self.focus_index]

    def _set_focus(self, name: str) -> None:
        if name in self.focus_order:
            self.focus_index = self.focus_order.index(name)

    def _move_focus(self, direction: int) -> None:
        self.focus_index = (self.focus_index + int(direction)) % len(self.focus_order)

    def _consume_handoff(self) -> bool:
        value = load_session_manager_handoff(self.handoff_path)
        if not value:
            return False

        ticket = load_launch_ticket(value.get("launch_ticket", ""))
        old_control = self.control_client
        if old_control is not None:
            try:
                old_control.post("/release", {})
            except Exception:
                pass

        self.client = OpenCodeClient(
            str(value.get("server") or self.client.base_url).strip(),
            str(value.get("workspace") or self.client.workspace).strip(),
            username=str(
                ticket.get("username")
                or self.client.username
                or ""
            ),
            password=str(
                ticket.get("password")
                or self.client.password
                or ""
            ),
        )
        control_url = str(value.get("control_url") or "").strip()
        control_token = str(ticket.get("control_token") or "").strip()
        self.control_client = (
            OdysseusControlClient(control_url, control_token)
            if control_url and control_token
            else None
        )
        self.session_id = str(
            value.get("session_id") or self.session_id
        ).strip()
        self.model = str(value.get("model") or self.model).strip()
        self.log_path = str(
            value.get("log_path") or self.log_path
        ).strip()
        self.snapshot = Snapshot()
        self.scroll_offsets["active"] = 0
        self.scroll_offsets["logs"] = 0
        self.log_follow = True
        self.message = "session credentials refreshed"
        self.refresh()
        self._refresh_logs()
        return True

    @staticmethod
    def _windows_control_pressed() -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes
            return bool(
                ctypes.windll.user32.GetAsyncKeyState(
                    0x11
                )
                & 0x8000
            )
        except Exception:
            return False

    def _start_window_drag_watcher(self) -> bool:
        if os.name != "nt":
            return False
        if getattr(self, "_window_drag_thread", None) is not None:
            return True
        try:
            import ctypes
            import threading
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            user32.GetAncestor.restype = wintypes.HWND
            user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
            user32.GetCursorPos.restype = wintypes.BOOL
            user32.GetWindowRect.argtypes = [
                wintypes.HWND,
                ctypes.POINTER(wintypes.RECT),
            ]
            user32.GetWindowRect.restype = wintypes.BOOL
            user32.GetClassNameW.argtypes = [
                wintypes.HWND,
                wintypes.LPWSTR,
                ctypes.c_int,
            ]
            user32.GetClassNameW.restype = ctypes.c_int
            user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
            user32.GetAsyncKeyState.restype = ctypes.c_short
            user32.SetWindowPos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            user32.SetWindowPos.restype = wintypes.BOOL

            stop_event = threading.Event()
            self._window_drag_stop = stop_event

            def watcher() -> None:
                vk_lbutton = 0x01
                vk_shift = 0x10
                swp_flags = 0x0001 | 0x0004 | 0x0010 | 0x4000
                was_down = False

                while not stop_event.wait(0.008):
                    is_down = bool(
                        user32.GetAsyncKeyState(vk_lbutton) & 0x8000
                    )
                    shift_down = bool(
                        user32.GetAsyncKeyState(vk_shift) & 0x8000
                    )

                    if is_down and not was_down and not shift_down:
                        foreground = user32.GetForegroundWindow()
                        hwnd = (
                            user32.GetAncestor(foreground, 2)
                            if foreground
                            else None
                        ) or foreground
                        if hwnd:
                            class_name = ctypes.create_unicode_buffer(256)
                            user32.GetClassNameW(
                                hwnd,
                                class_name,
                                len(class_name),
                            )
                            cursor = wintypes.POINT()
                            rect = wintypes.RECT()
                            if (
                                "CASCADIA" in class_name.value.upper()
                                and user32.GetCursorPos(ctypes.byref(cursor))
                                and user32.GetWindowRect(hwnd, ctypes.byref(rect))
                                and rect.left <= cursor.x <= rect.right
                                and rect.top <= cursor.y <= rect.top + 100
                            ):
                                start_x = cursor.x
                                start_y = cursor.y
                                start_left = rect.left
                                start_top = rect.top

                                while (
                                    not stop_event.is_set()
                                    and user32.GetAsyncKeyState(vk_lbutton) & 0x8000
                                    and not user32.GetAsyncKeyState(vk_shift) & 0x8000
                                ):
                                    current = wintypes.POINT()
                                    if not user32.GetCursorPos(ctypes.byref(current)):
                                        break
                                    user32.SetWindowPos(
                                        hwnd,
                                        None,
                                        start_left + current.x - start_x,
                                        start_top + current.y - start_y,
                                        0,
                                        0,
                                        swp_flags,
                                    )
                                    time.sleep(0.008)

                    was_down = is_down

            thread = threading.Thread(
                target=watcher,
                name="OpenCodeSessionManagerHeaderDragWatcher",
                daemon=True,
            )
            self._window_drag_thread = thread
            thread.start()
            return True
        except Exception:
            self._window_drag_stop = None
            self._window_drag_thread = None
            return False

    def _stop_window_drag_watcher(self) -> None:
        stop_event = getattr(self, "_window_drag_stop", None)
        if stop_event is not None:
            stop_event.set()
        self._window_drag_stop = None
        self._window_drag_thread = None

    def _terminal_size(self) -> tuple[int, int]:
        if self._terminal_size_override:
            columns, rows = self._terminal_size_override
        else:
            value = shutil.get_terminal_size((120, 36))
            columns, rows = value.columns, value.lines

        # Reserve one terminal column and row. Filling the final cell can
        # trigger auto-wrap and scroll the upper panels out of view.
        return max(72, int(columns) - 1), max(18, int(rows) - 1)

    def refresh(self) -> None:
        snapshot = Snapshot()
        health = self.client.get("/global/health")
        snapshot.server_live = health.ok
        if health.ok and isinstance(health.value, dict):
            snapshot.version = str(health.value.get("version") or "")
        elif not health.ok:
            snapshot.error = health.error or "OpenCode server unavailable"

        if self.session_id:
            encoded = urllib.parse.quote(self.session_id, safe="")
            session = self.client.get(f"/session/{encoded}")
            if session.ok and isinstance(session.value, dict):
                snapshot.session = session.value

        statuses = self.client.get("/session/status")
        if statuses.ok:
            snapshot.session_status = statuses.value

        commands = self.client.get("/command")
        if commands.ok and isinstance(commands.value, list):
            snapshot.commands = [
                item for item in commands.value if isinstance(item, dict)
            ]

        mcp = self.client.get("/mcp")
        if mcp.ok:
            snapshot.mcp = mcp.value

        config = self.client.get("/config")
        if config.ok and isinstance(config.value, dict):
            snapshot.config = config.value

        if self.control_client is not None:
            control = self.control_client.get("/state")
            if control.ok and isinstance(control.value, dict):
                snapshot.control = control.value
                native_session_id = str(
                    control.value.get("opencode_session_id") or ""
                ).strip()
                if native_session_id:
                    self.session_id = native_session_id
                model_id = str(
                    control.value.get("model_id") or ""
                ).strip()
                if model_id:
                    self.model = model_id
            elif not snapshot.error:
                snapshot.error = control.error or "control bridge unavailable"

        self.snapshot = snapshot

    def _refresh_logs(self) -> bool:
        previous = self.log_lines
        if not self.log_path:
            current = ["No session log path was provided."]
        else:
            path = Path(self.log_path).expanduser()
            if not path.is_file():
                current = [f"Waiting for log file: {path}"]
            else:
                try:
                    current = path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-500:] or ["Log file is empty."]
                except Exception as exc:
                    current = [f"Log read failed: {exc}"]
        self.log_lines = current
        if self.log_follow:
            self.scroll_offsets["logs"] = max(0, len(current) - 1)
        return current != previous

    def _viewport(self, lines: list[str], height: int, key: str):
        body_height = max(1, int(height))
        maximum = max(0, len(lines) - body_height)
        offset = max(0, min(maximum, int(self.scroll_offsets.get(key, 0))))
        self.scroll_offsets[key] = offset
        return lines[offset:offset + body_height], offset, maximum

    def _scroll(self, key: str, amount: int) -> None:
        self.scroll_offsets[key] = max(
            0, int(self.scroll_offsets.get(key, 0)) + int(amount)
        )
        if key == "logs":
            self.log_follow = False

    def _panel(
        self,
        title: str,
        lines: list[str],
        width: int,
        height: int,
        *,
        focused: bool = False,
        scroll_key: str = "",
    ) -> list[str]:
        width = max(8, int(width))
        height = max(3, int(height))
        inner = width - 2
        body_height = height - 2
        border = FOCUS_BORDER if focused else BORDER
        label = f" {title} "
        top = (
            PANEL_BG + border + "┌─" + TITLE + BOLD + label
            + RESET + PANEL_BG + border
            + ("─" * max(0, inner - len(label) - 1))
            + "┐" + RESET
        )

        visible = lines[:body_height]
        offset = maximum = 0
        if scroll_key:
            visible, offset, maximum = self._viewport(
                lines, body_height, scroll_key
            )

        thumb = -1
        if maximum > 0:
            thumb = int(
                round((offset / maximum) * max(0, body_height - 1))
            )

        output = [top]
        for index in range(body_height):
            value = visible[index] if index < len(visible) else ""
            value = clip(str(value), inner)
            right = "█" if index == thumb else "│"
            output.append(
                PANEL_BG + border + "│" + RESET
                + PANEL_BG + TEXT + pad(value, inner)
                + border + right + RESET
            )

        output.append(
            PANEL_BG + border + "└" + ("─" * inner) + "┘" + RESET
        )
        return output

    @staticmethod
    def _join_panels(left: list[str], right: list[str]) -> list[str]:
        count = max(len(left), len(right))
        return [
            (left[index] if index < len(left) else "")
            + (right[index] if index < len(right) else "")
            for index in range(count)
        ]

    def _active_lines(self, width: int) -> list[str]:
        if self.tab == "SESSION":
            status = "-"
            if isinstance(self.snapshot.session_status, dict):
                status = self.snapshot.session_status.get(self.session_id, "-")
            return [
                f"{TITLE}{BOLD}SESSION{RESET}", "",
                f"Native session   {self.session_id or '(not bound)'}",
                f"Status           {plain(status)}",
                f"Title            {plain(self.snapshot.session.get('title'))}",
                f"Model            {self.model or '-'}",
                f"Workspace        {self.client.workspace}",
            ]
        if self.tab == "PERMISSIONS":
            return [
                f"{TITLE}{BOLD}PERMISSIONS{RESET}", "",
                "OpenCode effective permission configuration", "",
                *mapping_rows(self.snapshot.config.get("permission")),
            ]
        if self.tab == "LIFECYCLE":
            return [
                f"{TITLE}{BOLD}LIFECYCLE{RESET}", "",
                f"Server           {'LIVE' if self.snapshot.server_live else 'OFFLINE'}",
                f"Version          {self.snapshot.version or '-'}",
                f"Model            {self.model or '-'}",
                f"Workspace        {self.client.workspace}",
                f"Session          {self.session_id or '-'}",
            ]
        if self.tab == "MCP":
            return [f"{TITLE}{BOLD}MCP SERVERS{RESET}", "", *mapping_rows(self.snapshot.mcp)]
        if self.tab == "PLUGINS":
            plugins = self.snapshot.config.get("plugin", self.snapshot.config.get("plugins"))
            rows = [plain(v) for v in plugins] if isinstance(plugins, list) else mapping_rows(plugins)
            return [f"{TITLE}{BOLD}PLUGINS{RESET}", "", *(rows or ["(none)"])]
        if self.tab == "EXTENSIONS":
            extensions = self.snapshot.config.get("extension", self.snapshot.config.get("extensions"))
            rows = [plain(v) for v in extensions] if isinstance(extensions, list) else mapping_rows(extensions)
            return [
                f"{TITLE}{BOLD}EXTENSIONS{RESET}", "", *(rows or ["(none)"]), "",
                f"{MUTED}Config-backed and read-only here.{RESET}",
            ]
        rows = []
        for command in self.snapshot.commands:
            if not isinstance(command, dict):
                continue
            name = command.get("name") or command.get("command") or "?"
            description = command.get("description") or ""
            rows.append(f"/{name:<18} {clip(description, max(12, width - 22))}")
        return [f"{TITLE}{BOLD}OPENCODE COMMANDS{RESET}", "", *(rows or ["No commands returned by OpenCode."])]

    def _available_commands(self) -> list[str]:
        groups = {
            "SESSION": (("/new [title]", "Create a new native session"), ("/abort", "Abort the active OpenCode run"), ("/delete", "Delete the current native session"), ("/refresh", "Refresh Session Manager state")),
            "PERMISSIONS": (("/permission ask", "Require approval"), ("/permission workspace", "Allow workspace-safe tools"), ("/approve REQUEST", "Approve once or for session"), ("/reject REQUEST", "Reject a request"), ("/clear-approvals", "Clear session approvals")),
            "LIFECYCLE": (("/restart", "Restart the managed server"), ("/stop", "Close the managed server"), ("/new [title]", "Create native session"), ("/delete", "Delete native session")),
            "MCP": (("/mcp-list", "List MCP servers"), ("/mcp-add NAME", "Add an MCP server"), ("/mcp-auth NAME", "Authenticate an MCP server"), ("/mcp-logout NAME", "Remove MCP authentication"), ("/mcp-debug NAME", "Open MCP diagnostics")),
            "PLUGINS": (("/plugin-install MODULE", "Install an OpenCode plugin"), ("/restart", "Apply plugin changes")),
            "EXTENSIONS": (("/plugin-install MODULE", "Install supported package"), ("/restart", "Apply configuration changes")),
        }
        values = groups.get(self.tab)
        if self.tab == "COMMAND":
            values = tuple(
                (f"/{item.get('name') or item.get('command') or '?'}", str(item.get("description") or ""))
                for item in self.snapshot.commands[:20] if isinstance(item, dict)
            )
        rows = [f"{CYAN}{command:<24}{RESET} {description}" for command, description in (values or ())]
        return rows or ["No contextual commands available."]

    def content_lines(self, width: int) -> list[str]:
        return [
            *self._active_lines(width), "",
            f"{TITLE}{BOLD}AVAILABLE COMMANDS{RESET}", "",
            *self._available_commands(),
        ]

    def _options_lines(self) -> list[str]:
        return [
            f"{SELECTED} {label:<14} {RESET}" if index == self.tab_index else f" {label}"
            for index, label in enumerate(TABS)
        ]

    def _actions_lines(self) -> list[str]:
        values = (("Restart Server", YELLOW), ("Close Server", DANGER))
        rows = []
        for index, (label, color) in enumerate(values):
            selected = self.focus_name == "actions" and index == self.action_index
            rows.append(
                f"{SELECTED} [ {label:<14} ] {RESET}"
                if selected else f"{color} [ {label:<14} ] {RESET}"
            )
        return [*rows, "", f"{MUTED}Enter or click to run{RESET}"]

    def _command_panel_lines(self, width: int) -> list[str]:
        prompt = f"> /{self.command_buffer}" if self.command_mode else "> Press / to enter a command"
        rows = [f"{CYAN}{prompt}{RESET}", ""]
        if self.command_buffer:
            prefix = self.command_buffer.casefold()
            suggestions = [value for value in self._command_names() if value.casefold().startswith(prefix)][:4]
            if suggestions:
                rows.extend([f"{MUTED}Suggestions{RESET}", *(f"  /{value}" for value in suggestions)])
        rows.extend(["", f"{MUTED}Status{RESET}  {self.message}", f"{MUTED}Tab complete  ↑↓ history  Enter run  Esc clear{RESET}"])
        return rows

    def _logs_panel_lines(self) -> list[str]:
        return [clip(line, 500) for line in self.log_lines] or ["No log entries yet."]

    def render(self) -> str:
        width, height = self._terminal_size()
        header_height, footer_height = 4, 4
        live = (
            f"{GREEN}{BOLD}LIVE{RESET}"
            if self.snapshot.server_live
            else f"{RED}{BOLD}OFFLINE{RESET}"
        )
        header = self._panel(
            "OpenCode Session Manager",
            [
                f"{MUTED}Server{RESET} {live}    {MUTED}Model{RESET} {CYAN}{clip(self.model or '-', max(8, width // 3))}{RESET}",
                f"{MUTED}Workspace{RESET} {clip(self.client.workspace, max(12, width // 2))}    {MUTED}Session{RESET} {clip(self.session_id or '-', max(8, width // 4))}",
            ],
            width,
            header_height,
        )

        if self.zoom_panel:
            title = {
                "options": "OPTIONS",
                "active": "ACTIVE PANEL",
                "command": "COMMAND",
                "logs": "TERMINAL LOGS",
                "actions": "SERVER ACTIONS",
            }.get(self.zoom_panel, self.zoom_panel.upper())
            values = {
                "options": self._options_lines(),
                "active": self.content_lines(width - 2),
                "command": self._command_panel_lines(width - 2),
                "logs": self._logs_panel_lines(),
                "actions": self._actions_lines(),
            }.get(self.zoom_panel, [])
            scroll_key = (
                self.zoom_panel
                if self.zoom_panel in {"active", "logs"}
                else ""
            )
            body_height = max(3, height - header_height - footer_height)
            body = self._panel(
                title,
                values,
                width,
                body_height,
                focused=True,
                scroll_key=scroll_key,
            )
            self._hitboxes = {
                self.zoom_panel: Rect(
                    1,
                    header_height + 1,
                    width,
                    body_height,
                )
            }
            footer = self._panel(
                "SHORTCUTS",
                [
                    "Z / Esc restore dashboard   ↑↓ scroll   PgUp/PgDn page   Shift+drag select",
                    "Q close TUI",
                ],
                width,
                footer_height,
            )
            return (
                BG
                + "\n".join((header + body + footer)[:height])
                + TERMINAL_RESET
            )

        logs_height = max(4, min(8, height // 5))
        dashboard_height = height - header_height - logs_height - footer_height
        if dashboard_height < 6:
            logs_height = max(3, logs_height - (6 - dashboard_height))
            dashboard_height = height - header_height - logs_height - footer_height
        lower_height = max(3, min(6, dashboard_height // 3))
        upper_height = dashboard_height - lower_height
        if upper_height < 3:
            upper_height = 3
            lower_height = dashboard_height - upper_height
        left_width = max(24, min(34, width // 3))
        right_width = width - left_width

        options = self._panel(
            "OPTIONS", self._options_lines(), left_width, upper_height,
            focused=self.focus_name == "options",
        )
        active = self._panel(
            "ACTIVE PANEL", self.content_lines(right_width - 2),
            right_width, upper_height,
            focused=self.focus_name == "active",
            scroll_key="active",
        )
        actions = self._panel(
            "SERVER ACTIONS", self._actions_lines(),
            left_width, lower_height,
            focused=self.focus_name == "actions",
        )
        command = self._panel(
            "COMMAND", self._command_panel_lines(right_width - 2),
            right_width, lower_height,
            focused=self.focus_name == "command",
        )
        logs = self._panel(
            "TERMINAL LOGS", self._logs_panel_lines(),
            width, logs_height,
            focused=self.focus_name == "logs",
            scroll_key="logs",
        )
        footer = self._panel(
            "SHORTCUTS",
            [
                "1 Options  2 Active  3 Command  4 Logs  5 Actions   Tab next   Ctrl+Arrows focus   Z maximize",
                "/ command   ↑↓ navigate/scroll   PgUp/PgDn page   Enter run   Ctrl+R restart   Ctrl+X stop   Q close",
            ],
            width,
            footer_height,
        )

        self._hitboxes = {}
        self._option_hitboxes = []
        self._action_hitboxes = []
        upper_y = header_height + 1
        lower_y = upper_y + upper_height
        logs_y = lower_y + lower_height
        self._hitboxes["options"] = Rect(1, upper_y, left_width, upper_height)
        self._hitboxes["active"] = Rect(left_width + 1, upper_y, right_width, upper_height)
        self._hitboxes["actions"] = Rect(1, lower_y, left_width, lower_height)
        self._hitboxes["command"] = Rect(left_width + 1, lower_y, right_width, lower_height)
        self._hitboxes["logs"] = Rect(1, logs_y, width, logs_height)
        for index in range(len(TABS)):
            self._option_hitboxes.append(
                Rect(2, upper_y + 1 + index, max(1, left_width - 2), 1)
            )
        for index in range(2):
            self._action_hitboxes.append(
                Rect(2, lower_y + 1 + index, max(1, left_width - 2), 1)
            )
        output = []
        output.extend(header)
        output.extend(self._join_panels(options, active))
        output.extend(self._join_panels(actions, command))
        output.extend(logs)
        output.extend(footer)
        return BG + "\n".join(output[:height]) + TERMINAL_RESET

    def execute(self, value: str) -> bool:
        try:
            command, arguments = parse_slash_command(value)
        except ValueError as exc:
            self.message = str(exc)
            return True

        if command in {"quit", "exit", "q"}:
            return False
        if command in {"refresh", "r"}:
            self.refresh()
            self.message = "refreshed"
            return True
        if command in {"help", "commands"}:
            self.tab_index = TABS.index("COMMAND")
            self.message = "command list"
            return True
        if command in {"sessions", "session"}:
            self.tab_index = TABS.index("SESSION")
            self.message = "session panel"
            return True
        if command in {
            "mcp",
            "plugins",
            "extensions",
            "permissions",
            "lifecycle",
        }:
            self.tab_index = TABS.index(command.upper())
            self.message = f"{command} panel"
            return True

        control = self.control_client
        package_actions = {
            "mcp-add": ("mcp", "add", arguments),
            "mcp-list": ("mcp", "list", ""),
            "mcp-auth": ("mcp", "auth", arguments),
            "mcp-logout": ("mcp", "logout", arguments),
            "mcp-debug": ("mcp", "debug", arguments),
            "plugin-install": ("plugin", "install", arguments),
        }
        if command in package_actions and control is not None:
            package, action, argument = package_actions[command]
            result = control.post(
                "/packages/open",
                {
                    "package": package,
                    "action": action,
                    "argument": argument,
                },
            )
            self.message = (
                f"{package} {action} terminal opened"
                if result.ok
                else (result.error or f"{package} {action} failed")
            )
            return True

        if command == "new" and control is not None:
            result = control.post(
                "/new",
                {"title": arguments or "OpenCode Session"},
            )
            if result.ok and isinstance(result.value, dict):
                self.session_id = str(
                    result.value.get("opencode_session_id") or ""
                )
                self.message = f"new session {self.session_id or 'created'}"
                self.refresh()
            else:
                self.message = result.error or "new session failed"
            return True

        if command == "delete" and control is not None:
            result = control.delete("/conversation")
            if result.ok:
                self.session_id = ""
                self.message = "native session mapping deleted"
                self.refresh()
            else:
                self.message = result.error or "delete failed"
            return True

        if command == "restart" and control is not None:
            result = control.post("/restart", {})
            if result.ok and isinstance(result.value, dict):
                self.session_id = str(
                    result.value.get("opencode_session_id") or ""
                )
                self.message = "OpenCode server restarted"
                self.refresh()
            else:
                self.message = result.error or "restart failed"
            return True

        if command == "stop" and control is not None:
            result = control.post("/stop", {})
            if result.ok:
                self.session_id = ""
                self.message = "OpenCode server stopped"
                self.exit_requested = True
                return False
            self.message = result.error or "stop failed"
            return True

        if command == "permission" and control is not None:
            mode = arguments.strip().lower()
            result = control.put(
                "/permissions/mode",
                {"mode": mode},
            )
            self.message = (
                f"permission mode {mode}"
                if result.ok
                else (result.error or "permission mode failed")
            )
            self.refresh()
            return True

        if command in {"approve", "reject"} and control is not None:
            parts = arguments.split()
            request_id = parts[0] if parts else ""
            decision = (
                "reject"
                if command == "reject"
                else (parts[1].lower() if len(parts) > 1 else "once")
            )
            if not request_id:
                self.message = f"/{command} requires a request id"
                return True
            result = control.post(
                "/permissions/reply",
                {
                    "odysseus_session_id": str(
                        self.snapshot.control.get(
                            "odysseus_session_id",
                            "",
                        )
                    ),
                    "request_id": request_id,
                    "decision": decision,
                },
            )
            self.message = (
                f"permission {decision}"
                if result.ok
                else (result.error or "permission reply failed")
            )
            self.refresh()
            return True

        if command == "clear-approvals" and control is not None:
            result = control.delete("/permissions/approvals")
            self.message = (
                "session approvals cleared"
                if result.ok
                else (result.error or "clear approvals failed")
            )
            self.refresh()
            return True

        if command == "new":
            result = self.client.post(
                "/session",
                {"title": arguments or "OpenCode Session"},
            )
            if result.ok and isinstance(result.value, dict):
                self.session_id = str(
                    result.value.get("id")
                    or result.value.get("sessionID")
                    or ""
                )
                self.message = f"new session {self.session_id or 'created'}"
                self.refresh()
            else:
                self.message = result.error or "new session failed"
            return True

        if not self.session_id:
            self.message = "no native session is selected"
            return True

        encoded = urllib.parse.quote(self.session_id, safe="")
        if command == "abort":
            result = self.client.post(f"/session/{encoded}/abort", {})
            self.message = "abort requested" if result.ok else result.error
            return True
        if command == "delete":
            result = self.client.delete(f"/session/{encoded}")
            if result.ok:
                self.session_id = ""
                self.message = "native session deleted"
                self.refresh()
            else:
                self.message = result.error
            return True

        result = self.client.post(
            f"/session/{encoded}/command",
            {"command": command, "arguments": arguments},
        )
        self.message = (
            f"/{command} submitted"
            if result.ok
            else (result.error or f"/{command} failed")
        )
        self.refresh()
        return True

    def _command_names(self) -> list[str]:
        names = {
            "abort",
            "commands",
            "delete",
            "exit",
            "extensions",
            "help",
            "lifecycle",
            "mcp",
            "mcp-add",
            "mcp-auth",
            "mcp-debug",
            "mcp-list",
            "mcp-logout",
            "new",
            "plugin-install",
            "permissions",
            "plugins",
            "q",
            "quit",
            "r",
            "refresh",
            "session",
            "sessions",
        }
        for item in self.snapshot.commands:
            if not isinstance(item, dict):
                continue
            value = str(
                item.get("name")
                or item.get("command")
                or ""
            ).strip().lstrip("/")
            if value:
                names.add(value)
        return sorted(names, key=str.casefold)

    def _autocomplete_command(self, value: str) -> str:
        text = str(value or "")
        command, separator, arguments = text.partition(" ")
        if separator:
            return text

        prefix = command.strip().lstrip("/")
        if not prefix:
            return text

        matches = [
            name
            for name in self._command_names()
            if name.casefold().startswith(prefix.casefold())
        ]
        if not matches:
            return text
        if len(matches) == 1:
            return matches[0]

        common = os.path.commonprefix(
            [name.casefold() for name in matches]
        )
        if len(common) <= len(prefix):
            return text

        exemplar = matches[0]
        return exemplar[:len(common)]

    def _remember_command(self, value: str) -> None:
        command = str(value or "").strip()
        if not command:
            return
        if not command.startswith("/"):
            command = "/" + command
        if not self.command_history or self.command_history[-1] != command:
            self.command_history.append(command)
            del self.command_history[:-100]
        self.history_index = None

    def _history_command(self, direction: int) -> str:
        if not self.command_history:
            return ""

        if self.history_index is None:
            self.history_index = len(self.command_history)

        self.history_index = max(
            0,
            min(
                len(self.command_history),
                self.history_index + int(direction),
            ),
        )
        if self.history_index >= len(self.command_history):
            return ""
        return self.command_history[self.history_index].lstrip("/")

    def _invoke_action(self, action: str) -> bool:
        normalized = str(action or "").strip().lower()
        if normalized == "restart":
            self.confirm_action = ""
            return self.execute("/restart")

        if normalized != "stop":
            return True

        now = time.monotonic()
        if (
            self.confirm_action != "stop"
            or now > self.confirm_deadline
        ):
            self.confirm_action = "stop"
            self.confirm_deadline = now + 4.0
            self.message = (
                "Click Close Server again or press Enter to confirm."
            )
            return True

        self.confirm_action = ""
        return self.execute("/stop")

    def _handle_mouse(
        self,
        sequence: str,
    ) -> bool:
        match = MOUSE_RE.fullmatch(
            sequence
        )
        if (
            not match
            or match.group(4) != "M"
        ):
            return True

        button = int(match.group(1))
        x = int(match.group(2))
        y = int(match.group(3))

        # Shift adds a modifier bit, so only an
        # unmodified left press becomes a drag.
        # Shift+drag therefore remains native
        # Windows Terminal text selection.
        if button == 0 and y <= 4:
            # Independent watcher owns window movement.
            return True

        if button & 64:
            amount = (
                -3
                if button == 64
                else 3
            )
            for key in (
                "active",
                "logs",
            ):
                rect = self._hitboxes.get(
                    key
                )
                if (
                    rect
                    and rect.contains(x, y)
                ):
                    self._set_focus(key)
                    self._scroll(
                        key,
                        amount,
                    )
                    return True
            return True

        for index, rect in enumerate(
            self._option_hitboxes
        ):
            if rect.contains(x, y):
                self.tab_index = index
                self._set_focus(
                    "options"
                )
                self.scroll_offsets[
                    "active"
                ] = 0
                self.message = (
                    f"{TABS[index].lower()}"
                    " panel"
                )
                return True

        for index, rect in enumerate(
            self._action_hitboxes
        ):
            if rect.contains(x, y):
                self.action_index = index
                self._set_focus(
                    "actions"
                )
                return self._invoke_action(
                    "restart"
                    if index == 0
                    else "stop"
                )

        for name in (
            "active",
            "command",
            "logs",
            "options",
            "actions",
        ):
            rect = self._hitboxes.get(
                name
            )
            if (
                rect
                and rect.contains(x, y)
            ):
                self._set_focus(name)
                if name == "command":
                    self.command_mode = True
                return True
        return True

    def _read_windows_key(
        self,
        msvcrt,
    ) -> str:
        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            special = msvcrt.getwch()
            code = ord(special)

            dedicated_control = {
                115: "CTRL_LEFT",
                116: "CTRL_RIGHT",
                141: "CTRL_UP",
                145: "CTRL_DOWN",
            }
            if code in dedicated_control:
                return dedicated_control[
                    code
                ]

            value = {
                72: "UP",
                80: "DOWN",
                75: "LEFT",
                77: "RIGHT",
                73: "PAGEUP",
                81: "PAGEDOWN",
            }.get(code, "")
            if (
                value
                in {
                    "UP",
                    "DOWN",
                    "LEFT",
                    "RIGHT",
                }
                and self._windows_control_pressed()
            ):
                return "CTRL_" + value
            return value

        if key == "\x1b":
            sequence = key
            deadline = (
                time.monotonic() + 0.04
            )
            while (
                time.monotonic() < deadline
            ):
                if msvcrt.kbhit():
                    sequence += (
                        msvcrt.getwch()
                    )
                    deadline = (
                        time.monotonic()
                        + 0.01
                    )
                else:
                    time.sleep(0.001)
            return sequence
        return key

    def _handle_key(
        self,
        key: str,
    ) -> bool:
        if key.startswith("\x1b[<"):
            return self._handle_mouse(key)

        control_focus = {
            "CTRL_LEFT": -1,
            "CTRL_UP": -1,
            "CTRL_RIGHT": 1,
            "CTRL_DOWN": 1,
            "\x1b[1;5D": -1,
            "\x1b[1;5A": -1,
            "\x1b[5D": -1,
            "\x1b[5A": -1,
            "\x1b[1;5C": 1,
            "\x1b[1;5B": 1,
            "\x1b[5C": 1,
            "\x1b[5B": 1,
        }
        if (
            key in control_focus
            and not self.command_mode
        ):
            self._move_focus(
                control_focus[key]
            )
            self.message = (
                f"focus: {self.focus_name}"
            )
            return True

        if self.zoom_panel:
            if key in {
                "z",
                "Z",
                "\x1b",
                "ESC",
            }:
                self.zoom_panel = ""
                self.message = (
                    "dashboard restored"
                )
                return True
            if key in {"q", "Q"}:
                return False

        if self.command_mode:
            if key in {"\r", "\n"}:
                command = (
                    "/"
                    + self.command_buffer
                )
                self.command_buffer = ""
                self.command_mode = False
                self._remember_command(
                    command
                )
                return self.execute(
                    command
                )
            if key in {
                "\x1b",
                "ESC",
            }:
                self.command_buffer = ""
                self.command_mode = False
                self.message = (
                    "command cancelled"
                )
                return True
            if key in {
                "\b",
                "\x7f",
            }:
                self.command_buffer = (
                    self.command_buffer[:-1]
                )
                return True
            if key == "\t":
                self.command_buffer = (
                    self._autocomplete_command(
                        self.command_buffer
                    )
                )
                return True
            if key == "UP":
                self.command_buffer = (
                    self._history_command(
                        -1
                    )
                )
                return True
            if key == "DOWN":
                self.command_buffer = (
                    self._history_command(
                        1
                    )
                )
                return True
            if (
                len(key) == 1
                and key.isprintable()
            ):
                self.command_buffer += key
            return True

        if key in {
            "q",
            "Q",
            "\x1b",
        }:
            return False
        if key in {"z", "Z"}:
            self.zoom_panel = (
                self.focus_name
            )
            self.message = (
                "maximized: "
                + self.zoom_panel
            )
            return True
        if key == "/":
            self.command_mode = True
            self.command_buffer = ""
            self._set_focus(
                "command"
            )
            self.message = (
                "command mode"
            )
            return True
        if key == "\t":
            self._move_focus(1)
            return True
        if key == "\x1b[Z":
            self._move_focus(-1)
            return True
        if key in {
            "1",
            "2",
            "3",
            "4",
            "5",
        }:
            self.focus_index = (
                int(key) - 1
            )
            return True
        if key in {"r", "R"}:
            self.refresh()
            self.message = "refreshed"
            return True
        if key == "\x12":
            return self._invoke_action(
                "restart"
            )
        if key == "\x18":
            return self._invoke_action(
                "stop"
            )

        if key in {"LEFT", "UP"}:
            if (
                self.focus_name
                == "options"
            ):
                self.tab_index = (
                    self.tab_index - 1
                ) % len(TABS)
                self.scroll_offsets[
                    "active"
                ] = 0
            elif self.focus_name in {
                "active",
                "logs",
            }:
                self._scroll(
                    self.focus_name,
                    -1,
                )
            elif (
                self.focus_name
                == "actions"
            ):
                self.action_index = (
                    self.action_index - 1
                ) % 2
            return True

        if key in {
            "RIGHT",
            "DOWN",
        }:
            if (
                self.focus_name
                == "options"
            ):
                self.tab_index = (
                    self.tab_index + 1
                ) % len(TABS)
                self.scroll_offsets[
                    "active"
                ] = 0
            elif self.focus_name in {
                "active",
                "logs",
            }:
                self._scroll(
                    self.focus_name,
                    1,
                )
            elif (
                self.focus_name
                == "actions"
            ):
                self.action_index = (
                    self.action_index + 1
                ) % 2
            return True

        if (
            key == "PAGEUP"
            and self.focus_name
            in {"active", "logs"}
        ):
            self._scroll(
                self.focus_name,
                -5,
            )
            return True
        if (
            key == "PAGEDOWN"
            and self.focus_name
            in {"active", "logs"}
        ):
            self._scroll(
                self.focus_name,
                5,
            )
            return True

        if key in {"\r", "\n"}:
            if (
                self.focus_name
                == "command"
            ):
                self.command_mode = True
                return True
            if (
                self.focus_name
                == "actions"
            ):
                return self._invoke_action(
                    "restart"
                    if self.action_index
                    == 0
                    else "stop"
                )
        return True

    def _redraw_prompt(self, buffer: str) -> None:
        self.command_buffer = str(buffer or "")
        sys.stdout.write(HOME + self.render() + CLEAR_TO_END)
        sys.stdout.flush()

    def prompt(self) -> bool:
        self.command_mode = True
        self.command_buffer = ""
        self._set_focus("command")
        self.message = "command mode"
        return True

    def run(self) -> int:
        if os.name == "nt":
            os.system("")
        self.refresh()
        self._refresh_logs()
        sys.stdout.write(ALT_ENTER + CLEAR + HIDE_CURSOR + MOUSE_ENABLE)
        sys.stdout.flush()
        last_log_refresh = time.monotonic()
        last_handoff_check = 0.0
        self._start_window_drag_watcher()
        try:
            while True:
                sys.stdout.write(HOME + self.render() + CLEAR_TO_END)
                sys.stdout.flush()
                if os.name == "nt":
                    import msvcrt
                    while not msvcrt.kbhit():
                        now = time.monotonic()
                        if now - last_handoff_check >= 0.25:
                            if self._consume_handoff():
                                sys.stdout.write(HOME + self.render() + CLEAR_TO_END)
                                sys.stdout.flush()
                            last_handoff_check = now
                        if now - last_log_refresh >= 0.75:
                            if self._refresh_logs():
                                sys.stdout.write(HOME + self.render() + CLEAR_TO_END)
                                sys.stdout.flush()
                            last_log_refresh = now
                        time.sleep(0.02)
                    key = self._read_windows_key(msvcrt)
                else:
                    self._consume_handoff()
                    key = sys.stdin.read(1)
                    if not key:
                        break
                if not self._handle_key(key):
                    break
        finally:
            self._stop_window_drag_watcher()
            if self.control_client is not None:
                try:
                    self.control_client.post("/release", {})
                except Exception:
                    pass
            sys.stdout.write(
                MOUSE_DISABLE + SHOW_CURSOR + TERMINAL_RESET + ALT_EXIT
            )
            sys.stdout.flush()
        return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Odysseus OpenCode Session Manager",
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--session", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--control-url", default="")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--launch-ticket", default="")
    parser.add_argument("--handoff-path", default="")
    parser.add_argument("--render-once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    ticket = load_launch_ticket(args.launch_ticket)
    token = str(
        ticket.get("control_token")
        or os.environ.get("ODYSSEUS_SESSION_MANAGER_TOKEN")
        or ""
    ).strip()
    client = OpenCodeClient(
        args.server,
        args.workspace,
        username=(
            ticket.get("username")
            or os.environ.get("OPENCODE_SERVER_USERNAME", "")
        ),
        password=(
            ticket.get("password")
            or os.environ.get("OPENCODE_SERVER_PASSWORD", "")
        ),
    )
    control = (
        OdysseusControlClient(args.control_url, token)
        if args.control_url and token
        else None
    )
    app = SessionManagerTUI(
        client,
        session_id=args.session,
        model=args.model,
        control_client=control,
        log_path=args.log_path,
        handoff_path=args.handoff_path,
    )
    app.refresh()
    app._refresh_logs()
    if args.render_once:
        print(app.render())
        return 0
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
