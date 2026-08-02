from __future__ import annotations

import atexit
from collections.abc import Callable
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tempfile
from typing import Any
from urllib.parse import urlparse

from src.chat_helpers import is_vision_model

from .config import OpenCodeRuntimeConfig


class OpenCodeServerLifecycleError(RuntimeError):
    """Raised when the managed OpenCode server cannot be started safely."""


class OpenCodeServerManager:
    """Connect to an existing server or own one local OpenCode process.

    This manager never stops a server it did not start. Managed startup is
    restricted to loopback URLs so Odysseus cannot accidentally launch or
    control a remote host.
    """

    MANAGED_PERMISSION_ACTIONS = (
        "read",
        "glob",
        "grep",
        "edit",
        "bash",
        "external_directory",
        "list",
        "lsp",
        "task",
        "doom_loop",
    )

    def __init__(
        self,
        config: OpenCodeRuntimeConfig,
        *,
        health_probe: Callable[[], dict[str, Any]],
        process_factory: Callable[..., Any] = subprocess.Popen,
        which: Callable[[str], str | None] = shutil.which,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        log_root: Path | str | None = None,
    ):
        self.config = config
        self.health_probe = health_probe
        self.process_factory = process_factory
        self.which = which
        self.sleep = sleep
        self.monotonic = monotonic
        self._process: Any | None = None
        self._log_handle: Any | None = None
        self._project_root = Path(__file__).resolve().parents[2]
        if log_root is not None:
            resolved_log_root = Path(log_root).expanduser().resolve()
        elif (
            os.environ.get("PYTEST_CURRENT_TEST")
            or "pytest" in sys.modules
        ):
            resolved_log_root = (
                Path(tempfile.gettempdir())
                / "odysseus-opencode-pytest"
                / str(os.getpid())
            )
        else:
            resolved_log_root = (
                self._project_root / "logs" / "OpenCode"
            )
        self._log_root = resolved_log_root
        self._log_path = (
            self._log_root
            / "server"
            / f"{dt.datetime.now().strftime('%d-%m-%Y')}.log"
        )
        self._lock = threading.RLock()
        self._atexit_registered = False
        self._generation = 0

    @property
    def owns_process(self) -> bool:
        with self._lock:
            return self._process is not None

    @property
    def log_path(self) -> Path:
        """Compatibility alias for the current dated server log."""

        return self._log_path

    @property
    def log_root(self) -> Path:
        return self._log_root

    def session_log_path(self, odysseus_session_id: str) -> Path:
        value = str(odysseus_session_id or "").strip()
        if not value:
            return self._log_path
        safe_name = "".join(
            character
            if character.isalnum() or character in {"-", "_", "."}
            else "_"
            for character in value
        ).strip("._")
        if not safe_name:
            safe_name = hashlib.sha256(
                value.encode("utf-8", errors="replace")
            ).hexdigest()[:24]
        return self._log_root / "sessions" / f"{safe_name}.log"

    def config_default_model(self) -> str:
        config_path = str(self.config.config_path or "").strip()
        if not config_path:
            return ""
        path = Path(config_path).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        for key in (
            "model",
            "default_model",
            "defaultModel",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _prepare_managed_config(self) -> tuple[str, list[dict[str, str]]]:
        """Create an Odysseus-owned portable OpenCode config overlay."""

        configured = str(self.config.config_path or "").strip()
        has_registry_provider = bool(
            str(self.config.provider_base_url or "").strip()
            and str(self.config.provider_id or "").strip()
            and str(self.config.model_id or "").strip()
        )
        if not configured and not has_registry_provider:
            return "", []

        payload: dict[str, Any] = {}
        if configured:
            source_path = Path(configured).expanduser().resolve()
            if not source_path.is_file():
                raise OpenCodeServerLifecycleError(
                    f"OpenCode config does not exist: {source_path}"
                )
            try:
                payload = json.loads(
                    source_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise OpenCodeServerLifecycleError(
                    f"OpenCode config could not be read: {source_path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise OpenCodeServerLifecycleError(
                    "OpenCode config must contain a JSON object."
                )

        changes: list[dict[str, str]] = []

        def manage_permissions(owner: dict[str, Any], location: str) -> None:
            permissions = owner.get("permission")
            if not isinstance(permissions, dict):
                permissions = {}
                owner["permission"] = permissions
            for action in self.MANAGED_PERMISSION_ACTIONS:
                previous = permissions.get(action)
                if previous == "ask":
                    continue
                permissions[action] = "ask"
                changes.append({
                    "location": location,
                    "action": action,
                    "previous": (
                        str(previous)
                        if previous is not None
                        else "<unset>"
                    ),
                    "effective": "ask",
                })

        def allow_safe_runtime_tools(
            owner: dict[str, Any],
            location: str,
        ) -> None:
            permissions = owner.get("permission")
            if not isinstance(permissions, dict):
                permissions = {}
                owner["permission"] = permissions

            previous = permissions.get("todowrite")
            if previous == "allow":
                return

            permissions["todowrite"] = "allow"
            changes.append({
                "location": location,
                "action": "todowrite",
                "previous": (
                    str(previous)
                    if previous is not None
                    else "<unset>"
                ),
                "effective": "allow",
            })

        manage_permissions(payload, "permission")
        allow_safe_runtime_tools(payload, "permission")
        for container_key in ("agent", "agents"):
            agents = payload.get(container_key)
            if not isinstance(agents, dict):
                continue
            for agent_name, agent_payload in agents.items():
                if isinstance(agent_payload, dict):
                    manage_permissions(
                        agent_payload,
                        f"{container_key}.{agent_name}.permission",
                    )
                    allow_safe_runtime_tools(
                        agent_payload,
                        f"{container_key}.{agent_name}.permission",
                    )

        if has_registry_provider:
            provider_id = str(self.config.provider_id).strip()
            model_id = str(self.config.model_id).strip()
            provider_url = urlparse(
                str(self.config.provider_base_url or "").strip()
            )
            provider_host = (provider_url.hostname or "").lower()
            is_ollama_provider = (
                provider_url.port == 11434
                or "ollama" in provider_host
            )
            model_ids = []
            seen = set()
            for value in (*tuple(self.config.provider_models or ()), model_id):
                candidate = str(value or "").strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                model_ids.append(candidate)

            providers = payload.get("provider")
            if not isinstance(providers, dict):
                providers = {}
                payload["provider"] = providers

            options: dict[str, Any] = {
                "baseURL": str(
                    self.config.provider_base_url
                ).strip().rstrip("/"),
            }
            if self.config.provider_api_key:
                options["apiKey"] = (
                    "{env:ODYSSEUS_OPENCODE_MODEL_API_KEY}"
                )

            providers[provider_id] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": (
                    str(self.config.provider_name or "").strip()
                    or "Odysseus Model Endpoint"
                ),
                "options": options,
                "models": {
                    candidate: {
                        "name": candidate,
                        # OpenCode treats custom OpenAI-compatible models as
                        # text-only unless their input modalities are explicit.
                        # A file part still appears in session history in that
                        # state, but is stripped before the provider request.
                        "modalities": {
                            "input": (
                                ["text", "image"]
                                if is_vision_model(candidate)
                                else ["text"]
                            ),
                            "output": ["text"],
                        },
                        **(
                            {"attachment": True}
                            if is_vision_model(candidate)
                            else {}
                        ),
                        **(
                            {
                                "options": {
                                    "reasoningEffort": "none",
                                }
                            }
                            if (
                                is_ollama_provider
                                and is_vision_model(candidate)
                            )
                            else {}
                        ),
                    }
                    for candidate in model_ids
                },
            }
            payload.setdefault(
                "$schema",
                "https://opencode.ai/config.json",
            )
            payload["model"] = f"{provider_id}/{model_id}"
            changes.append({
                "location": f"provider.{provider_id}",
                "action": "model_registry",
                "previous": "<generated>",
                "effective": f"{provider_id}/{model_id}",
            })

        if (
            os.environ.get("PYTEST_CURRENT_TEST")
            or "pytest" in sys.modules
        ):
            config_root = self._log_root / "runtime-config"
        else:
            config_root = (
                self._project_root
                / "data"
                / "opencode_runtime"
            )
        config_root.mkdir(parents=True, exist_ok=True)
        output_path = config_root / "managed_config.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(output_path)
        return str(output_path), changes

    def log_event(self, event: str, **fields: Any) -> None:
        """Append one redacted event to the server or Odysseus-session log."""

        payload = {
            "timestamp": dt.datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "source": "odysseus-opencode-bridge",
            "event": str(event or "UNKNOWN"),
        }
        for key, value in fields.items():
            payload[str(key)] = self._sanitize_log_value(
                str(key),
                value,
            )

        line = (
            "[ODYSSEUS_BRIDGE] "
            + json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        session_id = str(
            fields.get("odysseus_session_id") or ""
        ).strip()
        target_path = (
            self.session_log_path(session_id)
            if session_id
            else self._log_path
        )

        with self._lock:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            handle = (
                self._log_handle
                if target_path == self._log_path
                else None
            )
            if handle is not None:
                handle.write(line)
                handle.flush()
                return
            with target_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            ) as fallback:
                fallback.write(line)

    @classmethod
    def _sanitize_log_value(
        cls,
        key: str,
        value: Any,
    ) -> Any:
        lowered = key.lower()
        if any(
            marker in lowered
            for marker in (
                "password",
                "authorization",
                "api_key",
                "apikey",
                "secret",
                "token",
            )
        ):
            return "[REDACTED]"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(child_key): cls._sanitize_log_value(
                    str(child_key),
                    child_value,
                )
                for child_key, child_value in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [
                cls._sanitize_log_value(key, item)
                for item in value
            ]
        if isinstance(value, str):
            return value[:1000]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)[:1000]

    def ensure_running(self) -> dict[str, Any]:
        with self._lock:
            running = self._probe_health()
            if running is not None:
                mode = (
                    "managed"
                    if self._process is not None
                    else "external"
                )
                self.log_event(
                    "SERVER_REUSED",
                    mode=mode,
                    owned=self._process is not None,
                    pid=self._pid(),
                    endpoint=self.config.base_url,
                    model_id=self.config.model_id,
                    provider_id=self.config.provider_id,
                    configured_workspace=(
                        self.config.workspace or str(Path.cwd())
                    ),
                )
                return {
                    "mode": mode,
                    "owned": self._process is not None,
                    "pid": self._pid(),
                    "health": running,
                }

            if not self.config.manage_server:
                raise OpenCodeServerLifecycleError(
                    "OpenCode server is unavailable and managed startup "
                    "is disabled."
                )

            host, port = self._managed_endpoint()
            executable = self._resolve_executable()
            command = self._command(executable, host, port)
            environment = os.environ.copy()
            (
                effective_config_path,
                permission_overlay_changes,
            ) = self._prepare_managed_config()
            if effective_config_path:
                environment["OPENCODE_CONFIG"] = effective_config_path
            if self.config.provider_api_key:
                environment[
                    "ODYSSEUS_OPENCODE_MODEL_API_KEY"
                ] = self.config.provider_api_key
            if self.config.username:
                environment["OPENCODE_SERVER_USERNAME"] = (
                    self.config.username
                )
            if self.config.password:
                environment["OPENCODE_SERVER_PASSWORD"] = (
                    self.config.password
                )

            workspace = (
                Path(self.config.workspace).resolve()
                if self.config.workspace
                else Path.cwd()
            )
            if not workspace.is_dir():
                raise OpenCodeServerLifecycleError(
                    f"OpenCode workspace does not exist: {workspace}"
                )

            creationflags = 0
            if os.name == "nt":
                creationflags = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )

            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._close_log_handle()
            self._log_handle = self._log_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            self._log_handle.write(
                "\n=== OpenCode managed server start ===\n"
            )
            self._log_handle.flush()
            self._generation += 1
            generation = self._generation
            self.log_event(
                "SERVER_START_REQUESTED",
                generation=generation,
                endpoint=self.config.base_url,
                provider_id=self.config.provider_id,
                model_id=self.config.model_id,
                config_default_model=self.config_default_model(),
                requested_model_id=self.config.model_id,
                effective_model_id=self.config.model_id,
                agent=self.config.agent,
                configured_workspace=(
                    self.config.workspace or str(Path.cwd())
                ),
                process_cwd=str(workspace),
                executable=executable,
                command=command,
                config_path=self.config.config_path or "",
                effective_config_path=effective_config_path,
                managed_permission_actions=list(
                    self.MANAGED_PERMISSION_ACTIONS
                ),
                permission_overlay_changes=permission_overlay_changes,
                managed=self.config.manage_server,
            )

            self._process = self.process_factory(
                command,
                cwd=str(workspace),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self._register_atexit()
            self.log_event(
                "SERVER_PROCESS_STARTED",
                generation=generation,
                pid=self._pid(),
                process_cwd=str(workspace),
                endpoint=self.config.base_url,
            )

            deadline = (
                self.monotonic()
                + self.config.startup_timeout_seconds
            )
            last_error = ""
            while self.monotonic() < deadline:
                if self._process.poll() is not None:
                    code = self._process.returncode
                    self.log_event(
                        "SERVER_STARTUP_EXITED",
                        generation=generation,
                        exit_code=code,
                    )
                    self._process = None
                    self._close_log_handle()
                    raise OpenCodeServerLifecycleError(
                        "Managed OpenCode server exited during startup "
                        f"with code {code}."
                    )
                try:
                    health = self.health_probe()
                    self.log_event(
                        "SERVER_HEALTHY",
                        generation=generation,
                        pid=self._pid(),
                        endpoint=self.config.base_url,
                        health=health,
                    )
                    return {
                        "mode": "managed",
                        "owned": True,
                        "pid": self._pid(),
                        "health": health,
                    }
                except Exception as exc:
                    last_error = str(exc)
                self.sleep(0.25)

            self.log_event(
                "SERVER_STARTUP_TIMEOUT",
                generation=generation,
                timeout_seconds=self.config.startup_timeout_seconds,
                last_error=last_error,
            )
            self.stop_owned()
            raise OpenCodeServerLifecycleError(
                "Managed OpenCode server did not become healthy within "
                f"{self.config.startup_timeout_seconds:g} seconds. "
                f"Last health error: {last_error}"
            )

    def stop_owned(self) -> bool:
        with self._lock:
            process = self._process
            if process is None:
                self.log_event(
                    "SERVER_STOP_SKIPPED",
                    reason="not_owned_or_not_running",
                )
                return False
            self._process = None
            pid = getattr(process, "pid", None)
            self.log_event(
                "SERVER_STOP_REQUESTED",
                pid=pid,
                endpoint=self.config.base_url,
            )

        if process.poll() is not None:
            self._close_log_handle()
            return True

        if os.name == "nt" and pid:
            taskkill = self.which("taskkill.exe") or self.which("taskkill")
            if taskkill:
                subprocess.run(
                    [taskkill, "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )

        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

        try:
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=3)
            except Exception:
                pass
        stopped = process.poll() is not None
        self.log_event(
            "SERVER_STOP_COMPLETED",
            pid=pid,
            stopped=stopped,
            exit_code=getattr(process, "returncode", None),
        )
        self._close_log_handle()
        return stopped

    def open_log_terminal(
        self,
        odysseus_session_id: str = "",
    ) -> dict[str, Any]:
        if os.name != "nt":
            raise OpenCodeServerLifecycleError(
                "OpenCode log terminal is currently supported on Windows."
            )

        session_id = str(odysseus_session_id or "").strip()
        target_path = (
            self.session_log_path(session_id)
            if session_id
            else self._log_path
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.touch(exist_ok=True)

        powershell = (
            self.which("powershell.exe")
            or self.which("pwsh.exe")
            or self.which("pwsh")
        )
        if not powershell:
            raise OpenCodeServerLifecycleError(
                "PowerShell was not found."
            )

        literal_path = str(target_path).replace("'", "''")
        scope_label = "Active chat" if session_id else "Server"
        model_label = str(self.config.model_id or "OpenCode").replace(
            "'", "''"
        )
        title = (
            f"Odysseus OpenCode — {scope_label} — {model_label}"
        ).replace("'", "''")
        command = (
            f"$Host.UI.RawUI.WindowTitle='{title}'; "
            f"Write-Host 'Following {scope_label} log for {model_label}'; "
            f"Write-Host '{literal_path}'; "
            "Write-Host ''; "
            f"Get-Content -LiteralPath '{literal_path}' -Tail 200 -Wait"
        )
        process = subprocess.Popen(
            [
                powershell,
                "-NoLogo",
                "-NoExit",
                "-Command",
                command,
            ],
            cwd=str(target_path.parent),
            creationflags=getattr(
                subprocess,
                "CREATE_NEW_CONSOLE",
                0,
            ),
        )
        terminal_pid = getattr(process, "pid", None)
        self.log_event(
            "LOG_TERMINAL_OPENED",
            odysseus_session_id=session_id,
            terminal_pid=terminal_pid,
            log_path=str(target_path),
            log_scope=("session" if session_id else "server"),
        )
        return {
            "opened": True,
            "pid": terminal_pid,
            "path": str(target_path),
            "scope": "session" if session_id else "server",
            "odysseus_session_id": session_id,
        }



    @staticmethod
    def _focus_existing_session_manager(
        title: str = "OpenCode Session Manager",
    ) -> bool:
        if os.name != "nt":
            return False

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            matches: list[int] = []
            callback_type = ctypes.WINFUNCTYPE(
                wintypes.BOOL,
                wintypes.HWND,
                wintypes.LPARAM,
            )

            def visit(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = int(user32.GetWindowTextLengthW(hwnd))
                if length <= 0:
                    return True
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                if str(title).casefold() in buffer.value.casefold():
                    matches.append(int(hwnd))
                return True

            user32.EnumWindows(callback_type(visit), 0)
            if not matches:
                return False

            hwnd = wintypes.HWND(matches[0])
            user32.ShowWindow(hwnd, 9)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False

    def _write_session_manager_launch_ticket(
        self,
        control_token: str = "",
    ) -> Path:
        ticket_root = self.log_root / "session-manager-tickets"
        ticket_root.mkdir(parents=True, exist_ok=True)
        ticket_name = hashlib.sha256(os.urandom(32)).hexdigest() + ".json"
        ticket_path = ticket_root / ticket_name

        payload = {
            "username": (
                self.config.username
                or os.environ.get("OPENCODE_SERVER_USERNAME", "")
            ),
            "password": (
                self.config.password
                or os.environ.get("OPENCODE_SERVER_PASSWORD", "")
            ),
            "control_token": str(control_token or "").strip(),
        }

        descriptor = os.open(
            str(ticket_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
        return ticket_path

    def _write_session_manager_handoff(
        self,
        handoff_path: Path,
        launch_ticket: Path,
        *,
        workspace: Path,
        native_session_id: str,
        control_url: str,
        log_path: Path,
    ) -> None:
        temporary = handoff_path.with_name(
            handoff_path.name
            + "."
            + hashlib.sha256(os.urandom(16)).hexdigest()
            + ".tmp"
        )
        payload = {
            "launch_ticket": str(launch_ticket),
            "server": self.config.base_url,
            "workspace": str(workspace),
            "model": self.config.model_id,
            "session_id": str(native_session_id or "").strip(),
            "control_url": str(control_url or "").strip(),
            "log_path": str(log_path),
        }
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, handoff_path)

    @staticmethod
    def _ensure_session_manager_terminal_profile() -> str:
        profile_name = 'Odysseus OpenCode Session Manager'
        if os.name != "nt":
            return ""
        local_app_data = str(os.environ.get("LOCALAPPDATA") or "").strip()
        if not local_app_data:
            return ""
        try:
            import json
            import uuid

            fragment_path = (
                Path(local_app_data)
                / "Microsoft"
                / "Windows Terminal"
                / "Fragments"
                / "Odysseus"
                / "session-manager.json"
            )
            fragment_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "profiles": [
                    {
                        "name": profile_name,
                        "hidden": True,
                        "scrollbarState": "hidden",
                        "historySize": 0,
                        "padding": "0",
                    }
                ]
            }
            temporary = fragment_path.with_name(
                fragment_path.name + "." + uuid.uuid4().hex + ".tmp"
            )
            try:
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                os.replace(temporary, fragment_path)
            finally:
                temporary.unlink(missing_ok=True)
            return profile_name
        except Exception:
            return ""


    def open_session_manager_terminal(
        self,
        opencode_session_id: str = "",
        *,
        control_url: str = "",
        control_token: str = "",
        odysseus_session_id: str = "",
    ) -> dict[str, Any]:
        launch = self.ensure_running()
        workspace = (
            Path(self.config.workspace).expanduser().resolve()
            if self.config.workspace
            else Path.cwd().resolve()
        )
        if not workspace.is_dir():
            raise OpenCodeServerLifecycleError(
                f"OpenCode workspace does not exist: {workspace}"
            )

        script_path = Path(__file__).with_name("session_manager_tui.py")
        if not script_path.is_file():
            raise OpenCodeServerLifecycleError(
                f"OpenCode Session Manager is missing: {script_path}"
            )

        native_session_id = str(opencode_session_id or "").strip()
        bridge_url = str(control_url or "").strip().rstrip("/")
        bridge_token = str(control_token or "").strip()
        if bool(bridge_url) != bool(bridge_token):
            raise OpenCodeServerLifecycleError(
                "Session Manager control URL and token must be provided together."
            )
        if bridge_url:
            parsed = urlparse(bridge_url)
            if (
                parsed.scheme not in {"http", "https"}
                or str(parsed.hostname or "").lower()
                not in {"127.0.0.1", "localhost", "::1"}
            ):
                raise OpenCodeServerLifecycleError(
                    "Session Manager control bridge must use a loopback URL."
                )

        log_session_id = str(
            odysseus_session_id or ""
        ).strip()
        log_path = (
            self.session_log_path(log_session_id)
            if log_session_id
            else self.log_path
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        launch_ticket = self._write_session_manager_launch_ticket(
            bridge_token
        )
        handoff_path = (
            self.log_root
            / "session-manager-handoff.json"
        )

        if self._focus_existing_session_manager():
            self._write_session_manager_handoff(
                handoff_path,
                launch_ticket,
                workspace=workspace,
                native_session_id=native_session_id,
                control_url=bridge_url,
                log_path=log_path,
            )
            self.log_event(
                "SESSION_MANAGER_TERMINAL_FOCUSED",
                configured_workspace=str(workspace),
                odysseus_session_id=log_session_id,
                opencode_session_id=native_session_id,
            )
            return {
                "opened": False,
                "focused": True,
                "terminal": "existing-window",
                "endpoint": self.config.base_url,
                "workspace": str(workspace),
                "opencode_session_id": native_session_id,
                "client": "odysseus-custom-tui",
            }

        handoff_path.unlink(missing_ok=True)
        tui_args = [
            sys.executable,
            str(script_path),
            "--server",
            self.config.base_url,
            "--workspace",
            str(workspace),
            "--model",
            self.config.model_id,
            "--log-path",
            str(log_path),
            "--launch-ticket",
            str(launch_ticket),
            "--handoff-path",
            str(handoff_path),
        ]
        if native_session_id:
            tui_args.extend(["--session", native_session_id])
        if bridge_url:
            tui_args.extend(["--control-url", bridge_url])

        environment = os.environ.copy()
        if self.config.username:
            environment["OPENCODE_SERVER_USERNAME"] = self.config.username
        if self.config.password:
            environment["OPENCODE_SERVER_PASSWORD"] = self.config.password
        if bridge_token:
            environment["ODYSSEUS_SESSION_MANAGER_TOKEN"] = bridge_token

        terminal = "console"
        creationflags = 0
        command = list(tui_args)

        if os.name == "nt":
            windows_terminal = shutil.which("wt.exe")
            if windows_terminal:
                terminal = "windows-terminal"
                terminal_profile = (
                    self._ensure_session_manager_terminal_profile()
                )
                command = [
                    windows_terminal,
                    "--window",
                    "new",
                    "--focus",
                    "--size",
                    "121,37",
                    "new-tab",
                    *(
                        ["--profile", terminal_profile]
                        if terminal_profile
                        else []
                    ),
                    "--title",
                    "OpenCode Session Manager",
                    "--suppressApplicationTitle",
                    "-d",
                    str(workspace),
                    *tui_args,
                ]
                creationflags = getattr(
                    subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0,
                )
            else:
                terminal = "windows-console"
                creationflags = (
                    getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                )

        try:
            self.process_factory(
                command,
                cwd=str(workspace),
                env=environment,
                creationflags=creationflags,
                start_new_session=(os.name != "nt"),
            )
        except Exception:
            handoff_path.unlink(missing_ok=True)
            launch_ticket.unlink(missing_ok=True)
            raise
        self.log_event(
            "SESSION_MANAGER_TERMINAL_OPENED",
            terminal=terminal,
            endpoint=self.config.base_url,
            configured_workspace=str(workspace),
            odysseus_session_id=log_session_id,
            opencode_session_id=native_session_id,
            command=command,
            server_mode=launch.get("mode"),
            control_enabled=bool(bridge_url),
        )
        return {
            "opened": True,
            "terminal": terminal,
            "endpoint": self.config.base_url,
            "workspace": str(workspace),
            "opencode_session_id": native_session_id,
            "client": "odysseus-custom-tui",
            "control_enabled": bool(bridge_url),
        }

    def open_native_tui(
        self,
        opencode_session_id: str,
        *,
        workspace: str = "",
    ) -> dict[str, Any]:
        """Open the installed OpenCode TUI attached to one native session."""
        launch = self.ensure_running()
        native_session_id = str(opencode_session_id or "").strip()
        if not native_session_id or any(
            marker in native_session_id for marker in ("\x00", "\r", "\n")
        ):
            raise OpenCodeServerLifecycleError(
                "A valid OpenCode session ID is required."
            )

        workspace_path = Path(
            str(workspace or self.config.workspace or Path.cwd())
        ).expanduser().resolve()
        if not workspace_path.is_dir():
            raise OpenCodeServerLifecycleError(
                f"OpenCode workspace does not exist: {workspace_path}"
            )

        executable = self._resolve_executable()
        cli_args = [
            executable,
            "attach",
            self.config.base_url,
            "--dir",
            str(workspace_path),
            "--session",
            native_session_id,
        ]
        environment = os.environ.copy()
        effective_config_path, _ = self._prepare_managed_config()
        if effective_config_path:
            environment["OPENCODE_CONFIG"] = effective_config_path
        if self.config.username:
            environment["OPENCODE_SERVER_USERNAME"] = self.config.username
        if self.config.password:
            environment["OPENCODE_SERVER_PASSWORD"] = self.config.password

        terminal = "direct"
        command = list(cli_args)
        creationflags = 0
        if os.name == "nt":
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            console_command = [
                comspec,
                "/d",
                "/s",
                "/k",
                subprocess.list2cmdline(cli_args),
            ]
            windows_terminal = shutil.which("wt.exe")
            if windows_terminal:
                terminal = "windows-terminal"
                command = [
                    windows_terminal,
                    "new-tab",
                    "--title",
                    "OpenCode",
                    "-d",
                    str(workspace_path),
                    *console_command,
                ]
                creationflags = getattr(
                    subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0,
                )
            else:
                terminal = "windows-console"
                command = console_command
                creationflags = (
                    getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                )

        try:
            process = self.process_factory(
                command,
                cwd=str(workspace_path),
                env=environment,
                creationflags=creationflags,
                start_new_session=(os.name != "nt"),
            )
        except FileNotFoundError as exc:
            raise OpenCodeServerLifecycleError(
                "OpenCode executable was not found."
            ) from exc
        except OSError as exc:
            raise OpenCodeServerLifecycleError(
                "The native OpenCode TUI could not be opened."
            ) from exc

        terminal_pid = getattr(process, "pid", None)
        self.log_event(
            "NATIVE_TUI_OPENED",
            terminal=terminal,
            terminal_pid=terminal_pid,
            endpoint=self.config.base_url,
            configured_workspace=str(workspace_path),
            opencode_session_id=native_session_id,
            server_mode=launch.get("mode"),
        )
        return {
            "opened": True,
            "terminal": terminal,
            "pid": terminal_pid,
            "endpoint": self.config.base_url,
            "workspace": str(workspace_path),
            "opencode_session_id": native_session_id,
            "client": "opencode-native-tui",
        }

    def open_package_terminal(
        self,
        package: str,
        action: str,
        argument: str = "",
    ) -> dict[str, Any]:
        package_value = str(package or "").strip().lower()
        action_value = str(action or "").strip().lower()
        argument_value = str(argument or "").strip()

        commands = {
            ("mcp", "add"): ("mcp", "add"),
            ("mcp", "list"): ("mcp", "list"),
            ("mcp", "auth"): ("mcp", "auth"),
            ("mcp", "logout"): ("mcp", "logout"),
            ("mcp", "debug"): ("mcp", "debug"),
            ("plugin", "install"): ("plugin",),
        }
        command_tail = commands.get((package_value, action_value))
        if command_tail is None:
            raise OpenCodeServerLifecycleError(
                "Unsupported OpenCode package action."
            )

        if len(argument_value) > 300 or any(
            marker in argument_value
            for marker in ("\x00", "\r", "\n")
        ):
            raise OpenCodeServerLifecycleError(
                "OpenCode package argument is invalid."
            )
        if (
            (package_value, action_value)
            in {
                ("mcp", "auth"),
                ("mcp", "logout"),
                ("mcp", "debug"),
                ("plugin", "install"),
            }
            and not argument_value
        ):
            raise OpenCodeServerLifecycleError(
                "This OpenCode package action requires an argument."
            )

        workspace = (
            Path(self.config.workspace).expanduser().resolve()
            if self.config.workspace
            else Path.cwd().resolve()
        )
        if not workspace.is_dir():
            raise OpenCodeServerLifecycleError(
                f"OpenCode workspace does not exist: {workspace}"
            )

        executable = self._resolve_executable()
        cli_args = [executable, *command_tail]
        if argument_value:
            cli_args.append(argument_value)

        environment = os.environ.copy()
        config_path = str(self.config.config_path or "").strip()
        if config_path:
            environment["OPENCODE_CONFIG"] = str(
                Path(config_path).expanduser().resolve()
            )

        terminal = "direct"
        creationflags = 0
        command = list(cli_args)
        title = (
            f"OpenCode {package_value} {action_value}"
        ).strip()

        if os.name == "nt":
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            command_line = subprocess.list2cmdline(cli_args)
            console_command = [
                comspec,
                "/d",
                "/s",
                "/k",
                command_line,
            ]
            windows_terminal = shutil.which("wt.exe")
            if windows_terminal:
                terminal = "windows-terminal"
                command = [
                    windows_terminal,
                    "new-tab",
                    "--title",
                    title,
                    "-d",
                    str(workspace),
                    *console_command,
                ]
                creationflags = getattr(
                    subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0,
                )
            else:
                terminal = "windows-console"
                command = console_command
                creationflags = (
                    getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                )

        process = self.process_factory(
            command,
            cwd=str(workspace),
            env=environment,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
        )
        terminal_pid = getattr(process, "pid", None)
        self.log_event(
            "PACKAGE_TERMINAL_OPENED",
            package=package_value,
            action=action_value,
            argument_length=len(argument_value),
            terminal=terminal,
            terminal_pid=terminal_pid,
            configured_workspace=str(workspace),
        )
        return {
            "opened": True,
            "terminal": terminal,
            "pid": terminal_pid,
            "package": package_value,
            "action": action_value,
            "workspace": str(workspace),
        }


    def _close_log_handle(self) -> None:
        handle = self._log_handle
        self._log_handle = None
        if handle is None:
            return
        try:
            handle.flush()
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

    def _probe_health(self) -> dict[str, Any] | None:
        try:
            return self.health_probe()
        except Exception:
            return None

    def _managed_endpoint(self) -> tuple[str, int]:
        parsed = urlparse(self.config.base_url)
        host = parsed.hostname or ""
        port = parsed.port
        if parsed.scheme != "http":
            raise OpenCodeServerLifecycleError(
                "Managed OpenCode startup requires an http loopback URL."
            )
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise OpenCodeServerLifecycleError(
                "Managed OpenCode startup is restricted to loopback URLs."
            )
        if port is None:
            raise OpenCodeServerLifecycleError(
                "Managed OpenCode URL must include an explicit port."
            )
        return host, port

    def _resolve_executable(self) -> str:
        configured = str(self.config.executable or "opencode")
        candidate = Path(configured)
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = (
            self.which(configured)
            or self.which("opencode.cmd")
            or self.which("opencode.exe")
            or self.which("opencode")
        )
        if not resolved:
            raise OpenCodeServerLifecycleError(
                "OpenCode executable was not found."
            )
        return resolved

    @staticmethod
    def _command(
        executable: str,
        host: str,
        port: int,
    ) -> list[str]:
        args = [
            executable,
            "serve",
            "--hostname",
            host,
            "--port",
            str(port),
        ]
        if os.name == "nt" and Path(executable).suffix.lower() in {
            ".cmd",
            ".bat",
        }:
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return [
                comspec,
                "/d",
                "/s",
                "/c",
                subprocess.list2cmdline(args),
            ]
        return args

    def _pid(self) -> int | None:
        if self._process is None:
            return None
        return getattr(self._process, "pid", None)

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self.stop_owned)
        self._atexit_registered = True
