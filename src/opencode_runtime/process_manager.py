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
        """Create a portable config overlay where Odysseus owns permissions.

        The user's base config is never modified. Managed actions resolve to
        ``ask`` so the per-Odysseus-session permission policy can either show
        approval UI or auto-approve proven workspace-local operations.
        """

        configured = str(self.config.config_path or "").strip()
        if not configured:
            return "", []

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

        def manage_permissions(
            owner: dict[str, Any],
            location: str,
        ) -> None:
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

        manage_permissions(payload, "permission")
        for container_key in ("agent", "agents"):
            agents = payload.get(container_key)
            if not isinstance(agents, dict):
                continue
            for agent_name, agent_payload in agents.items():
                if not isinstance(agent_payload, dict):
                    continue
                manage_permissions(
                    agent_payload,
                    f"{container_key}.{agent_name}.permission",
                )

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
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ) + "\n",
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
        command = (
            "$Host.UI.RawUI.WindowTitle='Odysseus OpenCode Logs'; "
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
