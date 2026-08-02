from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import fnmatch
import os
import secrets
import json
from pathlib import Path
import threading
import time
from typing import Any, AsyncIterator, Callable
import uuid

from .config import OpenCodeRuntimeConfig
from .normalizer import (
    OpenCodeEventNormalizer,
    extract_final_assistant_message,
)
from .model_registry import normalize_provider_base_url
from .process_manager import (
    OpenCodeServerLifecycleError,
    OpenCodeServerManager,
)
from .transport import OpenCodeHttpTransport, OpenCodeTransportError
import hashlib

TranscriptPersister = Callable[
    [str, str, str, dict[str, Any]],
    None,
]
SessionResolver = Callable[
    [str],
    dict[str, str] | None,
]


WORKSPACE_PATH_RUNTIME_RULE = """
Odysseus workspace path rule:
- Treat the configured OpenCode workspace as the project root.
- For files inside that workspace, use workspace-relative paths such as
  routes/example.py or src/package/module.py.
- Never prefix workspace files with / or \\ because that turns them into
  filesystem-root paths and may trigger external-directory permissions.
- Use an absolute path only when the user explicitly requests a location
  outside the configured workspace.
""".strip()


def sse_frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@dataclass(slots=True)
class ActiveOpenCodeRun:
    run_id: str
    odysseus_session_id: str
    opencode_session_id: str
    cancel_event: threading.Event = field(
        default_factory=threading.Event
    )
    status: str = "starting"


@dataclass(slots=True)
class PendingPermission:
    request_id: str
    odysseus_session_id: str
    opencode_session_id: str
    run_id: str
    action: str
    resources: tuple[str, ...]
    save: tuple[str, ...]
    title: str
    directory: str
    call_id: str
    created_at: float = field(default_factory=time.time)

    def public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action": self.action,
            "resources": list(self.resources),
            "title": self.title,
            "directory": self.directory,
            "call_id": self.call_id,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class SessionApproval:
    action: str
    resources: tuple[str, ...]
    workspace: str
    created_at: float = field(default_factory=time.time)

    def public_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "resources": list(self.resources),
            "workspace": self.workspace,
            "created_at": self.created_at,
        }



@dataclass(slots=True)
class SessionManagerControlGrant:
    odysseus_session_id: str
    expires_at: float


class OpenCodeRuntimeService:
    """Standalone OpenCode runtime lifecycle, without fallback routing."""

    PERMISSION_MODES = {"ask", "workspace"}
    WORKSPACE_SAFE_ACTIONS = {
        "read",
        "edit",
        "glob",
        "grep",
        "list",
        "lsp",
    }
    SENSITIVE_PATH_MARKERS = {
        ".env",
        ".git/config",
        "id_rsa",
        "id_ed25519",
        "credentials",
        "secret",
        ".pem",
        ".p12",
        ".pfx",
    }

    def __init__(
        self,
        config: OpenCodeRuntimeConfig | None = None,
        transport: OpenCodeHttpTransport | None = None,
        server_manager: OpenCodeServerManager | None = None,
        transcript_persister: TranscriptPersister | None = None,
        session_resolver: SessionResolver | None = None,
    ):
        base_config = config or OpenCodeRuntimeConfig.from_env()
        system_prompt = str(base_config.system_prompt or "").strip()
        if WORKSPACE_PATH_RUNTIME_RULE not in system_prompt:
            system_prompt = (
                system_prompt
                + "\n\n"
                + WORKSPACE_PATH_RUNTIME_RULE
            ).strip()
            base_config = replace(
                base_config,
                system_prompt=system_prompt,
            )
        self.config = base_config
        self.transport = transport or OpenCodeHttpTransport(self.config)
        self.server_manager = server_manager or OpenCodeServerManager(
            self.config,
            health_probe=self.transport.health,
        )
        self._transcript_persister = transcript_persister
        self._session_resolver = session_resolver
        self._sessions: dict[str, str] = {}
        self._runs: dict[str, ActiveOpenCodeRun] = {}
        self._run_by_conversation: dict[str, str] = {}
        self._run_errors: dict[str, str] = {}
        self._permission_modes: dict[str, str] = {}
        self._permission_approvals: dict[
            str,
            list[SessionApproval],
        ] = {}
        self._pending_permissions: dict[str, PendingPermission] = {}
        self._session_manager_controls: dict[
            str,
            SessionManagerControlGrant,
        ] = {}
        self._lock = threading.RLock()


    SESSION_MANAGER_CONTROL_TTL_SECONDS = 7200.0

    def _session_manager_control_root(
        self,
    ) -> Path:
        manager_root = getattr(
            self.server_manager,
            "log_root",
            None,
        )
        if manager_root:
            base = Path(manager_root)
        else:
            manager_log = getattr(
                self.server_manager,
                "log_path",
                None,
            )
            if manager_log:
                base = Path(manager_log).parent
            else:
                workspace = str(
                    getattr(
                        self.config,
                        "workspace",
                        "",
                    )
                    or ""
                ).strip()
                base = (
                    Path(workspace)
                    if workspace
                    else Path.cwd()
                )

        root = (
            base.expanduser().resolve()
            / "session-manager-controls"
        )
        root.mkdir(
            parents=True,
            exist_ok=True,
        )
        return root

    def _session_manager_control_path(
        self,
        token: str,
    ) -> Path:
        value = str(token or "").strip()
        digest = hashlib.sha256(
            value.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()
        return (
            self._session_manager_control_root()
            / f"{digest}.json"
        )

    def _write_session_manager_control(
        self,
        token: str,
        grant: SessionManagerControlGrant,
    ) -> None:
        path = self._session_manager_control_path(
            token
        )
        temporary = path.with_name(
            path.name
            + "."
            + secrets.token_hex(8)
            + ".tmp"
        )
        payload = {
            "odysseus_session_id": (
                grant.odysseus_session_id
            ),
            "expires_at": grant.expires_at,
        }

        descriptor = os.open(
            str(temporary),
            (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
            ),
            0o600,
        )
        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                )
                handle.write("\n")
            os.replace(
                temporary,
                path,
            )
        except Exception:
            try:
                temporary.unlink(
                    missing_ok=True
                )
            except Exception:
                pass
            raise

    def _read_session_manager_control(
        self,
        token: str,
    ) -> SessionManagerControlGrant | None:
        value = str(token or "").strip()
        if not value:
            return None

        path = self._session_manager_control_path(
            value
        )
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
            conversation_id = str(
                payload.get(
                    "odysseus_session_id"
                )
                or ""
            ).strip()
            expires_at = float(
                payload.get("expires_at")
                or 0.0
            )
        except Exception:
            return None

        if (
            not conversation_id
            or expires_at <= time.time()
        ):
            try:
                path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass
            return None

        return SessionManagerControlGrant(
            odysseus_session_id=(
                conversation_id
            ),
            expires_at=expires_at,
        )

    def _purge_expired_session_manager_controls(
        self,
    ) -> int:
        removed = 0
        now = time.time()
        for path in (
            self._session_manager_control_root()
            .glob("*.json")
        ):
            try:
                payload = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
                expires_at = float(
                    payload.get("expires_at")
                    or 0.0
                )
            except Exception:
                expires_at = 0.0

            if expires_at > now:
                continue
            try:
                path.unlink(
                    missing_ok=True
                )
                removed += 1
            except Exception:
                pass
        return removed

    def issue_session_manager_control(
        self,
        odysseus_session_id: str,
        *,
        ttl_seconds: float | None = None,
    ) -> dict[str, Any]:
        conversation_id = str(
            odysseus_session_id or ""
        ).strip()
        if not conversation_id:
            raise ValueError(
                "Odysseus session id is required."
            )

        now = time.time()
        ttl = (
            self.SESSION_MANAGER_CONTROL_TTL_SECONDS
            if ttl_seconds is None
            else max(
                30.0,
                float(ttl_seconds),
            )
        )
        token = secrets.token_urlsafe(32)
        grant = SessionManagerControlGrant(
            odysseus_session_id=(
                conversation_id
            ),
            expires_at=now + ttl,
        )

        self._purge_expired_session_manager_controls()
        self._write_session_manager_control(
            token,
            grant,
        )
        with self._lock:
            self._session_manager_controls[
                token
            ] = grant

        self._log(
            "SESSION_MANAGER_CONTROL_ISSUED",
            odysseus_session_id=(
                conversation_id
            ),
            expires_at=grant.expires_at,
            persistence="file-backed",
        )
        return {
            "token": token,
            "expires_at": grant.expires_at,
            "odysseus_session_id": (
                conversation_id
            ),
        }

    def resolve_session_manager_control(
        self,
        token: str,
    ) -> str:
        value = str(token or "").strip()
        if not value:
            raise PermissionError(
                "Session Manager control "
                "token is required."
            )

        grant = (
            self._read_session_manager_control(
                value
            )
        )
        if grant is None:
            with self._lock:
                self._session_manager_controls.pop(
                    value,
                    None,
                )
            raise PermissionError(
                "Session Manager control token "
                "is invalid or expired."
            )

        with self._lock:
            self._session_manager_controls[
                value
            ] = grant
        return grant.odysseus_session_id

    def release_session_manager_control(
        self,
        token: str,
    ) -> bool:
        value = str(token or "").strip()
        if not value:
            return False

        grant = (
            self._read_session_manager_control(
                value
            )
        )
        try:
            self._session_manager_control_path(
                value
            ).unlink(
                missing_ok=True
            )
        except Exception:
            pass

        with self._lock:
            cached = (
                self._session_manager_controls.pop(
                    value,
                    None,
                )
            )
        grant = grant or cached
        if grant is None:
            return False

        self._log(
            "SESSION_MANAGER_CONTROL_RELEASED",
            odysseus_session_id=(
                grant.odysseus_session_id
            ),
        )
        return True

    def revoke_session_manager_controls(
        self,
    ) -> int:
        with self._lock:
            memory_count = len(
                self._session_manager_controls
            )
            self._session_manager_controls.clear()

        file_count = 0
        for path in (
            self._session_manager_control_root()
            .glob("*.json")
        ):
            try:
                path.unlink(
                    missing_ok=True
                )
                file_count += 1
            except Exception:
                pass

        removed = max(
            memory_count,
            file_count,
        )
        if removed:
            self._log(
                "SESSION_MANAGER_CONTROLS_REVOKED",
                removed=removed,
            )
        return removed

    def session_manager_control_state(
        self,
        token: str,
    ) -> dict[str, Any]:
        value = str(token or "").strip()
        conversation_id = (
            self.resolve_session_manager_control(
                value
            )
        )
        grant = (
            self._read_session_manager_control(
                value
            )
        )
        with self._lock:
            native_session_id = (
                self._sessions.get(
                    conversation_id,
                    "",
                )
            )

        return {
            "ok": True,
            "odysseus_session_id": (
                conversation_id
            ),
            "opencode_session_id": (
                native_session_id
            ),
            "model_id": (
                self.config.model_id
            ),
            "endpoint_id": (
                self.config.endpoint_id
            ),
            "workspace": str(
                self._workspace_root()
            ),
            "server_owned": (
                self.server_manager.owns_process
            ),
            "permission": (
                self.permission_state(
                    conversation_id
                )
            ),
            "expires_at": (
                grant.expires_at
                if grant is not None
                else 0.0
            ),
        }

    def validate_turn_selection(
        self,
        *,
        endpoint_id: str,
        provider_id: str,
        provider_base_url: str,
        provider_models: tuple[str, ...] = (),
        model_id: str,
    ) -> str:
        selected_provider = str(provider_id or "").strip()
        selected_model = str(model_id or "").strip()
        if not selected_model:
            raise OpenCodeServerLifecycleError("OpenCode model must not be empty.")
        if not self._provider_selection_matches_loaded(
            provider_id=selected_provider,
            provider_base_url=provider_base_url,
        ):
            raise OpenCodeServerLifecycleError(
                "The selected model belongs to a different OpenCode provider."
            )
        loaded_models = {
            str(item or "").strip()
            for item in self.config.provider_models
            if str(item or "").strip()
        }
        selected_models = {
            str(item or "").strip()
            for item in provider_models
            if str(item or "").strip()
        }
        available_models = selected_models or loaded_models
        if available_models and selected_model not in available_models:
            raise OpenCodeServerLifecycleError(
                "The selected model is not available from the selected OpenCode provider."
            )
        return str(self.config.provider_id or selected_provider).strip()

    @staticmethod
    def _provider_url(value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            return ""
        return normalize_provider_base_url(normalized).casefold()

    def _provider_selection_matches_loaded(
        self,
        *,
        provider_id: str,
        provider_base_url: str,
    ) -> bool:
        selected_provider = str(provider_id or "").strip()
        loaded_provider = str(self.config.provider_id or "").strip()
        selected_base = self._provider_url(provider_base_url)
        loaded_base = self._provider_url(self.config.provider_base_url)
        if selected_base and loaded_base:
            return selected_base == loaded_base
        return bool(selected_provider and selected_provider == loaded_provider)

    @staticmethod
    def _native_timestamp(value: Any) -> tuple[str | None, float]:
        if value is None or value == "":
            return None, 0.0
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            milliseconds = float(value)
            if abs(milliseconds) < 100_000_000_000:
                milliseconds *= 1000.0
            try:
                rendered = datetime.fromtimestamp(
                    milliseconds / 1000.0,
                    tz=timezone.utc,
                ).isoformat().replace("+00:00", "Z")
            except (OverflowError, OSError, ValueError):
                return None, 0.0
            return rendered, milliseconds
        text = str(value or "").strip()
        if not text:
            return None, 0.0
        try:
            numeric = float(text)
        except ValueError:
            numeric = None
        if numeric is not None:
            return OpenCodeRuntimeService._native_timestamp(numeric)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
        except ValueError:
            return text, 0.0
        return parsed.isoformat().replace("+00:00", "Z"), parsed.timestamp() * 1000.0

    @staticmethod
    def _native_message_text(message: dict[str, Any]) -> str:
        parts = message.get("parts") if isinstance(message.get("parts"), list) else []
        text = " ".join(
            str(part.get("text") or "").strip()
            for part in parts
            if isinstance(part, dict)
            and str(part.get("type") or "") == "text"
            and str(part.get("text") or "").strip()
        )
        return " ".join(text.split())

    @classmethod
    def _native_session_identity(
        cls,
        item: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> tuple[str, int, str]:
        raw_model = item.get("model")
        model_name = ""
        if isinstance(raw_model, dict):
            model_name = str(
                raw_model.get("id")
                or raw_model.get("modelID")
                or ""
            ).strip()
        elif isinstance(raw_model, str):
            model_name = raw_model.strip()

        latest_user = ""
        latest_assistant = ""
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            info = message.get("info") if isinstance(message.get("info"), dict) else {}
            role = str(info.get("role") or "").strip().lower()
            if not model_name and role == "user":
                message_model = info.get("model")
                if isinstance(message_model, dict):
                    model_name = str(
                        message_model.get("modelID")
                        or message_model.get("id")
                        or ""
                    ).strip()
            text = cls._native_message_text(message)
            if not text:
                continue
            if role == "user" and not latest_user:
                latest_user = text
            elif role == "assistant" and not latest_assistant:
                latest_assistant = text
            if latest_user and latest_assistant and model_name:
                break

        preview = latest_user or latest_assistant
        if len(preview) > 180:
            preview = preview[:177].rstrip() + "..."
        return model_name, len(messages), preview

    @classmethod
    def _native_session_view(
        cls,
        item: dict[str, Any],
        active_id: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        session_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or item.get("name") or "").strip()
        if not title:
            title = "OpenCode Session"
        raw_time = item.get("time") if isinstance(item.get("time"), dict) else {}
        created_at, created_sort = cls._native_timestamp(
            raw_time.get("created", item.get("createdAt"))
        )
        updated_at, updated_sort = cls._native_timestamp(
            raw_time.get("updated", item.get("updatedAt"))
        )
        model_name, message_count, preview = cls._native_session_identity(
            item,
            messages or [],
        )
        return {
            "id": session_id,
            "title": title,
            "active": bool(session_id and session_id == active_id),
            "created_at": created_at,
            "updated_at": updated_at,
            "model_name": model_name,
            "message_count": message_count,
            "preview": preview,
            "_sort_timestamp": updated_sort or created_sort,
        }

    async def list_native_sessions(
        self,
        odysseus_session_id: str,
        *,
        workspace: str = "",
    ) -> dict[str, Any]:
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise ValueError("Odysseus session id is required.")
        await self.health()
        items = await asyncio.to_thread(
            self.transport.list_sessions,
            str(workspace or "").strip(),
        )
        normalized_workspace = str(workspace or "").strip()

        async def load_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
            session_id = str(item.get("id") or "").strip()
            if not session_id:
                return []
            try:
                return await asyncio.to_thread(
                    self.transport.list_messages,
                    session_id,
                    limit=10_000,
                    directory=normalized_workspace,
                )
            except Exception:
                return []

        message_lists = await asyncio.gather(*(
            load_messages(item) for item in items
        ))
        with self._lock:
            active_id = self._sessions.get(conversation_id, "")
        sessions = [
            self._native_session_view(item, active_id, messages)
            for item, messages in zip(items, message_lists)
            if str(item.get("id") or "").strip()
        ]
        sessions.sort(key=lambda item: item["_sort_timestamp"], reverse=True)
        for item in sessions:
            item.pop("_sort_timestamp", None)
        return {
            "ok": True,
            "odysseus_session_id": conversation_id,
            "active_opencode_session_id": active_id,
            "sessions": sessions,
        }

    async def create_conversation_session(
        self,
        odysseus_session_id: str,
        *,
        title: str,
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        session_title = str(title or "").strip()
        if not conversation_id:
            raise ValueError("Odysseus session id is required.")
        if not session_title:
            raise ValueError("OpenCode session title is required.")
        created = await asyncio.to_thread(self.transport.create_session, session_title)
        with self._lock:
            previous = self._sessions.get(conversation_id, "")
            self._sessions[conversation_id] = created
        message = f'OpenCode session created: "{session_title}".'
        self._persist_transcript_message(
            conversation_id,
            "assistant",
            message,
            {
                "runtime": "opencode",
                "kind": "native_session_created",
                "deterministic": True,
                "opencode_session_id": created,
                "previous_opencode_session_id": previous,
            },
        )
        return {
            "ok": True,
            "message": message,
            "odysseus_session_id": conversation_id,
            "opencode_session_id": created,
            "previous_opencode_session_id": previous,
            "title": session_title,
        }

    async def replace_conversation_session(
        self,
        odysseus_session_id: str,
        *,
        title: str = "",
    ) -> dict[str, Any]:
        return await self.create_conversation_session(
            odysseus_session_id,
            title=str(title or "").strip() or "OpenCode Session",
        )

    async def attach_conversation_session(
        self,
        odysseus_session_id: str,
        opencode_session_id: str,
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        native_id = str(opencode_session_id or "").strip()
        if not conversation_id or not native_id:
            raise ValueError("Both Odysseus and OpenCode session ids are required.")
        match = await asyncio.to_thread(self.transport.get_session, native_id)
        if match is None:
            raise KeyError("OpenCode session was not found.")
        title = str(match.get("title") or match.get("name") or "OpenCode Session").strip()
        with self._lock:
            previous = self._sessions.get(conversation_id, "")
            self._sessions[conversation_id] = native_id
        if previous:
            message = f'OpenCode session switched to "{title}".'
        else:
            message = f'OpenCode session started: "{title}".'
        self._persist_transcript_message(
            conversation_id,
            "assistant",
            message,
            {
                "runtime": "opencode",
                "kind": "native_session_attached",
                "deterministic": True,
                "opencode_session_id": native_id,
                "previous_opencode_session_id": previous,
            },
        )
        return {
            "ok": True,
            "message": message,
            "odysseus_session_id": conversation_id,
            "opencode_session_id": native_id,
            "previous_opencode_session_id": previous,
            "title": title,
        }

    async def rename_native_session(
        self,
        odysseus_session_id: str,
        opencode_session_id: str,
        title: str,
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        native_id = str(opencode_session_id or "").strip()
        normalized_title = str(title or "").strip()
        if not conversation_id or not native_id:
            raise ValueError("Both Odysseus and OpenCode session ids are required.")
        if not normalized_title:
            raise ValueError("OpenCode session title is required.")
        if await asyncio.to_thread(self.transport.get_session, native_id) is None:
            raise KeyError("OpenCode session was not found.")
        updated = await asyncio.to_thread(
            self.transport.update_session_title, native_id, normalized_title
        )
        return {
            "ok": True,
            "odysseus_session_id": conversation_id,
            "opencode_session_id": native_id,
            "title": str(updated.get("title") or normalized_title),
        }

    async def delete_native_session(
        self,
        odysseus_session_id: str,
        opencode_session_id: str,
        *,
        title: str = "",
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        native_id = str(opencode_session_id or "").strip()
        if not conversation_id or not native_id:
            raise ValueError("Both Odysseus and OpenCode session ids are required.")
        existing = await asyncio.to_thread(self.transport.get_session, native_id)
        resolved_title = str(
            (existing or {}).get("title")
            or (existing or {}).get("name")
            or title
            or "OpenCode Session"
        ).strip()
        deleted = await asyncio.to_thread(self.transport.delete_session, native_id)
        with self._lock:
            active_cleared = self._sessions.get(conversation_id, "") == native_id
            if active_cleared:
                self._sessions.pop(conversation_id, None)
        message = f'OpenCode session deleted: "{resolved_title}".'
        self._persist_transcript_message(
            conversation_id,
            "assistant",
            message,
            {
                "runtime": "opencode",
                "kind": "native_session_deleted",
                "deterministic": True,
                "opencode_session_id": "",
                "deleted_opencode_session_id": native_id,
                "active_session_cleared": active_cleared,
            },
        )
        return {
            "ok": True,
            "deleted": bool(deleted),
            "not_found": existing is None or not deleted,
            "active_cleared": active_cleared,
            "message": message,
            "odysseus_session_id": conversation_id,
            "opencode_session_id": native_id,
            "title": resolved_title,
        }

    async def open_package_terminal(
        self,
        odysseus_session_id: str,
        *,
        package: str,
        action: str,
        argument: str = "",
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise ValueError("Odysseus session id is required.")

        await self.health()
        result = await asyncio.to_thread(
            self.server_manager.open_package_terminal,
            package,
            action,
            argument,
        )
        self._log(
            "PACKAGE_CONTROL_OPENED",
            odysseus_session_id=conversation_id,
            package=result.get("package"),
            action=result.get("action"),
            terminal=result.get("terminal"),
            terminal_pid=result.get("pid"),
        )
        return {
            "ok": True,
            "odysseus_session_id": conversation_id,
            **result,
        }

    async def open_native_tui(
        self,
        odysseus_session_id: str,
        opencode_session_id: str,
        *,
        workspace: str = "",
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        native_id = str(opencode_session_id or "").strip()
        if not conversation_id:
            raise OpenCodeServerLifecycleError(
                "Odysseus session id is required."
            )
        if not native_id:
            raise OpenCodeServerLifecycleError(
                "A valid OpenCode session ID is required."
            )

        await self.health()
        existing = await asyncio.to_thread(
            self.transport.get_session,
            native_id,
        )
        if existing is None:
            raise KeyError("OpenCode session was not found.")
        result = await asyncio.to_thread(
            self.server_manager.open_native_tui,
            native_id,
            workspace=str(workspace or "").strip(),
        )
        self._log(
            "NATIVE_TUI_OPENED",
            odysseus_session_id=conversation_id,
            opencode_session_id=native_id,
            terminal=result.get("terminal"),
            endpoint=result.get("endpoint"),
        )
        return {
            "ok": True,
            "odysseus_session_id": conversation_id,
            **result,
        }

    async def restart_current_server(
        self,
        odysseus_session_id: str,
    ) -> dict[str, Any]:
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise ValueError("Odysseus session id is required.")

        result = await self.restart_server(
            endpoint_id=self.config.endpoint_id,
            provider_id=self.config.provider_id,
            provider_name=self.config.provider_name,
            provider_base_url=self.config.provider_base_url,
            provider_models=self.config.provider_models,
            provider_api_key=self.config.provider_api_key,
            model_id=self.config.model_id,
            workspace=str(self._workspace_root()),
        )
        native_session_id = await self._get_or_create_session(
            conversation_id,
            title="OpenCode Session",
        )
        return {
            **result,
            "ok": True,
            "odysseus_session_id": conversation_id,
            "opencode_session_id": native_session_id,
        }

    @classmethod
    def _normalize_permission_mode(cls, mode: str) -> str:
        value = str(mode or "ask").strip().lower()
        if value not in cls.PERMISSION_MODES:
            raise ValueError(
                "OpenCode permission mode must be 'ask' or 'workspace'."
            )
        return value

    def set_permission_mode(
        self,
        odysseus_session_id: str,
        mode: str,
    ) -> dict[str, Any]:
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise ValueError("Odysseus session id is required.")
        normalized = self._normalize_permission_mode(mode)
        with self._lock:
            self._permission_modes[conversation_id] = normalized
        self._log(
            "PERMISSION_MODE_UPDATED",
            odysseus_session_id=conversation_id,
            permission_mode=normalized,
        )
        return self.permission_state(conversation_id)

    def permission_state(
        self,
        odysseus_session_id: str,
    ) -> dict[str, Any]:
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise ValueError("Odysseus session id is required.")
        with self._lock:
            mode = self._permission_modes.get(conversation_id, "ask")
            approvals = [
                item.public_dict()
                for item in self._permission_approvals.get(
                    conversation_id,
                    [],
                )
            ]
            pending = [
                item.public_dict()
                for item in self._pending_permissions.values()
                if item.odysseus_session_id == conversation_id
            ]
        return {
            "ok": True,
            "odysseus_session_id": conversation_id,
            "mode": mode,
            "workspace": str(
                Path(
                    self.config.workspace or Path.cwd()
                ).expanduser().resolve()
            ),
            "approvals": approvals,
            "pending": pending,
        }

    def clear_permission_approvals(
        self,
        odysseus_session_id: str,
    ) -> dict[str, Any]:
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise ValueError("Odysseus session id is required.")
        with self._lock:
            removed = len(
                self._permission_approvals.pop(conversation_id, [])
            )
        self._log(
            "PERMISSION_APPROVALS_CLEARED",
            odysseus_session_id=conversation_id,
            removed=removed,
        )
        return {
            "ok": True,
            "odysseus_session_id": conversation_id,
            "removed": removed,
        }

    @staticmethod
    def _permission_values(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = []
        return tuple(
            str(item).strip()[:1000]
            for item in values
            if str(item).strip()
        )

    @staticmethod
    def _normalize_resource(value: str) -> str:
        return os.path.normcase(
            str(value or "").strip().replace("\\", "/")
        )

    def _workspace_root(self) -> Path:
        return Path(
            self.config.workspace or Path.cwd()
        ).expanduser().resolve()

    @staticmethod
    def _path_is_within(root: Path, candidate: Path) -> bool:
        try:
            root_value = os.path.normcase(str(root.resolve()))
            candidate_value = os.path.normcase(
                str(candidate.resolve(strict=False))
            )
            return (
                os.path.commonpath([root_value, candidate_value])
                == root_value
            )
        except (OSError, ValueError):
            return False

    def _directory_is_within_workspace(
        self,
        directory: str,
    ) -> bool:
        root = self._workspace_root()
        if not str(directory or "").strip():
            return True
        return self._path_is_within(
            root,
            Path(directory).expanduser(),
        )

    def _resource_path_within_workspace(
        self,
        resource: str,
        *,
        directory: str,
    ) -> bool:
        raw = str(resource or "").strip().strip("'\"")
        if not raw:
            return False
        if "://" in raw:
            return False

        wildcard_at = min(
            [
                index
                for token in ("*", "?")
                if (index := raw.find(token)) >= 0
            ]
            or [-1]
        )
        if wildcard_at == 0:
            return self._directory_is_within_workspace(directory)
        if wildcard_at > 0:
            raw = raw[:wildcard_at]

        raw = raw.rstrip("/\\")
        if not raw:
            return self._directory_is_within_workspace(directory)

        root = self._workspace_root()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            base = (
                Path(directory).expanduser()
                if str(directory or "").strip()
                else root
            )
            path = base / path
        return self._path_is_within(root, path)

    @classmethod
    def _resource_is_sensitive(cls, resource: str) -> bool:
        value = cls._normalize_resource(resource)
        return any(marker in value for marker in cls.SENSITIVE_PATH_MARKERS)

    def _workspace_policy_allows(
        self,
        pending: PendingPermission,
    ) -> bool:
        action = pending.action.lower()
        if action not in self.WORKSPACE_SAFE_ACTIONS:
            return False
        if not self._directory_is_within_workspace(pending.directory):
            return False
        if any(
            self._resource_is_sensitive(resource)
            for resource in pending.resources
        ):
            return False

        if action in {"read", "edit"}:
            return bool(pending.resources) and all(
                self._resource_path_within_workspace(
                    resource,
                    directory=pending.directory,
                )
                for resource in pending.resources
            )

        # Search/list/LSP actions are scoped by OpenCode's event directory.
        # Any attempt to escape that directory should arrive separately as
        # external_directory, which always remains approval-gated.
        return True

    def _session_approval_matches(
        self,
        conversation_id: str,
        pending: PendingPermission,
    ) -> bool:
        workspace = os.path.normcase(str(self._workspace_root()))
        requested = tuple(
            self._normalize_resource(value)
            for value in pending.resources
        )
        with self._lock:
            approvals = list(
                self._permission_approvals.get(conversation_id, [])
            )
        for approval in approvals:
            if approval.action != pending.action:
                continue
            if os.path.normcase(approval.workspace) != workspace:
                continue
            approved = tuple(
                self._normalize_resource(value)
                for value in approval.resources
            )
            if approved == requested:
                return True
        return False

    def _store_session_approval(
        self,
        conversation_id: str,
        pending: PendingPermission,
    ) -> None:
        approval = SessionApproval(
            action=pending.action,
            resources=pending.resources,
            workspace=str(self._workspace_root()),
        )
        with self._lock:
            bucket = self._permission_approvals.setdefault(
                conversation_id,
                [],
            )
            if not any(
                item.action == approval.action
                and item.resources == approval.resources
                and os.path.normcase(item.workspace)
                == os.path.normcase(approval.workspace)
                for item in bucket
            ):
                bucket.append(approval)

    def _remove_pending_permission(
        self,
        request_id: str,
    ) -> PendingPermission | None:
        with self._lock:
            return self._pending_permissions.pop(
                str(request_id or ""),
                None,
            )

    async def _reply_to_pending(
        self,
        pending: PendingPermission,
        *,
        reply: str,
        source: str,
    ) -> bool:
        workspace = str(self._workspace_root())
        effective_directory = (
            str(pending.directory or "").strip()
            or workspace
        )
        reply_fields = {
            "odysseus_session_id": pending.odysseus_session_id,
            "opencode_session_id": pending.opencode_session_id,
            "request_id": pending.request_id,
            "action": pending.action,
            "resources": list(pending.resources),
            "reply": reply,
            "source": source,
            "directory": effective_directory,
        }
        self._log(
            "PERMISSION_REPLY_SUBMITTING",
            **reply_fields,
        )
        try:
            accepted = await asyncio.to_thread(
                self.transport.reply_permission,
                pending.request_id,
                reply,
                directory=effective_directory,
            )
        except Exception as exc:
            self._log(
                "PERMISSION_REPLY_FAILED",
                **reply_fields,
                error=str(exc),
            )
            raise
        if not accepted:
            error = "OpenCode rejected the permission response."
            self._log(
                "PERMISSION_REPLY_FAILED",
                **reply_fields,
                error=error,
            )
            raise OpenCodeTransportError(error)
        self._remove_pending_permission(pending.request_id)
        self._log(
            "PERMISSION_REPLIED",
            **reply_fields,
        )
        return True

    async def resolve_permission_request(
        self,
        *,
        odysseus_session_id: str,
        request_id: str,
        decision: str,
    ) -> dict[str, Any]:
        conversation_id = str(odysseus_session_id or "").strip()
        normalized = str(decision or "").strip().lower()
        if normalized not in {"once", "session", "reject"}:
            raise ValueError(
                "Permission decision must be once, session, or reject."
            )
        with self._lock:
            pending = self._pending_permissions.get(
                str(request_id or "").strip()
            )
        if pending is None:
            raise KeyError("OpenCode permission request was not found.")
        if pending.odysseus_session_id != conversation_id:
            raise PermissionError(
                "Permission request does not belong to this Odysseus chat."
            )

        if normalized == "session":
            self._store_session_approval(conversation_id, pending)
        reply = "reject" if normalized == "reject" else "once"
        await self._reply_to_pending(
            pending,
            reply=reply,
            source=(
                "user_session_approval"
                if normalized == "session"
                else "user"
            ),
        )
        return {
            "ok": True,
            "request_id": pending.request_id,
            "decision": normalized,
            "mode": self._permission_modes.get(
                conversation_id,
                "ask",
            ),
            "approval_count": len(
                self._permission_approvals.get(
                    conversation_id,
                    [],
                )
            ),
        }

    async def _apply_permission_policy(
        self,
        pending: PendingPermission,
    ) -> dict[str, Any]:
        conversation_id = pending.odysseus_session_id
        mode = self._permission_modes.get(conversation_id, "ask")

        source = ""
        if self._session_approval_matches(conversation_id, pending):
            source = "odysseus_session_approval"
        elif mode == "workspace" and self._workspace_policy_allows(pending):
            source = "workspace_policy"

        if not source:
            return {
                "type": "runtime.permission_requested",
                "runtime": "opencode",
                **pending.public_dict(),
                "mode": mode,
            }

        await self._reply_to_pending(
            pending,
            reply="once",
            source=source,
        )
        return {
            "type": "runtime.permission_resolved",
            "runtime": "opencode",
            "request_id": pending.request_id,
            "action": pending.action,
            "resources": list(pending.resources),
            "decision": "once",
            "source": source,
        }

    def _persist_transcript_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any],
    ) -> bool:
        persister = self._transcript_persister
        if persister is None:
            return False

        normalized_session_id = str(session_id or "").strip()
        normalized_role = str(role or "").strip().lower()
        normalized_content = str(content or "")
        safe_metadata = dict(metadata or {})

        if not normalized_session_id:
            raise OpenCodeTransportError(
                "OpenCode transcript persistence requires a session id."
            )
        if normalized_role not in {"user", "assistant"}:
            raise OpenCodeTransportError(
                "OpenCode transcript persistence received an invalid role."
            )
        if not normalized_content.strip():
            raise OpenCodeTransportError(
                "OpenCode transcript persistence received empty content."
            )

        try:
            persister(
                normalized_session_id,
                normalized_role,
                normalized_content,
                safe_metadata,
            )
        except Exception as exc:
            self._log(
                "TRANSCRIPT_MESSAGE_PERSIST_FAILED",
                odysseus_session_id=normalized_session_id,
                role=normalized_role,
                content_length=len(normalized_content),
                run_id=safe_metadata.get("run_id"),
                error_type=type(exc).__name__,
            )
            raise OpenCodeTransportError(
                "OpenCode transcript persistence failed."
            ) from exc

        self._log(
            "TRANSCRIPT_MESSAGE_PERSISTED",
            odysseus_session_id=normalized_session_id,
            role=normalized_role,
            content_length=len(normalized_content),
            run_id=safe_metadata.get("run_id"),
            opencode_session_id=safe_metadata.get(
                "opencode_session_id"
            ),
            opencode_message_id=safe_metadata.get(
                "opencode_message_id"
            ),
        )
        return True

    def _log(self, event: str, **fields: Any) -> None:
        logger = getattr(self.server_manager, "log_event", None)
        if not callable(logger):
            return
        try:
            logger(event, **fields)
        except Exception:
            # Observability must never break the runtime path.
            pass

    @staticmethod
    def _unwrap_global_event(
        raw: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        if not isinstance(raw, dict):
            return "", {}, ""
        directory = str(raw.get("directory") or "")
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            payload = raw
        event_type = str(payload.get("type") or "")
        properties = payload.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        return event_type, properties, directory

    @staticmethod
    def _session_error_summary(properties: dict[str, Any]) -> str:
        value: Any = (
            properties.get("error")
            or properties.get("message")
            or properties.get("data")
            or properties
        )
        if isinstance(value, dict):
            for key in ("message", "error", "detail"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()[:1000]
            nested = value.get("data")
            if isinstance(nested, dict):
                for key in ("message", "error", "detail"):
                    candidate = nested.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()[:1000]
            candidate = value.get("name")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:1000]
            try:
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                )[:1000]
            except Exception:
                return str(value)[:1000]
        return str(value or "OpenCode session error.")[:1000]

    def _observe_raw_event(
        self,
        raw: dict[str, Any],
        *,
        run_id: str,
        odysseus_session_id: str,
        opencode_session_id: str,
    ) -> PendingPermission | None:
        event_type, properties, directory = self._unwrap_global_event(
            raw
        )
        if not event_type:
            return None

        session_id = str(
            properties.get("sessionID")
            or properties.get("sessionId")
            or ""
        )
        if session_id and session_id != opencode_session_id:
            return None

        observed_types = {
            "session.created",
            "session.updated",
            "session.status",
            "session.idle",
            "session.error",
            "todo.updated",
            "permission.asked",
            "permission.updated",
            "permission.replied",
            "permission.v2.asked",
            "permission.v2.replied",
        }
        if event_type in observed_types:
            error_payload: Any = None
            if event_type == "session.error":
                error_payload = (
                    properties.get("error")
                    or properties.get("data")
                    or properties
                )
                self._run_errors[run_id] = (
                    self._session_error_summary(properties)
                )
            self._log(
                "OPENCODE_EVENT",
                run_id=run_id,
                odysseus_session_id=odysseus_session_id,
                opencode_session_id=opencode_session_id,
                event_type=event_type,
                event_directory=directory,
                status=properties.get("status"),
                error_payload=error_payload,
            )

        if event_type in {
            "permission.replied",
            "permission.v2.replied",
        }:
            replied_id = str(
                properties.get("requestID")
                or properties.get("requestId")
                or properties.get("id")
                or ""
            ).strip()
            if replied_id:
                self._remove_pending_permission(replied_id)
            return None

        if event_type not in {
            "permission.asked",
            "permission.updated",
            "permission.v2.asked",
        }:
            return None

        request_id = str(
            properties.get("id")
            or properties.get("requestID")
            or properties.get("requestId")
            or ""
        ).strip()
        action = str(
            properties.get("action")
            or properties.get("permission")
            or properties.get("type")
            or ""
        ).strip()
        resources = self._permission_values(
            properties.get("resources")
            if properties.get("resources") is not None
            else (
                properties.get("patterns")
                if properties.get("patterns") is not None
                else properties.get("pattern")
            )
        )
        save = self._permission_values(properties.get("save"))
        metadata = (
            properties.get("metadata")
            if isinstance(properties.get("metadata"), dict)
            else {}
        )
        source = (
            properties.get("source")
            if isinstance(properties.get("source"), dict)
            else {}
        )
        call_id = str(
            source.get("callID")
            or source.get("callId")
            or properties.get("callID")
            or properties.get("callId")
            or ""
        )
        title = str(
            properties.get("title")
            or metadata.get("title")
            or metadata.get("description")
            or action
            or "OpenCode permission"
        )[:500]

        if not request_id or not action:
            self._log(
                "PERMISSION_EVENT_IGNORED",
                run_id=run_id,
                odysseus_session_id=odysseus_session_id,
                reason="missing_request_id_or_action",
                event_type=event_type,
            )
            return None

        pending = PendingPermission(
            request_id=request_id,
            odysseus_session_id=odysseus_session_id,
            opencode_session_id=opencode_session_id,
            run_id=run_id,
            action=action,
            resources=resources,
            save=save,
            title=title,
            directory=str(directory or ""),
            call_id=call_id,
        )
        with self._lock:
            self._pending_permissions[request_id] = pending

        self._log(
            "PERMISSION_REQUESTED",
            odysseus_session_id=odysseus_session_id,
            opencode_session_id=opencode_session_id,
            run_id=run_id,
            request_id=request_id,
            action=action,
            resources=list(resources),
            save=list(save),
            title=title,
            call_id=call_id,
            directory=str(directory or ""),
            permission_mode=self._permission_modes.get(
                odysseus_session_id,
                "ask",
            ),
        )
        return pending

    def _observe_normalized_event(
        self,
        event: dict[str, Any],
        *,
        run_id: str,
        odysseus_session_id: str,
        opencode_session_id: str,
    ) -> None:
        event_type = str(event.get("type") or "")
        if event_type not in {
            "tool.started",
            "tool.completed",
            "runtime.plan.updated",
            "runtime.status",
            "runtime.failed",
            "runtime.cancelled",
        }:
            return
        self._log(
            "NORMALIZED_EVENT",
            run_id=run_id,
            odysseus_session_id=odysseus_session_id,
            opencode_session_id=opencode_session_id,
            event_type=event_type,
            tool=event.get("tool"),
            call_id=event.get("call_id"),
            status=event.get("status"),
            success=event.get("success"),
            summary=event.get("summary"),
            output_summary=event.get("output_summary"),
            error=event.get("error"),
        )

    def _observe_message_snapshot(
        self,
        messages: list[dict[str, Any]],
        *,
        run_id: str,
        odysseus_session_id: str,
        opencode_session_id: str,
    ) -> None:
        paths: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            info = message.get("info")
            if not isinstance(info, dict):
                continue
            path = info.get("path")
            if not isinstance(path, dict):
                continue
            observed = {
                "cwd": str(path.get("cwd") or ""),
                "root": str(path.get("root") or ""),
            }
            if observed not in paths:
                paths.append(observed)
        self._log(
            "MESSAGE_SNAPSHOT",
            run_id=run_id,
            odysseus_session_id=odysseus_session_id,
            opencode_session_id=opencode_session_id,
            message_count=len(messages),
            observed_paths=paths,
        )

    async def health(self) -> dict[str, Any]:
        self._log(
            "HEALTH_CHECK_REQUESTED",
            endpoint=self.config.base_url,
            model_id=self.config.model_id,
            configured_workspace=(
                self.config.workspace or str(Path.cwd())
            ),
        )
        launch = await asyncio.to_thread(
            self.server_manager.ensure_running
        )
        value = launch["health"]
        self._log(
            "HEALTH_CHECK_COMPLETED",
            mode=launch["mode"],
            owned=launch["owned"],
            pid=launch["pid"],
            healthy=bool(value.get("healthy", True)),
        )
        return {
            "ok": bool(value.get("healthy", True)),
            "runtime": "opencode",
            "launch": {
                "mode": launch["mode"],
                "owned": launch["owned"],
                "pid": launch["pid"],
            },
            "server": value,
            "model": {
                "endpoint_id": self.config.endpoint_id,
                "provider_id": self.config.provider_id,
                "model_id": self.config.model_id,
                "agent": self.config.agent,
            },
            "workspace": self.config.workspace or str(Path.cwd()),
            "log_root": str(
                getattr(self.server_manager, "log_root", "") or ""
            ),
            "server_log_path": str(
                getattr(self.server_manager, "log_path", "") or ""
            ),
            "log_path": str(
                getattr(self.server_manager, "log_path", "") or ""
            ),
            "config": {
                "path": self.config.config_path or "",
                "default_model": (
                    self.server_manager.config_default_model()
                    if hasattr(
                        self.server_manager,
                        "config_default_model",
                    )
                    else ""
                ),
                "requested_model": self.config.model_id,
                "effective_model": self.config.model_id,
            },
        }

    async def stream(
        self,
        *,
        message: str,
        odysseus_session_id: str,
        title: str = "",
        permission_mode: str = "ask",
        provider_id: str = "",
        model_id: str = "",
        attachment_parts: tuple[dict[str, Any], ...] = (),
        attachment_refs: tuple[dict[str, Any], ...] = (),
    ) -> AsyncIterator[str]:
        normalized_message = str(message or "").strip()
        conversation_id = str(odysseus_session_id or "").strip()
        has_attachments = bool(attachment_parts)
        if not normalized_message and not has_attachments:
            yield sse_frame({
                "type": "runtime.failed",
                "runtime": "opencode",
                "error": "Message must not be empty.",
            })
            return
        if not conversation_id:
            yield sse_frame({
                "type": "runtime.failed",
                "runtime": "opencode",
                "error": "Odysseus session id is required.",
            })
            return

        try:
            normalized_permission_mode = self._normalize_permission_mode(
                permission_mode
            )
        except ValueError as exc:
            yield sse_frame({
                "type": "runtime.failed",
                "runtime": "opencode",
                "error": str(exc),
            })
            return
        with self._lock:
            self._permission_modes[conversation_id] = (
                normalized_permission_mode
            )

        self._log(
            "RUN_REQUESTED",
            odysseus_session_id=conversation_id,
            prompt_length=len(normalized_message),
            attachment_count=len(attachment_parts),
            title_length=len(str(title or "")),
            model_id=str(model_id or self.config.model_id),
            configured_workspace=(
                self.config.workspace or str(Path.cwd())
            ),
        )

        with self._lock:
            existing_run_id = self._run_by_conversation.get(conversation_id)
            if existing_run_id and existing_run_id in self._runs:
                self._log(
                    "RUN_REJECTED",
                    odysseus_session_id=conversation_id,
                    reason="already_active",
                    active_run_id=existing_run_id,
                )
                yield sse_frame({
                    "type": "runtime.failed",
                    "runtime": "opencode",
                    "error": "An OpenCode run is already active for this chat.",
                })
                return

        run_id = uuid.uuid4().hex
        session_id = ""
        event_thread: threading.Thread | None = None
        queue: asyncio.Queue[dict[str, Any] | BaseException] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        try:
            launch = await asyncio.to_thread(
                self.server_manager.ensure_running
            )
            self._log(
                "RUN_SERVER_READY",
                run_id=run_id,
                odysseus_session_id=conversation_id,
                mode=launch.get("mode"),
                owned=launch.get("owned"),
                pid=launch.get("pid"),
            )
            if has_attachments:
                with self._lock:
                    session_id = self._sessions.get(conversation_id, "")
                if not session_id:
                    raise OpenCodeServerLifecycleError(
                        "Start or select an OpenCode session before sending an image."
                    )
            else:
                session_id = await self._get_or_create_session(
                    conversation_id,
                    title=title,
                )
            run = ActiveOpenCodeRun(
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
            )
            with self._lock:
                self._runs[run_id] = run
                self._run_by_conversation[conversation_id] = run_id

            self._log(
                "RUN_STARTED",
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
                model_id=self.config.model_id,
                configured_workspace=(
                    self.config.workspace or str(Path.cwd())
                ),
            )
            try:
                baseline_messages = await asyncio.to_thread(
                    self.transport.list_messages,
                    session_id,
                )
            except OpenCodeTransportError:
                baseline_messages = []
            normalizer = OpenCodeEventNormalizer.from_snapshot(
                session_id,
                baseline_messages,
            )

            def event_worker() -> None:
                self._log(
                    "EVENT_STREAM_OPENING",
                    run_id=run_id,
                    odysseus_session_id=conversation_id,
                    opencode_session_id=session_id,
                )
                try:
                    for raw in self.transport.iter_global_events(
                        run.cancel_event
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, raw)
                except BaseException as exc:
                    self._log(
                        "EVENT_STREAM_ERROR",
                        run_id=run_id,
                        error=str(exc),
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    self._log(
                        "EVENT_STREAM_CLOSED",
                        run_id=run_id,
                        cancelled=run.cancel_event.is_set(),
                    )

            event_thread = threading.Thread(
                target=event_worker,
                daemon=True,
                name=f"OpenCodeEvents-{run_id[:8]}",
            )
            event_thread.start()

            # Persist the user turn before OpenCode starts work.
            await asyncio.to_thread(
                self._persist_transcript_message,
                conversation_id,
                "user",
                normalized_message or "[Image attachment]",
                {
                    "runtime": "opencode",
                    "run_id": run_id,
                    "opencode_session_id": session_id,
                    "provider_id": self.config.provider_id,
                    "model_id": self.config.model_id,
                    "transcript_phase": "user",
                    "workspace": str(self._workspace_root()),
                    "attachments": list(attachment_refs),
                },
            )

            yield sse_frame({
                "type": "runtime.started",
                "runtime": "opencode",
                "run_id": run_id,
                "opencode_session_id": session_id,
                "objective": "Complete the requested task",
                "label": "OpenCode runtime",
            })

            self._log(
                "PROMPT_SUBMITTING",
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
                prompt_length=len(normalized_message),
                attachment_count=len(attachment_parts),
                requested_model_id=self.config.model_id,
                effective_model_id=self.config.model_id,
            )
            if provider_id or model_id:
                submit_kwargs: dict[str, Any] = {
                    "provider_id": str(provider_id or self.config.provider_id),
                    "model_id": str(model_id or self.config.model_id),
                }
                if attachment_parts:
                    submit_kwargs["file_parts"] = attachment_parts
                await asyncio.to_thread(
                    self.transport.submit_prompt,
                    session_id,
                    normalized_message,
                    **submit_kwargs,
                )
            elif attachment_parts:
                await asyncio.to_thread(
                    self.transport.submit_prompt,
                    session_id,
                    normalized_message,
                    file_parts=attachment_parts,
                )
            else:
                await asyncio.to_thread(
                    self.transport.submit_prompt,
                    session_id,
                    normalized_message,
                )
            self._log(
                "PROMPT_ACCEPTED",
                run_id=run_id,
                opencode_session_id=session_id,
            )
            yield sse_frame({
                "type": "runtime.accepted",
                "runtime": "opencode",
                "run_id": run_id,
                "opencode_session_id": session_id,
                "attachment_count": len(attachment_parts),
            })
            run.status = "running"
            yield sse_frame({
                "type": "runtime.status",
                "runtime": "opencode",
                "status": "running",
                "label": "OpenCode is working",
            })

            started = time.monotonic()
            idle_seen = False
            next_message_reconcile = 0.0
            message_poll_failures = 0
            while not idle_seen:
                if run.cancel_event.is_set():
                    run.status = "cancelled"
                    yield sse_frame({
                        "type": "runtime.cancelled",
                        "runtime": "opencode",
                        "run_id": run_id,
                    })
                    return

                if (
                    time.monotonic() - started
                    > self.config.run_timeout_seconds
                ):
                    await asyncio.to_thread(
                        self.transport.abort_session,
                        session_id,
                    )
                    run.cancel_event.set()
                    run.status = "failed"
                    yield sse_frame({
                        "type": "runtime.failed",
                        "runtime": "opencode",
                        "run_id": run_id,
                        "error": "OpenCode run timed out.",
                    })
                    return

                now = time.monotonic()
                if now >= next_message_reconcile:
                    try:
                        snapshot = await asyncio.to_thread(
                            self.transport.list_messages,
                            session_id,
                        )
                        message_poll_failures = 0
                        self._observe_message_snapshot(
                            snapshot,
                            run_id=run_id,
                            odysseus_session_id=conversation_id,
                            opencode_session_id=session_id,
                        )
                        for event in normalizer.reconcile_messages(snapshot):
                            event["run_id"] = run_id
                            self._observe_normalized_event(
                                event,
                                run_id=run_id,
                                odysseus_session_id=conversation_id,
                                opencode_session_id=session_id,
                            )
                            yield sse_frame(event)
                    except OpenCodeTransportError:
                        message_poll_failures += 1
                        if message_poll_failures >= 3:
                            raise
                    next_message_reconcile = now + 0.75

                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=0.35,
                    )
                except asyncio.TimeoutError:
                    continue

                if isinstance(item, BaseException):
                    raise OpenCodeTransportError(
                        f"OpenCode event stream failed: {item}"
                    )

                pending_permission = self._observe_raw_event(
                    item,
                    run_id=run_id,
                    odysseus_session_id=conversation_id,
                    opencode_session_id=session_id,
                )
                if pending_permission is not None:
                    permission_event = await self._apply_permission_policy(
                        pending_permission
                    )
                    yield sse_frame(permission_event)

                for event in normalizer.normalize(item):
                    if event.get("type") == "runtime.idle":
                        idle_seen = True
                        self._log(
                            "RUN_IDLE_OBSERVED",
                            run_id=run_id,
                            opencode_session_id=session_id,
                        )
                        continue
                    event["run_id"] = run_id
                    self._observe_normalized_event(
                        event,
                        run_id=run_id,
                        odysseus_session_id=conversation_id,
                        opencode_session_id=session_id,
                    )
                    yield sse_frame(event)

            messages = await asyncio.to_thread(
                self.transport.list_messages,
                session_id,
            )
            self._observe_message_snapshot(
                messages,
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
            )
            for event in normalizer.reconcile_messages(messages):
                event["run_id"] = run_id
                self._observe_normalized_event(
                    event,
                    run_id=run_id,
                    odysseus_session_id=conversation_id,
                    opencode_session_id=session_id,
                )
                yield sse_frame(event)
            final = extract_final_assistant_message(
                messages,
                excluded_message_ids=normalizer.baseline_message_ids,
            )
            if not final["text"]:
                run.status = "failed"
                session_error = self._run_errors.pop(run_id, "")
                reason = (
                    "session_error"
                    if session_error
                    else "idle_without_final_response"
                )
                failure_message = (
                    f"OpenCode session failed: {session_error}"
                    if session_error
                    else (
                        "OpenCode returned no assistant output. "
                        "Start a fresh OpenCode session and retry."
                    )
                )
                self._log(
                    "RUN_FAILED",
                    run_id=run_id,
                    odysseus_session_id=conversation_id,
                    opencode_session_id=session_id,
                    reason=reason,
                    error=session_error or None,
                )
                yield sse_frame({
                    "type": "runtime.failed",
                    "runtime": "opencode",
                    "run_id": run_id,
                    "error": failure_message,
                })
                return

            # Persist the final assistant turn before reporting completion.
            await asyncio.to_thread(
                self._persist_transcript_message,
                conversation_id,
                "assistant",
                final["text"],
                {
                    "runtime": "opencode",
                    "run_id": run_id,
                    "opencode_session_id": session_id,
                    "provider_id": final["provider_id"],
                    "model_id": final["model_id"],
                    "opencode_message_id": final["message_id"],
                    "transcript_phase": "assistant",
                    "workspace": str(self._workspace_root()),
                },
            )

            run.status = "completed"
            self._log(
                "RUN_COMPLETED",
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
                provider_id=final["provider_id"],
                model_id=final["model_id"],
                message_id=final["message_id"],
                response_length=len(final["text"]),
            )
            yield sse_frame({
                "type": "runtime.completed",
                "runtime": "opencode",
                "run_id": run_id,
                "content": final["text"],
                "provider_id": final["provider_id"],
                "model_id": final["model_id"],
                "message_id": final["message_id"],
            })

        except (
            OpenCodeServerLifecycleError,
            OpenCodeTransportError,
        ) as exc:
            self._log(
                "RUN_FAILED",
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            yield sse_frame({
                "type": "runtime.failed",
                "runtime": "opencode",
                "run_id": run_id,
                "error": str(exc),
            })
        except asyncio.CancelledError:
            self._log(
                "RUN_TASK_CANCELLED",
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
            )
            if session_id:
                await asyncio.to_thread(
                    self.transport.abort_session,
                    session_id,
                )
            raise
        finally:
            with self._lock:
                run = self._runs.pop(run_id, None)
                self._run_by_conversation.pop(conversation_id, None)
            self._run_errors.pop(run_id, None)
            with self._lock:
                stale_permission_ids = [
                    request_id
                    for request_id, pending in self._pending_permissions.items()
                    if pending.run_id == run_id
                ]
                for request_id in stale_permission_ids:
                    self._pending_permissions.pop(request_id, None)
            if run is not None:
                run.cancel_event.set()
            if event_thread is not None:
                event_thread.join(timeout=2.0)
            self._log(
                "RUN_CLEANUP_COMPLETED",
                run_id=run_id,
                odysseus_session_id=conversation_id,
                opencode_session_id=session_id,
            )

    def _assert_idle(self) -> None:
        with self._lock:
            if self._runs:
                raise OpenCodeServerLifecycleError(
                    "An OpenCode run is active. Stop it before changing "
                    "server configuration."
                )

    async def restart_server(
        self,
        *,
        model_id: str,
        workspace: str,
        endpoint_id: str = "",
        provider_id: str = "",
        provider_name: str = "",
        provider_base_url: str = "",
        provider_models: tuple[str, ...] = (),
        provider_api_key: str = "",
    ) -> dict[str, Any]:
        self._assert_idle()
        self._log(
            "SERVER_RESTART_REQUESTED",
            requested_model_id=str(model_id or "").strip(),
            requested_workspace=str(workspace or "").strip(),
            requested_endpoint_id=str(endpoint_id or "").strip(),
            requested_provider_id=str(provider_id or "").strip(),
            current_model_id=self.config.model_id,
            current_workspace=self.config.workspace or str(Path.cwd()),
        )

        selected_model = str(model_id or "").strip()
        selected_workspace = str(workspace or "").strip()
        selected_endpoint = (
            str(endpoint_id or "").strip()
            or self.config.endpoint_id
        )
        selected_provider = (
            str(provider_id or "").strip()
            or self.config.provider_id
        )
        selected_provider_name = (
            str(provider_name or "").strip()
            or self.config.provider_name
        )
        selected_provider_base = (
            str(provider_base_url or "").strip().rstrip("/")
            or self.config.provider_base_url
        )
        selected_provider_models = tuple(
            str(item or "").strip()
            for item in (
                provider_models
                or self.config.provider_models
                or ()
            )
            if str(item or "").strip()
        )
        if not selected_model:
            raise OpenCodeServerLifecycleError(
                "OpenCode model must not be empty."
            )
        if not selected_workspace:
            raise OpenCodeServerLifecycleError(
                "OpenCode workspace must not be empty."
            )

        workspace_path = Path(selected_workspace).expanduser().resolve()
        if not workspace_path.is_dir():
            raise OpenCodeServerLifecycleError(
                f"OpenCode workspace does not exist: {workspace_path}"
            )

        previous_workspace = self._workspace_root()
        workspace_changed = (
            os.path.normcase(str(previous_workspace))
            != os.path.normcase(str(workspace_path))
        )

        external_healthy = False
        if not self.server_manager.owns_process:
            try:
                await asyncio.to_thread(self.transport.health)
                external_healthy = True
            except Exception:
                external_healthy = False
        if external_healthy:
            raise OpenCodeServerLifecycleError(
                "The connected OpenCode server is external. Stop it "
                "manually before applying managed workspace settings."
            )

        with self._lock:
            if workspace_changed:
                approval_count = sum(
                    len(items)
                    for items in self._permission_approvals.values()
                )
                self._permission_approvals.clear()
                self._pending_permissions.clear()
            else:
                approval_count = 0

        if workspace_changed:
            self._log(
                "PERMISSION_APPROVALS_INVALIDATED",
                reason="workspace_changed",
                previous_workspace=str(previous_workspace),
                new_workspace=str(workspace_path),
                removed=approval_count,
            )

        await asyncio.to_thread(self.server_manager.stop_owned)

        updated = replace(
            self.config,
            provider_id=selected_provider,
            model_id=selected_model,
            endpoint_id=selected_endpoint,
            provider_name=selected_provider_name,
            provider_base_url=selected_provider_base,
            provider_models=selected_provider_models,
            provider_api_key=str(provider_api_key or ""),
            workspace=str(workspace_path),
        )
        self.config = updated
        self.transport = OpenCodeHttpTransport(updated)
        self.server_manager = OpenCodeServerManager(
            updated,
            health_probe=self.transport.health,
            log_root=getattr(
                self.server_manager,
                "log_root",
                None,
            ),
        )
        result = await self.health()
        self._log(
            "SERVER_RESTART_COMPLETED",
            endpoint_id=selected_endpoint,
            provider_id=selected_provider,
            model_id=selected_model,
            workspace=str(workspace_path),
            launch=result.get("launch"),
        )
        self.revoke_session_manager_controls()
        return result

    async def stop_server(self) -> dict[str, Any]:
        self._assert_idle()
        self._log(
            "SERVER_STOP_API_REQUESTED",
            owned=self.server_manager.owns_process,
        )
        if not self.server_manager.owns_process:
            return {
                "ok": True,
                "stopped": False,
                "owned": False,
                "reason": "not_owned",
            }

        stopped = await asyncio.to_thread(
            self.server_manager.stop_owned
        )
        with self._lock:
            self._sessions.clear()
        return {
            "ok": True,
            "stopped": bool(stopped),
            "owned": True,
        }



    async def prepare_runtime(
        self,
        *,
        odysseus_session_id: str,
        title: str,
        permission_mode: str,
        model_id: str,
        workspace: str,
        endpoint_id: str = "",
        provider_id: str = "",
        provider_name: str = "",
        provider_base_url: str = "",
        provider_models: tuple[str, ...] = (),
        provider_api_key: str = "",
        create_if_missing: bool = True,
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise OpenCodeServerLifecycleError(
                "Odysseus session id is required."
            )

        selected_model = str(model_id or "").strip()
        selected_workspace = str(workspace or "").strip()
        if not selected_model:
            raise OpenCodeServerLifecycleError(
                "OpenCode model must not be empty."
            )
        if not selected_workspace:
            raise OpenCodeServerLifecycleError(
                "OpenCode workspace must not be empty."
            )

        workspace_path = Path(selected_workspace).expanduser().resolve()
        if not workspace_path.is_dir():
            raise OpenCodeServerLifecycleError(
                f"OpenCode workspace does not exist: {workspace_path}"
            )

        current_workspace = self._workspace_root()
        provider_matches = self._provider_selection_matches_loaded(
            provider_id=provider_id,
            provider_base_url=provider_base_url,
        )
        selected_provider_models = {
            str(item or "").strip()
            for item in provider_models
            if str(item or "").strip()
        }
        loaded_provider_models = {
            str(item or "").strip()
            for item in self.config.provider_models
            if str(item or "").strip()
        }
        available_models = selected_provider_models or loaded_provider_models
        selected_model_loaded = (
            not available_models
            or selected_model in available_models
        )
        config_matches = (
            provider_matches
            and selected_model_loaded
            and os.path.normcase(str(current_workspace))
            == os.path.normcase(str(workspace_path))
        )

        if config_matches:
            result = await self.health()
            self.config = replace(
                self.config,
                endpoint_id=(
                    str(endpoint_id or "").strip()
                    or self.config.endpoint_id
                ),
                provider_name=(
                    str(provider_name or "").strip()
                    or self.config.provider_name
                ),
                provider_base_url=(
                    str(provider_base_url or "").strip().rstrip("/")
                    or self.config.provider_base_url
                ),
                provider_models=tuple(sorted(
                    loaded_provider_models | selected_provider_models
                )),
                model_id=selected_model,
            )
            self.transport.config = self.config
            restarted = False
        else:
            result = await self.restart_server(
                endpoint_id=endpoint_id,
                provider_id=provider_id,
                provider_name=provider_name,
                provider_base_url=provider_base_url,
                provider_models=provider_models,
                provider_api_key=provider_api_key,
                model_id=selected_model,
                workspace=str(workspace_path),
            )
            restarted = True

        normalized_permission = self._normalize_permission_mode(
            permission_mode
        )
        with self._lock:
            self._permission_modes[conversation_id] = normalized_permission

        if create_if_missing:
            native_session_id = await self._get_or_create_session(
                conversation_id,
                title=title or "OpenCode Session",
            )
        else:
            with self._lock:
                native_session_id = self._sessions.get(conversation_id, "")
        if create_if_missing and native_session_id:
            ready_message = (
                f"OpenCode is live with {self.config.model_id} in "
                f"{self._workspace_root()}. The current chat is connected "
                f"to native session {native_session_id}."
            )
            self._persist_transcript_message(
                conversation_id,
                "assistant",
                ready_message,
                {
                    "runtime": "opencode",
                    "kind": "runtime_ready",
                    "deterministic": True,
                    "opencode_session_id": native_session_id,
                    "provider_id": self.config.provider_id,
                    "model_id": self.config.model_id,
                    "workspace": str(self._workspace_root()),
                },
            )
        else:
            ready_message = "OpenCode runtime prepared."
        self._log(
            "RUNTIME_PREPARED",
            odysseus_session_id=conversation_id,
            opencode_session_id=native_session_id,
            restarted=restarted,
            model_id=self.config.model_id,
            configured_workspace=str(self._workspace_root()),
        )
        return {
            **result,
            "ok": True,
            "ready": bool(create_if_missing and native_session_id),
            "restarted": restarted,
            "message": ready_message,
            "odysseus_session_id": conversation_id,
            "opencode_session_id": native_session_id,
            "permission_mode": normalized_permission,
            "model": {
                "endpoint_id": self.config.endpoint_id,
                "provider_id": self.config.provider_id,
                "model_id": self.config.model_id,
                "agent": self.config.agent,
            },
            "workspace": str(self._workspace_root()),
        }


    async def open_session_manager_terminal(
        self,
        odysseus_session_id: str,
        *,
        control_base_url: str = "",
    ) -> dict[str, Any]:
        self._assert_idle()
        conversation_id = str(odysseus_session_id or "").strip()
        if not conversation_id:
            raise OpenCodeServerLifecycleError(
                "Odysseus session id is required."
            )

        await self.health()
        with self._lock:
            native_session_id = self._sessions.get(conversation_id, "")
        if not native_session_id:
            raise OpenCodeServerLifecycleError(
                "Start or select an OpenCode session first."
            )

        control = None
        manager_kwargs: dict[str, Any] = {
            "odysseus_session_id": conversation_id,
        }
        if str(control_base_url or "").strip():
            control = self.issue_session_manager_control(conversation_id)
            manager_kwargs.update({
                "control_url": str(control_base_url).strip(),
                "control_token": control["token"],
            })

        try:
            result = await asyncio.to_thread(
                self.server_manager.open_session_manager_terminal,
                native_session_id,
                **manager_kwargs,
            )
        except Exception:
            if control is not None:
                self.release_session_manager_control(control["token"])
            raise

        self._log(
            "SESSION_MANAGER_TERMINAL_OPENED",
            odysseus_session_id=conversation_id,
            opencode_session_id=native_session_id,
            terminal=result.get("terminal"),
            endpoint=result.get("endpoint"),
            control_enabled=bool(control),
        )
        return {
            "ok": True,
            "odysseus_session_id": conversation_id,
            "opencode_session_id": native_session_id,
            "control_enabled": bool(control),
            "control_expires_at": (
                control["expires_at"] if control is not None else 0.0
            ),
            **result,
        }

    async def open_log_terminal(
        self,
        odysseus_session_id: str = "",
    ) -> dict[str, Any]:
        value = await asyncio.to_thread(
            self.server_manager.open_log_terminal,
            str(odysseus_session_id or "").strip(),
        )
        return {"ok": True, **value}

    async def cancel(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(str(run_id or ""))
        if run is None:
            self._log(
                "RUN_CANCEL_NOT_FOUND",
                run_id=str(run_id or ""),
            )
            return False
        self._log(
            "RUN_CANCEL_REQUESTED",
            run_id=run.run_id,
            odysseus_session_id=run.odysseus_session_id,
            opencode_session_id=run.opencode_session_id,
        )
        run.cancel_event.set()
        run.status = "cancelled"
        return await asyncio.to_thread(
            self.transport.abort_session,
            run.opencode_session_id,
        )

    async def forget_conversation(
        self,
        odysseus_session_id: str,
    ) -> bool:
        conversation_id = str(odysseus_session_id or "")
        with self._lock:
            session_id = self._sessions.pop(conversation_id, "")
            self._permission_modes.pop(conversation_id, None)
            self._permission_approvals.pop(conversation_id, None)
            stale_permission_ids = [
                request_id
                for request_id, pending in self._pending_permissions.items()
                if pending.odysseus_session_id == conversation_id
            ]
            for request_id in stale_permission_ids:
                self._pending_permissions.pop(request_id, None)
        if not session_id:
            self._log(
                "SESSION_FORGET_SKIPPED",
                odysseus_session_id=conversation_id,
                reason="no_mapping",
            )
            return False
        deleted = await asyncio.to_thread(
            self.transport.delete_session,
            session_id,
        )
        self._log(
            "SESSION_FORGOTTEN",
            odysseus_session_id=conversation_id,
            opencode_session_id=session_id,
            deleted=deleted,
        )
        return deleted

    async def _restore_persisted_session(
        self,
        conversation_id: str,
    ) -> str:
        resolver = self._session_resolver
        if resolver is None:
            return ""

        try:
            saved = resolver(conversation_id)
        except Exception as exc:
            self._log(
                "SESSION_RESTORE_SKIPPED",
                odysseus_session_id=conversation_id,
                reason="resolver_failed",
                error_type=type(exc).__name__,
            )
            return ""

        if not isinstance(saved, dict):
            return ""

        candidate = str(
            saved.get("opencode_session_id") or ""
        ).strip()
        if not candidate:
            return ""

        saved_workspace = str(
            saved.get("workspace") or ""
        ).strip()
        current_workspace = str(self._workspace_root())
        try:
            workspace_matches = (
                bool(saved_workspace)
                and os.path.normcase(
                    str(Path(saved_workspace).expanduser().resolve())
                )
                == os.path.normcase(current_workspace)
            )
        except (OSError, ValueError):
            workspace_matches = False

        if not workspace_matches:
            self._log(
                "SESSION_RESTORE_REJECTED",
                odysseus_session_id=conversation_id,
                opencode_session_id=candidate,
                reason="workspace_mismatch",
            )
            return ""

        saved_provider = str(
            saved.get("provider_id") or ""
        ).strip()
        saved_model = str(saved.get("model_id") or "").strip()

        if (
            saved_provider
            and self.config.provider_id
            and saved_provider != self.config.provider_id
        ):
            self._log(
                "SESSION_RESTORE_REJECTED",
                odysseus_session_id=conversation_id,
                opencode_session_id=candidate,
                reason="provider_mismatch",
            )
            return ""

        if (
            saved_model
            and self.config.model_id
            and saved_model != self.config.model_id
        ):
            self._log(
                "SESSION_RESTORE_REJECTED",
                odysseus_session_id=conversation_id,
                opencode_session_id=candidate,
                reason="model_mismatch",
            )
            return ""

        try:
            messages = await asyncio.to_thread(
                self.transport.list_messages,
                candidate,
            )
        except OpenCodeTransportError:
            self._log(
                "SESSION_RESTORE_REJECTED",
                odysseus_session_id=conversation_id,
                opencode_session_id=candidate,
                reason="native_session_unavailable",
            )
            return ""

        with self._lock:
            current = self._sessions.setdefault(
                conversation_id,
                candidate,
            )
        if current != candidate:
            return current

        self._log(
            "SESSION_RESTORED",
            odysseus_session_id=conversation_id,
            opencode_session_id=candidate,
            configured_workspace=current_workspace,
            message_count=len(messages),
        )
        return candidate

    async def _get_or_create_session(
        self,
        conversation_id: str,
        *,
        title: str,
    ) -> str:
        with self._lock:
            existing = self._sessions.get(conversation_id)
        if existing:
            self._log(
                "SESSION_REUSED",
                odysseus_session_id=conversation_id,
                opencode_session_id=existing,
                configured_workspace=(
                    self.config.workspace or str(Path.cwd())
                ),
            )
            return existing

        restored = await self._restore_persisted_session(
            conversation_id
        )
        if restored:
            return restored

        session_title = str(title or "").strip()
        if not session_title:
            session_title = f"Odysseus chat {conversation_id[:12]}"
        created = await asyncio.to_thread(
            self.transport.create_session,
            session_title,
        )
        self._log(
            "SESSION_CREATED",
            odysseus_session_id=conversation_id,
            opencode_session_id=created,
            title_length=len(session_title),
            configured_workspace=(
                self.config.workspace or str(Path.cwd())
            ),
        )
        with self._lock:
            current = self._sessions.setdefault(
                conversation_id,
                created,
            )
        if current != created:
            await asyncio.to_thread(
                self.transport.delete_session,
                created,
            )
        return current
