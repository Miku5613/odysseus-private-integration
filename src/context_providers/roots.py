"""Portable authorized-root metadata.

This module owns the config representation for local context roots. It is not
wired into the runtime yet.

Important boundary:
- tracked files may define examples/defaults
- real local paths belong in untracked local config or environment variables
- raw resolved paths should not be shown to the model
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_ALLOWED_PROVIDER_TYPES = frozenset(
    {
        "local_dir",
        "obsidian_vault",
        "atlas_session",
        "mcp_integration",
    }
)

_ALLOWED_CAPABILITIES = frozenset(
    {
        "context.list",
        "context.read",
        "context.search",
        "context.status",
        "notes.write_session",
        "index.search",
        "provider.status",
    }
)

_ALLOWED_SAFETY_SCOPES = frozenset(
    {
        "read_only_sandbox",
        "approval_gated_write",
        "trusted_transient",
    }
)

_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,127}$")


class RootConfigError(ValueError):
    """Raised when authorized-root config is malformed."""


@dataclass(frozen=True)
class AuthorizedRoot:
    """Machine-portable description of a bounded context source.

    `local_path` is allowed only in untracked local config.

    `path_env` is useful for important host-level paths. For many roots, an
    untracked local config file is usually more practical than one env var per
    root.
    """

    id: str
    provider_type: str
    display_name: str
    allowed_capabilities: tuple[str, ...] = ("context.list", "context.read")
    safety_scope: str = "read_only_sandbox"
    enabled: bool = True

    path_env: str | None = None
    local_path: str | None = None
    connection_id: str | None = None

    last_verified_status: str | None = None
    associated_index_id: str | None = None
    owner_scope: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AuthorizedRoot":
        if not isinstance(raw, Mapping):
            raise RootConfigError("Authorized root entry must be an object.")

        root_id = str(raw.get("id", "")).strip()
        if not root_id or not _ROOT_ID_RE.match(root_id):
            raise RootConfigError(
                "Authorized root id must be 2-128 chars and contain only letters, "
                "numbers, dot, underscore, colon, or dash."
            )

        provider_type = str(raw.get("provider_type", "")).strip()
        if provider_type not in _ALLOWED_PROVIDER_TYPES:
            raise RootConfigError(f"Unsupported provider_type for root {root_id!r}: {provider_type!r}")

        display_name = str(raw.get("display_name", "")).strip()
        if not display_name:
            raise RootConfigError(f"Authorized root {root_id!r} is missing display_name.")

        capabilities_raw = raw.get("allowed_capabilities", ("context.list", "context.read"))
        if not isinstance(capabilities_raw, (list, tuple)):
            raise RootConfigError(f"allowed_capabilities for root {root_id!r} must be a list.")

        capabilities = tuple(sorted({str(c).strip() for c in capabilities_raw if str(c).strip()}))
        if not capabilities:
            raise RootConfigError(f"Authorized root {root_id!r} must have at least one capability.")

        invalid_caps = [c for c in capabilities if c not in _ALLOWED_CAPABILITIES]
        if invalid_caps:
            raise RootConfigError(
                f"Authorized root {root_id!r} has unsupported capabilities: {invalid_caps}"
            )

        safety_scope = str(raw.get("safety_scope", "read_only_sandbox")).strip()
        if safety_scope not in _ALLOWED_SAFETY_SCOPES:
            raise RootConfigError(f"Unsupported safety_scope for root {root_id!r}: {safety_scope!r}")

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise RootConfigError(f"metadata for root {root_id!r} must be an object.")

        path_env = _optional_str(raw.get("path_env"))
        local_path = _optional_str(raw.get("local_path"))
        connection_id = _optional_str(raw.get("connection_id"))

        if provider_type in {"local_dir", "obsidian_vault"} and not (path_env or local_path):
            raise RootConfigError(
                f"Authorized root {root_id!r} requires path_env or local_path."
            )

        if provider_type == "mcp_integration" and not connection_id:
            raise RootConfigError(
                f"Authorized root {root_id!r} with provider_type=mcp_integration requires connection_id."
            )

        return cls(
            id=root_id,
            provider_type=provider_type,
            display_name=display_name,
            allowed_capabilities=capabilities,
            safety_scope=safety_scope,
            enabled=bool(raw.get("enabled", True)),
            path_env=path_env,
            local_path=local_path,
            connection_id=connection_id,
            last_verified_status=_optional_str(raw.get("last_verified_status")),
            associated_index_id=_optional_str(raw.get("associated_index_id")),
            owner_scope=_optional_str(raw.get("owner_scope")),
            metadata=dict(metadata),
        )

    def has_capability(self, capability: str) -> bool:
        return capability in self.allowed_capabilities

    def resolve_path(self, env: Mapping[str, str] | None = None) -> Path | None:
        """Resolve machine-local path without exposing it to the model.

        Returns None for non-filesystem providers or missing env variables.
        """

        env_map = env if env is not None else os.environ

        raw_path: str | None = None
        if self.path_env:
            raw_path = env_map.get(self.path_env)
        elif self.local_path:
            raw_path = self.local_path

        if not raw_path:
            return None

        return Path(os.path.expandvars(os.path.expanduser(raw_path))).resolve()

    def public_summary(self) -> dict[str, Any]:
        """Return model/UI-safe root metadata without raw local filesystem paths."""

        return {
            "id": self.id,
            "provider_type": self.provider_type,
            "display_name": self.display_name,
            "allowed_capabilities": list(self.allowed_capabilities),
            "safety_scope": self.safety_scope,
            "enabled": self.enabled,
            "last_verified_status": self.last_verified_status,
            "associated_index_id": self.associated_index_id,
            "owner_scope": self.owner_scope,
            "has_path_env": bool(self.path_env),
            "has_local_path": bool(self.local_path),
            "has_connection_id": bool(self.connection_id),
        }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_authorized_roots(
    config_path: str | Path = "config/local_roots.json",
) -> list[AuthorizedRoot]:
    """Load authorized roots from local config.

    Missing config is valid and returns an empty list. This lets a fresh clone
    start cleanly before the user configures roots.
    """

    path = Path(config_path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RootConfigError(f"Invalid JSON in authorized roots config {path}: {exc}") from exc

    if isinstance(data, Mapping):
        entries = data.get("roots", [])
    else:
        entries = data

    if not isinstance(entries, list):
        raise RootConfigError("Authorized roots config must be a list or an object with a roots list.")

    roots = [AuthorizedRoot.from_dict(entry) for entry in entries]

    seen: set[str] = set()
    for root in roots:
        if root.id in seen:
            raise RootConfigError(f"Duplicate authorized root id: {root.id}")
        seen.add(root.id)

    return roots


def verify_authorized_root(
    root: AuthorizedRoot,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify root availability without mutating config."""

    resolved = root.resolve_path(env=env)
    if root.provider_type in {"local_dir", "obsidian_vault", "atlas_session"}:
        if resolved is None:
            status = "missing_path"
        elif not resolved.exists():
            status = "path_not_found"
        elif not resolved.is_dir():
            status = "not_a_directory"
        else:
            status = "ok"
    elif root.provider_type == "mcp_integration":
        status = "configured" if root.connection_id else "missing_connection"
    else:
        status = "unsupported_provider"

    return {
        "id": root.id,
        "provider_type": root.provider_type,
        "enabled": root.enabled,
        "status": status,
        "has_resolved_path": resolved is not None,
    }


__all__ = [
    "AuthorizedRoot",
    "RootConfigError",
    "load_authorized_roots",
    "verify_authorized_root",
]
