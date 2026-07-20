"""Small name-based tool risk mapper.

This module is intentionally pure and non-behavioral. It only classifies tool
names so later runtime/UI patches can make consistent decisions about which
tools are safe to show directly, which need confirmation, and which should be
blocked or proposal-only.

Patch 4B does not wire this into runtime execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Literal


ToolRiskLevel = Literal["safe_read", "caution", "destructive", "unknown"]


@dataclass(frozen=True)
class ToolRisk:
    """Risk classification for a tool name."""

    name: str
    level: ToolRiskLevel
    reason: str
    requires_confirmation: bool
    proposal_only: bool = False


SAFE_READ_NAMES = frozenset(
    {
        "ask_user",
        "get_workspace",
        "glob",
        "grep",
        "list_cached_models",
        "list_cookbook_servers",
        "list_downloads",
        "list_email_accounts",
        "list_emails",
        "list_models",
        "list_sessions",
        "list_served_models",
        "list_serve_presets",
        "ls",
        "read_email",
        "read_file",
        "resolve_contact",
        "search_chats",
        "search_hf_models",
        "tail_serve_output",
        "web_fetch",
        "web_search",
    }
)

SAFE_READ_PREFIXES = (
    "list_",
    "read_",
    "search_",
    "get_",
)

SAFE_READ_SUBSTRINGS = (
    "_list_",
    "_read_",
    "_search_",
    "_get_",
    "obsidian_list_notes",
    "obsidian_get_note",
    "obsidian_open_in_ui",
)

DESTRUCTIVE_SUBSTRINGS = (
    "delete",
    "trash",
    "archive",
    "remove",
    "purge",
    "expunge",
    "truncate",
    "cancel_",
    "stop_served_model",
)

CAUTION_SUBSTRINGS = (
    "add",
    "adopt_",
    "api_call",
    "append",
    "bash",
    "bulk_email",
    "calendar",
    "create",
    "download_model",
    "edit",
    "email",
    "generate",
    "image",
    "manage",
    "memory",
    "note",
    "patch",
    "python",
    "reply",
    "send",
    "serve",
    "set",
    "skill",
    "task",
    "token",
    "toggle",
    "ui_control",
    "update",
    "webhook",
    "write",
)

# Exact names that are mutating but not destructive by themselves.
CAUTION_NAMES = frozenset(
    {
        "app_api",
        "chat_with_model",
        "create_document",
        "create_session",
        "download_model",
        "edit_document",
        "edit_file",
        "edit_image",
        "manage_atlas",
        "manage_bg_jobs",
        "manage_calendar",
        "manage_contact",
        "manage_documents",
        "manage_endpoints",
        "manage_mcp",
        "manage_memory",
        "manage_notes",
        "manage_settings",
        "manage_skills",
        "manage_tasks",
        "manage_tokens",
        "manage_webhooks",
        "pipeline",
        "python",
        "reply_to_email",
        "send_email",
        "send_to_session",
        "serve_model",
        "serve_preset",
        "suggest_document",
        "trigger_research",
        "ui_control",
        "update_document",
        "update_plan",
        "write_file",
    }
)


def _normalize_tool_name(tool_name: str) -> str:
    return str(tool_name or "").strip()


def _schema_action_values(schema: Mapping[str, object] | None) -> set[str]:
    """Best-effort extraction of action enum values from an OpenAI-style schema."""

    if not isinstance(schema, Mapping):
        return set()

    values: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            enum = node.get("enum")
            if isinstance(enum, list):
                for item in enum:
                    if isinstance(item, str):
                        values.add(item.lower())
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    return values


def classify_tool_risk(
    tool_name: str,
    schema: Mapping[str, object] | None = None,
) -> ToolRisk:
    """Classify a tool by stable name/schema hints.

    This is deliberately conservative:
    - read/list/search/get tools are safe_read
    - mutating/external tools are caution
    - delete/archive/trash/remove/truncate/etc. are destructive
    - unknown names remain unknown rather than pretending they are safe
    """

    name = _normalize_tool_name(tool_name)
    low = name.lower()

    if not low:
        return ToolRisk(
            name=name,
            level="unknown",
            reason="empty tool name",
            requires_confirmation=True,
        )

    action_values = _schema_action_values(schema)
    if action_values & {"delete", "remove", "trash", "archive", "truncate"}:
        return ToolRisk(
            name=name,
            level="destructive",
            reason="schema exposes destructive action values",
            requires_confirmation=True,
            proposal_only=True,
        )

    if low in SAFE_READ_NAMES:
        return ToolRisk(
            name=name,
            level="safe_read",
            reason="known read-only tool name",
            requires_confirmation=False,
        )

    if any(part in low for part in DESTRUCTIVE_SUBSTRINGS):
        return ToolRisk(
            name=name,
            level="destructive",
            reason="tool name contains destructive operation hint",
            requires_confirmation=True,
            proposal_only=True,
        )

    if any(part in low for part in SAFE_READ_SUBSTRINGS) or low.startswith(SAFE_READ_PREFIXES):
        return ToolRisk(
            name=name,
            level="safe_read",
            reason="tool name looks read-only",
            requires_confirmation=False,
        )

    if low in CAUTION_NAMES or any(part in low for part in CAUTION_SUBSTRINGS):
        return ToolRisk(
            name=name,
            level="caution",
            reason="tool name indicates mutation, external side effect, or broad capability",
            requires_confirmation=True,
        )

    return ToolRisk(
        name=name,
        level="unknown",
        reason="no stable risk hint matched",
        requires_confirmation=True,
    )


__all__ = [
    "ToolRisk",
    "ToolRiskLevel",
    "classify_tool_risk",
]
