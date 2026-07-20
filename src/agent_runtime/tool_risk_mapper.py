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


ToolRiskPolicyAction = Literal["allow", "confirm", "propose_only"]


@dataclass(frozen=True)
class ToolRiskDecision:
    """Runtime-facing policy decision derived from tool risk."""

    risk: ToolRisk
    action: ToolRiskPolicyAction
    reason: str


def decide_tool_policy_action(
    tool_name: str,
    schema: Mapping[str, object] | None = None,
) -> ToolRiskDecision:
    """Convert a risk classification into a runtime policy action.

    This is still not wired into execution. It is the stable seam future runtime
    patches can call when deciding whether a tool may run directly, needs user
    confirmation, or should only produce a proposal/copy.
    """

    risk = classify_tool_risk(tool_name, schema=schema)

    if risk.level == "safe_read":
        return ToolRiskDecision(
            risk=risk,
            action="allow",
            reason="safe read-only tools may run directly",
        )

    if risk.level == "destructive" or risk.proposal_only:
        return ToolRiskDecision(
            risk=risk,
            action="propose_only",
            reason="destructive tools must not directly mutate user data",
        )

    return ToolRiskDecision(
        risk=risk,
        action="confirm",
        reason="mutating, external, or unknown tools require confirmation",
    )


def summarize_tool_policy_actions(
    tools: Mapping[str, Mapping[str, object] | None] | list[str] | tuple[str, ...],
) -> dict[ToolRiskPolicyAction, int]:
    """Return counts by policy action for logs/UI badges."""

    summary: dict[ToolRiskPolicyAction, int] = {
        "allow": 0,
        "confirm": 0,
        "propose_only": 0,
    }

    if isinstance(tools, Mapping):
        iterable = tools.items()
    else:
        iterable = ((name, None) for name in tools)

    for name, schema in iterable:
        decision = decide_tool_policy_action(str(name), schema=schema)
        summary[decision.action] += 1

    return summary


@dataclass(frozen=True)
class ToolExecutionGate:
    """Direct execution decision for a tool.

    This is the narrow runtime seam for future integration:
    - allowed=True only for safe_read tools
    - confirm/propose_only tools are blocked from direct execution until the
      confirmation/proposal layer handles them.
    """

    decision: ToolRiskDecision
    allowed: bool
    blocked_reason: str | None = None


READ_ACTIONS = frozenset({
    "get", "list", "read", "search", "find", "view", "show",
    "inspect", "status", "count", "check", "query",
})

DESTRUCTIVE_ACTIONS = frozenset({
    "delete", "remove", "trash", "truncate", "purge", "destroy",
})

CAUTION_ACTIONS = frozenset({
    "add", "append", "create", "edit", "update", "write",
    "send", "reply", "move", "rename", "archive", "mark_read",
    "mark_unread", "star", "unstar", "apply", "set",
})


def _extract_call_action(arguments: Mapping[str, object] | None) -> str | None:
    if not isinstance(arguments, Mapping):
        return None

    for key in ("action", "operation", "op", "command", "mode"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return None


def decide_tool_call_policy_action(
    tool_name: str,
    arguments: Mapping[str, object] | None = None,
    schema: Mapping[str, object] | None = None,
) -> ToolRiskDecision:
    """Classify the concrete tool call, including its selected action.

    Multi-action tools must not inherit the most dangerous action in their
    schema when the current call is explicitly read-only.
    """

    action = _extract_call_action(arguments)

    if action in DESTRUCTIVE_ACTIONS:
        risk = ToolRisk(
            name=tool_name,
            level="destructive",
            reason=f"tool call requests destructive action: {action}",
            requires_confirmation=True,
            proposal_only=True,
        )
        return ToolRiskDecision(
            risk=risk,
            action="propose_only",
            reason="destructive tool calls must not directly mutate user data",
        )

    if action in READ_ACTIONS:
        base = classify_tool_risk(tool_name, schema=None)

        # An explicit read action on a multi-action management tool is safe to
        # perform directly unless the tool name itself is intrinsically destructive.
        if base.level != "destructive":
            risk = ToolRisk(
                name=tool_name,
                level="safe_read",
                reason=f"tool call explicitly requests read-only action: {action}",
                requires_confirmation=False,
                proposal_only=False,
            )
            return ToolRiskDecision(
                risk=risk,
                action="allow",
                reason="explicit read-only tool action may run directly",
            )

    if action in CAUTION_ACTIONS:
        risk = ToolRisk(
            name=tool_name,
            level="caution",
            reason=f"tool call requests mutating action: {action}",
            requires_confirmation=True,
            proposal_only=False,
        )
        return ToolRiskDecision(
            risk=risk,
            action="confirm",
            reason="mutating tool call requires user confirmation",
        )

    return decide_tool_policy_action(tool_name, schema=schema)


def gate_tool_execution(
    tool_name: str,
    schema: Mapping[str, object] | None = None,
    *,
    confirmation_available: bool = False,
) -> ToolExecutionGate:
    """Return whether a tool may execute immediately.

    This function is conservative by design. Until a real confirmation UI/runtime
    handoff is wired in, caution and unknown tools are not auto-executed.
    """

    decision = decide_tool_policy_action(tool_name, schema=schema)

    if decision.action == "allow":
        return ToolExecutionGate(
            decision=decision,
            allowed=True,
            blocked_reason=None,
        )

    if decision.action == "confirm":
        suffix = "confirmation required"
        if not confirmation_available:
            suffix = "confirmation required but confirmation handoff is not wired"
        return ToolExecutionGate(
            decision=decision,
            allowed=False,
            blocked_reason=suffix,
        )

    return ToolExecutionGate(
        decision=decision,
        allowed=False,
        blocked_reason="direct execution blocked; produce a proposal instead",
    )


@dataclass(frozen=True)
class ToolGateSplit:
    """Direct/gated split for a tool surface.

    direct contains tools that may be shown/executed without confirmation.
    gated contains tools that require confirmation or proposal handling.
    """

    direct: tuple[ToolRiskDecision, ...]
    gated: tuple[ToolRiskDecision, ...]


def split_tools_for_direct_execution(
    tools: Mapping[str, Mapping[str, object] | None] | list[str] | tuple[str, ...],
) -> ToolGateSplit:
    """Split tools into direct-safe and gated groups.

    This is non-executing. It prepares the runtime/UI to pass only direct-safe
    tools to an automatic execution path while keeping mutating, destructive,
    and unknown tools behind a confirmation/proposal path.
    """

    if isinstance(tools, Mapping):
        iterable = tools.items()
    else:
        iterable = ((name, None) for name in tools)

    direct: list[ToolRiskDecision] = []
    gated: list[ToolRiskDecision] = []

    for name, schema in iterable:
        decision = decide_tool_policy_action(str(name), schema=schema)
        if decision.action == "allow":
            direct.append(decision)
        else:
            gated.append(decision)

    direct.sort(key=lambda item: item.risk.name.lower())
    gated.sort(key=lambda item: item.risk.name.lower())

    return ToolGateSplit(
        direct=tuple(direct),
        gated=tuple(gated),
    )


def _decision_payload(decision: ToolRiskDecision) -> dict[str, object]:
    """Serialize one tool policy decision for UI/runtime surfaces."""

    return {
        "name": decision.risk.name,
        "risk": decision.risk.level,
        "action": decision.action,
        "reason": decision.reason,
        "risk_reason": decision.risk.reason,
        "requires_confirmation": decision.risk.requires_confirmation,
        "proposal_only": decision.risk.proposal_only,
    }


def build_tool_gate_payload(
    tools: Mapping[str, Mapping[str, object] | None] | list[str] | tuple[str, ...],
) -> dict[str, object]:
    """Build a JSON-safe direct/gated tool payload.

    This does not execute or block anything by itself. It gives future Agent
    Tools UI/runtime patches one stable shape to display or consume.
    """

    split = split_tools_for_direct_execution(tools)
    policy_counts = summarize_tool_policy_actions(tools)

    return {
        "direct": [_decision_payload(item) for item in split.direct],
        "gated": [_decision_payload(item) for item in split.gated],
        "counts": {
            "direct": len(split.direct),
            "gated": len(split.gated),
            **policy_counts,
        },
    }


def group_tools_by_risk(
    tools: Mapping[str, Mapping[str, object] | None] | list[str] | tuple[str, ...],
) -> dict[ToolRiskLevel, list[ToolRisk]]:
    """Classify many tools and group them by risk level.

    Accepts either:
    - a mapping of tool_name -> schema
    - a list/tuple of tool names

    This is still non-behavioral. It only summarizes the tool surface so later
    UI/runtime patches can present safe/caution/destructive tools consistently.
    """

    grouped: dict[ToolRiskLevel, list[ToolRisk]] = {
        "safe_read": [],
        "caution": [],
        "destructive": [],
        "unknown": [],
    }

    if isinstance(tools, Mapping):
        iterable = tools.items()
    else:
        iterable = ((name, None) for name in tools)

    for name, schema in iterable:
        risk = classify_tool_risk(str(name), schema=schema)
        grouped[risk.level].append(risk)

    for risks in grouped.values():
        risks.sort(key=lambda item: item.name.lower())

    return grouped


def summarize_tool_risks(
    tools: Mapping[str, Mapping[str, object] | None] | list[str] | tuple[str, ...],
) -> dict[ToolRiskLevel, int]:
    """Return counts by risk level for logs/UI badges."""

    grouped = group_tools_by_risk(tools)
    return {level: len(items) for level, items in grouped.items()}


__all__ = [
    "ToolExecutionGate",
    "ToolGateSplit",
    "ToolRisk",
    "ToolRiskDecision",
    "ToolRiskLevel",
    "ToolRiskPolicyAction",
    "build_tool_gate_payload",
    "classify_tool_risk",
    "decide_tool_call_policy_action",
    "decide_tool_policy_action",
    "gate_tool_execution",
    "group_tools_by_risk",
    "split_tools_for_direct_execution",
    "summarize_tool_policy_actions",
    "summarize_tool_risks",
]
