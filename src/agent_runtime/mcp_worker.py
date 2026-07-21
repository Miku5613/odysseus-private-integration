from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult
from src.agent_runtime.tool_risk_mapper import (
    ToolRisk,
    ToolRiskDecision,
    decide_tool_call_policy_action,
)
from src.mcp_manager import mcp_tool_is_readonly
from src.tool_utils import get_mcp_manager


MCP_WORKER_CAPABILITY = "mcp_specialized"


class McpManagerLike(Protocol):
    def get_all_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(
        self,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...


ManagerProvider = Callable[[], McpManagerLike | None]


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 1
    }


def _requested_text(handoff: WorkerHandoff) -> str:
    values = (
        handoff.context.get("requested_capability"),
        handoff.context.get("target_service"),
        handoff.context.get("domain"),
        handoff.context.get("resource"),
        handoff.context.get("tool_name"),
        handoff.objective,
    )

    return " ".join(str(value) for value in values if value)


def _tool_score(tool: Mapping[str, Any], requested: str) -> int:
    requested_tokens = _tokens(requested)

    if not requested_tokens:
        return 0

    qualified = str(tool.get("qualified_name") or "").lower()
    name = str(tool.get("name") or "").lower()
    server = str(tool.get("server_name") or "").lower()
    description = str(tool.get("description") or "").lower()

    score = 0

    for token in requested_tokens:
        if token in name:
            score += 8
        if token in qualified:
            score += 6
        if token in server:
            score += 4
        if token in description:
            score += 2

    return score


def select_mcp_tool(
    tools: Sequence[Mapping[str, Any]],
    handoff: WorkerHandoff,
) -> dict[str, Any] | None:
    explicit = str(
        handoff.context.get("qualified_tool_name")
        or handoff.context.get("qualified_name")
        or ""
    ).strip()

    available = [
        dict(tool)
        for tool in tools
        if not bool(tool.get("is_disabled"))
    ]

    if explicit:
        for tool in available:
            if tool.get("qualified_name") == explicit:
                return tool

        return None

    requested = _requested_text(handoff)

    ranked = sorted(
        (
            (_tool_score(tool, requested), tool)
            for tool in available
        ),
        key=lambda item: (
            item[0],
            str(item[1].get("qualified_name") or ""),
        ),
        reverse=True,
    )

    if not ranked or ranked[0][0] <= 0:
        return None

    top_score = ranked[0][0]
    top = [tool for score, tool in ranked if score == top_score]

    if len(top) != 1:
        return None

    return dict(top[0])


def _annotation_value(
    annotations: object,
    key: str,
) -> object:
    if isinstance(annotations, Mapping):
        return annotations.get(key)

    if annotations is not None:
        return getattr(annotations, key, None)

    return None


def decide_mcp_policy(
    tool: Mapping[str, Any],
    arguments: Mapping[str, object] | None,
) -> ToolRiskDecision:
    qualified_name = str(tool.get("qualified_name") or "")
    raw_name = str(tool.get("name") or qualified_name)
    schema = tool.get("input_schema")

    decision = decide_tool_call_policy_action(
        raw_name,
        arguments,
        schema if isinstance(schema, Mapping) else None,
    )

    annotations = tool.get("annotations")
    destructive_hint = _annotation_value(
        annotations,
        "destructiveHint",
    )

    if destructive_hint is True:
        risk = ToolRisk(
            name=qualified_name,
            level="destructive",
            reason="MCP server marked this tool destructive",
            requires_confirmation=True,
            proposal_only=True,
        )

        return ToolRiskDecision(
            risk=risk,
            action="propose_only",
            reason="destructive MCP tools cannot directly mutate user data",
        )

    if decision.action != "confirm":
        return decision

    if mcp_tool_is_readonly(dict(tool)):
        risk = ToolRisk(
            name=qualified_name,
            level="safe_read",
            reason="MCP read-only annotation or read-only name heuristic",
            requires_confirmation=False,
            proposal_only=False,
        )

        return ToolRiskDecision(
            risk=risk,
            action="allow",
            reason="read-only MCP tools may run directly",
        )

    return decision


def _policy_data(
    decision: ToolRiskDecision,
) -> dict[str, object]:
    return {
        "policy_action": decision.action,
        "risk_level": decision.risk.level,
        "policy_reason": decision.reason,
        "risk_reason": decision.risk.reason,
        "requires_confirmation": decision.risk.requires_confirmation,
        "proposal_only": decision.risk.proposal_only,
    }


async def run_mcp_worker(
    handoff: WorkerHandoff,
    *,
    session_id: str | None = None,
    manager_provider: ManagerProvider = get_mcp_manager,
) -> WorkerResult:
    if handoff.capability != MCP_WORKER_CAPABILITY:
        raise ValueError(
            "MCP Worker requires capability=mcp_specialized"
        )

    manager = manager_provider()

    if manager is None:
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="failed",
            summary="No MCP manager is available.",
            error="MCP manager is not initialized.",
        )

    try:
        tools = manager.get_all_tools()
    except Exception as exc:
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="failed",
            summary="MCP capability discovery failed.",
            error=str(exc),
        )

    tool = select_mcp_tool(tools, handoff)

    if tool is None:
        available = sorted(
            str(item.get("qualified_name") or "")
            for item in tools
            if not bool(item.get("is_disabled"))
        )

        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="needs_clarification",
            summary=(
                "No single connected MCP tool matched the requested "
                "capability with enough confidence."
            ),
            missing_information=(
                "specific MCP capability or tool",
            ),
            data={
                "available_tools": available,
                "session_id": session_id,
            },
        )

    arguments = handoff.context.get("arguments") or {}

    if not isinstance(arguments, Mapping):
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="needs_clarification",
            summary="MCP tool arguments must be a JSON object.",
            missing_information=("valid MCP arguments",),
            data={
                "tool": tool.get("qualified_name"),
                "session_id": session_id,
            },
        )

    call_arguments = dict(arguments)
    owner = handoff.context.get("owner")

    if owner and "_odysseus_owner" not in call_arguments:
        call_arguments["_odysseus_owner"] = owner

    policy = decide_mcp_policy(tool, call_arguments)
    policy_data = _policy_data(policy)
    qualified_name = str(tool.get("qualified_name") or "")

    if policy.action == "confirm":
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="needs_clarification",
            summary=(
                f"Confirmation is required before running {qualified_name}."
            ),
            missing_information=("user confirmation",),
            data={
                "tool": qualified_name,
                "arguments": call_arguments,
                "session_id": session_id,
                **policy_data,
            },
        )

    if policy.action == "propose_only":
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="partial",
            summary=(
                f"{qualified_name} was not executed because the action "
                "is destructive. A proposal may be produced instead."
            ),
            data={
                "tool": qualified_name,
                "arguments": call_arguments,
                "session_id": session_id,
                **policy_data,
            },
        )

    if policy.action != "allow":
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="failed",
            summary="MCP policy returned an unsupported action.",
            data={
                "tool": qualified_name,
                **policy_data,
            },
            error=f"Unsupported policy action: {policy.action}",
        )

    try:
        payload = await manager.call_tool(
            qualified_name,
            call_arguments,
        )
    except Exception as exc:
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="failed",
            summary="MCP tool execution failed.",
            data={
                "tool": qualified_name,
                "session_id": session_id,
                **policy_data,
            },
            error=str(exc),
        )

    if not isinstance(payload, dict):
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="failed",
            summary="MCP tool returned an invalid payload.",
            data={
                "tool": qualified_name,
                **policy_data,
            },
            error="Expected a dictionary result.",
        )

    exit_code = payload.get("exit_code")

    if payload.get("error") or (
        isinstance(exit_code, int) and exit_code != 0
    ):
        return WorkerResult(
            capability=MCP_WORKER_CAPABILITY,
            status="failed",
            summary="MCP tool returned an error.",
            data={
                "tool": qualified_name,
                "payload": payload,
                "session_id": session_id,
                **policy_data,
            },
            error=str(
                payload.get("error")
                or payload.get("output")
                or "Unknown MCP failure."
            ),
        )

    summary = str(
        payload.get("output")
        or payload.get("response")
        or f"{qualified_name} completed."
    )

    return WorkerResult(
        capability=MCP_WORKER_CAPABILITY,
        status="completed",
        summary=summary,
        data={
            "tool": qualified_name,
            "payload": payload,
            "session_id": session_id,
            **policy_data,
        },
    )


__all__ = [
    "MCP_WORKER_CAPABILITY",
    "decide_mcp_policy",
    "run_mcp_worker",
    "select_mcp_tool",
]
