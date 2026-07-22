from __future__ import annotations

import json
from typing import Any

from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult
from src.agent_runtime.tool_risk_mapper import gate_tool_execution


WEB_SEARCH_CAPABILITY = "web_search"


async def run_web_search_worker(
    handoff: WorkerHandoff,
    *,
    session_id: str | None = None,
) -> WorkerResult:
    """Execute the shared web-search capability through the central tool gate."""

    if handoff.capability != WEB_SEARCH_CAPABILITY:
        raise ValueError("Web search worker requires capability=web_search")

    query = str(
        handoff.context.get("query")
        or handoff.context.get("topic")
        or handoff.objective
        or ""
    ).strip()
    if not query:
        return WorkerResult(
            capability=WEB_SEARCH_CAPABILITY,
            status="needs_clarification",
            summary="A web-search query is required.",
            missing_information=("web-search query",),
        )

    disabled_tools = set(handoff.context.get("disabled_tools") or ())
    if WEB_SEARCH_CAPABILITY in disabled_tools:
        return WorkerResult(
            capability=WEB_SEARCH_CAPABILITY,
            status="failed",
            summary="Web search is disabled for this turn.",
            error="web_search disabled by the request tool policy",
        )

    gate = gate_tool_execution(WEB_SEARCH_CAPABILITY)
    if not gate.allowed:
        return WorkerResult(
            capability=WEB_SEARCH_CAPABILITY,
            status="failed",
            summary="The central policy gate blocked web search.",
            data={"policy_action": gate.decision.action},
            error=gate.blocked_reason,
        )

    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    description, payload = await execute_tool_block(
        ToolBlock(
            WEB_SEARCH_CAPABILITY,
            json.dumps({"query": query}, ensure_ascii=False),
        ),
        session_id=session_id,
        disabled_tools=disabled_tools or None,
        owner=handoff.context.get("owner"),
        workspace=handoff.context.get("workspace"),
        tool_policy=handoff.context.get("tool_policy"),
    )

    result: dict[str, Any] = payload if isinstance(payload, dict) else {
        "output": str(payload or ""),
    }
    error = result.get("error")
    exit_code = result.get("exit_code", 0)
    if error or (isinstance(exit_code, int) and exit_code != 0):
        return WorkerResult(
            capability=WEB_SEARCH_CAPABILITY,
            status="failed",
            summary="Web search execution failed.",
            data={"description": description, "payload": result},
            error=str(error or result.get("output") or "unknown search failure"),
        )

    output = str(
        result.get("output")
        or result.get("results")
        or result.get("stdout")
        or "Web search completed."
    )
    return WorkerResult(
        capability=WEB_SEARCH_CAPABILITY,
        status="completed",
        summary=output,
        data={
            "description": description,
            "query": query,
            "payload": result,
        },
    )


__all__ = ["WEB_SEARCH_CAPABILITY", "run_web_search_worker"]
