import pytest

from src.agent_runtime.mcp_worker import run_mcp_worker, select_mcp_tool
from src.agent_runtime.supervisor_contracts import WorkerHandoff


class FakeMcpManager:
    def __init__(self, tools, payload=None):
        self.tools = tools
        self.payload = payload or {"output": "ok", "exit_code": 0}
        self.calls = []

    def get_all_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.payload


def test_dynamic_selection_is_not_bound_to_one_server():
    tools = [
        {
            "server_id": "alpha",
            "server_name": "Alpha Service",
            "name": "lookup_records",
            "qualified_name": "mcp__alpha__lookup_records",
            "description": "Find records in Alpha.",
            "input_schema": {},
            "annotations": {"readOnlyHint": True},
            "is_disabled": False,
        },
        {
            "server_id": "beta",
            "server_name": "Beta Service",
            "name": "query_metrics",
            "qualified_name": "mcp__beta__query_metrics",
            "description": "Query operational metrics.",
            "input_schema": {},
            "annotations": {"readOnlyHint": True},
            "is_disabled": False,
        },
    ]

    handoff = WorkerHandoff(
        capability="mcp_specialized",
        objective="Query Beta operational metrics.",
    )

    selected = select_mcp_tool(tools, handoff)

    assert selected["qualified_name"] == "mcp__beta__query_metrics"


@pytest.mark.asyncio
async def test_readonly_mcp_executes_without_confirmation():
    tools = [
        {
            "server_id": "notes",
            "server_name": "Notes",
            "name": "lookup_note",
            "qualified_name": "mcp__notes__lookup_note",
            "description": "Read a note.",
            "input_schema": {},
            "annotations": {"readOnlyHint": True},
            "is_disabled": False,
        }
    ]

    manager = FakeMcpManager(tools)
    handoff = WorkerHandoff(
        capability="mcp_specialized",
        objective="Look up my note.",
        context={
            "qualified_tool_name": "mcp__notes__lookup_note",
            "arguments": {"title": "Roadmap"},
        },
    )

    result = await run_mcp_worker(
        handoff,
        manager_provider=lambda: manager,
    )

    assert result.status == "completed"
    assert manager.calls == [
        ("mcp__notes__lookup_note", {"title": "Roadmap"})
    ]
    assert result.data["policy_action"] == "allow"


@pytest.mark.asyncio
async def test_unknown_mutating_mcp_requires_confirmation():
    tools = [
        {
            "server_id": "notes",
            "server_name": "Notes",
            "name": "change_note",
            "qualified_name": "mcp__notes__change_note",
            "description": "Change a note.",
            "input_schema": {},
            "annotations": {"readOnlyHint": False},
            "is_disabled": False,
        }
    ]

    manager = FakeMcpManager(tools)
    handoff = WorkerHandoff(
        capability="mcp_specialized",
        objective="Change my note.",
        context={
            "qualified_tool_name": "mcp__notes__change_note",
            "arguments": {"content": "Updated"},
        },
    )

    result = await run_mcp_worker(
        handoff,
        manager_provider=lambda: manager,
    )

    assert result.status == "needs_clarification"
    assert result.data["policy_action"] == "confirm"
    assert manager.calls == []


@pytest.mark.asyncio
async def test_destructive_hint_is_proposal_only():
    tools = [
        {
            "server_id": "files",
            "server_name": "Files",
            "name": "purge_item",
            "qualified_name": "mcp__files__purge_item",
            "description": "Permanently remove an item.",
            "input_schema": {},
            "annotations": {"destructiveHint": True},
            "is_disabled": False,
        }
    ]

    manager = FakeMcpManager(tools)
    handoff = WorkerHandoff(
        capability="mcp_specialized",
        objective="Purge this item.",
        context={
            "qualified_tool_name": "mcp__files__purge_item",
            "arguments": {"id": "abc"},
        },
    )

    result = await run_mcp_worker(
        handoff,
        manager_provider=lambda: manager,
    )

    assert result.status == "partial"
    assert result.data["policy_action"] == "propose_only"
    assert manager.calls == []


@pytest.mark.asyncio
async def test_disabled_or_missing_tool_does_not_execute():
    tools = [
        {
            "server_id": "alpha",
            "server_name": "Alpha",
            "name": "lookup_data",
            "qualified_name": "mcp__alpha__lookup_data",
            "description": "Lookup data.",
            "input_schema": {},
            "annotations": {"readOnlyHint": True},
            "is_disabled": True,
        }
    ]

    manager = FakeMcpManager(tools)
    handoff = WorkerHandoff(
        capability="mcp_specialized",
        objective="Lookup Alpha data.",
    )

    result = await run_mcp_worker(
        handoff,
        manager_provider=lambda: manager,
    )

    assert result.status == "needs_clarification"
    assert manager.calls == []
