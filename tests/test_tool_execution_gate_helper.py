from src.agent_runtime.tool_risk_mapper import gate_tool_execution


def test_safe_read_tool_executes_immediately() -> None:
    gate = gate_tool_execution("read_file")

    assert gate.allowed is True
    assert gate.blocked_reason is None
    assert gate.decision.action == "allow"


def test_caution_tool_is_blocked_until_confirmation_handoff_exists() -> None:
    gate = gate_tool_execution("write_file")

    assert gate.allowed is False
    assert gate.decision.action == "confirm"
    assert gate.blocked_reason == "confirmation required but confirmation handoff is not wired"


def test_caution_tool_still_does_not_auto_execute_when_confirmation_available() -> None:
    gate = gate_tool_execution("send_email", confirmation_available=True)

    assert gate.allowed is False
    assert gate.decision.action == "confirm"
    assert gate.blocked_reason == "confirmation required"


def test_unknown_tool_fails_closed() -> None:
    gate = gate_tool_execution("brand_new_mcp_tool")

    assert gate.allowed is False
    assert gate.decision.risk.level == "unknown"
    assert gate.decision.action == "confirm"


def test_destructive_tool_is_proposal_only() -> None:
    gate = gate_tool_execution("mcp__abc__obsidian_delete_note")

    assert gate.allowed is False
    assert gate.decision.risk.level == "destructive"
    assert gate.decision.action == "propose_only"
    assert gate.blocked_reason == "direct execution blocked; produce a proposal instead"


def test_schema_destructive_action_is_proposal_only() -> None:
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "archive", "delete"],
            }
        },
    }

    gate = gate_tool_execution("bulk_email", schema=schema)

    assert gate.allowed is False
    assert gate.decision.action == "propose_only"
