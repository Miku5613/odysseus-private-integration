from src.agent_runtime.tool_risk_mapper import (
    decide_tool_policy_action,
    summarize_tool_policy_actions,
)


def test_safe_read_tools_are_allowed() -> None:
    decision = decide_tool_policy_action("read_file")

    assert decision.risk.level == "safe_read"
    assert decision.action == "allow"
    assert "read-only" in decision.reason


def test_caution_tools_require_confirmation() -> None:
    for name in ["write_file", "send_email", "python", "mcp__abc__obsidian_write_note"]:
        decision = decide_tool_policy_action(name)

        assert decision.risk.level == "caution", name
        assert decision.action == "confirm", name


def test_unknown_tools_require_confirmation() -> None:
    decision = decide_tool_policy_action("custom_new_mcp_tool")

    assert decision.risk.level == "unknown"
    assert decision.action == "confirm"


def test_destructive_tools_are_proposal_only() -> None:
    for name in ["delete_email", "mcp__abc__obsidian_delete_note", "stop_served_model"]:
        decision = decide_tool_policy_action(name)

        assert decision.risk.level == "destructive", name
        assert decision.action == "propose_only", name


def test_schema_destructive_actions_are_proposal_only() -> None:
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "archive", "delete"],
            }
        },
    }

    decision = decide_tool_policy_action("bulk_email", schema=schema)

    assert decision.risk.level == "destructive"
    assert decision.action == "propose_only"


def test_summarize_tool_policy_actions_counts_actions() -> None:
    summary = summarize_tool_policy_actions(
        {
            "read_file": None,
            "grep": None,
            "write_file": None,
            "custom_tool": None,
            "delete_email": None,
            "bulk_email": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["mark_read", "delete"],
                    }
                },
            },
        }
    )

    assert summary == {
        "allow": 2,
        "confirm": 2,
        "propose_only": 2,
    }
