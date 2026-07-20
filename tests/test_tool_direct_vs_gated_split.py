from src.agent_runtime.tool_risk_mapper import split_tools_for_direct_execution


def test_split_tools_for_direct_execution_separates_safe_and_gated_names() -> None:
    split = split_tools_for_direct_execution(
        [
            "write_file",
            "read_file",
            "delete_email",
            "custom_unknown_tool",
            "web_search",
        ]
    )

    assert [item.risk.name for item in split.direct] == ["read_file", "web_search"]
    assert [item.risk.name for item in split.gated] == [
        "custom_unknown_tool",
        "delete_email",
        "write_file",
    ]


def test_split_tools_for_direct_execution_preserves_policy_actions() -> None:
    split = split_tools_for_direct_execution(
        [
            "read_file",
            "write_file",
            "mcp__abc__obsidian_delete_note",
            "new_mcp_tool",
        ]
    )

    actions = {item.risk.name: item.action for item in [*split.direct, *split.gated]}

    assert actions == {
        "read_file": "allow",
        "write_file": "confirm",
        "mcp__abc__obsidian_delete_note": "propose_only",
        "new_mcp_tool": "confirm",
    }


def test_split_tools_for_direct_execution_accepts_schema_mapping() -> None:
    split = split_tools_for_direct_execution(
        {
            "read_file": None,
            "bulk_email": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["mark_read", "archive", "delete"],
                    }
                },
            },
            "send_email": None,
        }
    )

    assert [item.risk.name for item in split.direct] == ["read_file"]
    assert [(item.risk.name, item.action) for item in split.gated] == [
        ("bulk_email", "propose_only"),
        ("send_email", "confirm"),
    ]


def test_split_tools_for_direct_execution_is_sorted_for_stable_ui() -> None:
    split = split_tools_for_direct_execution(
        [
            "web_search",
            "grep",
            "read_file",
            "write_file",
            "send_email",
        ]
    )

    assert [item.risk.name for item in split.direct] == [
        "grep",
        "read_file",
        "web_search",
    ]
    assert [item.risk.name for item in split.gated] == [
        "send_email",
        "write_file",
    ]
