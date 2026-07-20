from src.agent_runtime.tool_risk_mapper import (
    group_tools_by_risk,
    summarize_tool_risks,
)


def test_group_tools_by_risk_accepts_name_list() -> None:
    grouped = group_tools_by_risk(
        [
            "read_file",
            "web_search",
            "write_file",
            "mcp__abc__obsidian_delete_note",
            "totally_custom_tool",
        ]
    )

    assert [item.name for item in grouped["safe_read"]] == ["read_file", "web_search"]
    assert [item.name for item in grouped["caution"]] == ["write_file"]
    assert [item.name for item in grouped["destructive"]] == ["mcp__abc__obsidian_delete_note"]
    assert [item.name for item in grouped["unknown"]] == ["totally_custom_tool"]


def test_group_tools_by_risk_accepts_schema_mapping() -> None:
    grouped = group_tools_by_risk(
        {
            "manage_notes": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "update", "delete"],
                    }
                },
            },
            "read_file": None,
            "write_file": None,
        }
    )

    assert [item.name for item in grouped["safe_read"]] == ["read_file"]
    assert [item.name for item in grouped["caution"]] == ["write_file"]
    assert [item.name for item in grouped["destructive"]] == ["manage_notes"]
    assert grouped["unknown"] == []


def test_summarize_tool_risks_returns_counts() -> None:
    summary = summarize_tool_risks(
        [
            "read_file",
            "grep",
            "write_file",
            "delete_email",
            "custom_alpha",
            "custom_beta",
        ]
    )

    assert summary == {
        "safe_read": 2,
        "caution": 1,
        "destructive": 1,
        "unknown": 2,
    }


def test_grouped_risks_are_sorted_by_name() -> None:
    grouped = group_tools_by_risk(["web_search", "grep", "read_file"])

    assert [item.name for item in grouped["safe_read"]] == [
        "grep",
        "read_file",
        "web_search",
    ]
