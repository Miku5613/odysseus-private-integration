from src.agent_runtime.tool_risk_mapper import build_tool_gate_payload


def test_build_tool_gate_payload_splits_direct_and_gated_tools() -> None:
    payload = build_tool_gate_payload(
        [
            "read_file",
            "web_search",
            "write_file",
            "mcp__abc__obsidian_delete_note",
            "brand_new_mcp_tool",
        ]
    )

    assert [item["name"] for item in payload["direct"]] == [
        "read_file",
        "web_search",
    ]
    assert [item["name"] for item in payload["gated"]] == [
        "brand_new_mcp_tool",
        "mcp__abc__obsidian_delete_note",
        "write_file",
    ]


def test_build_tool_gate_payload_includes_policy_metadata() -> None:
    payload = build_tool_gate_payload(
        [
            "read_file",
            "write_file",
            "delete_email",
            "custom_tool",
        ]
    )

    direct = payload["direct"][0]
    gated = {item["name"]: item for item in payload["gated"]}

    assert direct == {
        "name": "read_file",
        "risk": "safe_read",
        "action": "allow",
        "reason": "safe read-only tools may run directly",
        "risk_reason": "known read-only tool name",
        "requires_confirmation": False,
        "proposal_only": False,
    }

    assert gated["write_file"]["action"] == "confirm"
    assert gated["write_file"]["requires_confirmation"] is True

    assert gated["delete_email"]["action"] == "propose_only"
    assert gated["delete_email"]["proposal_only"] is True

    assert gated["custom_tool"]["risk"] == "unknown"
    assert gated["custom_tool"]["action"] == "confirm"


def test_build_tool_gate_payload_counts_actions_and_groups() -> None:
    payload = build_tool_gate_payload(
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

    assert payload["counts"] == {
        "direct": 2,
        "gated": 4,
        "allow": 2,
        "confirm": 2,
        "propose_only": 2,
    }


def test_build_tool_gate_payload_is_json_safe() -> None:
    import json

    payload = build_tool_gate_payload(["read_file", "write_file"])

    encoded = json.dumps(payload, sort_keys=True)
    assert '"direct"' in encoded
    assert '"gated"' in encoded
    assert '"counts"' in encoded
