from src.agent_runtime.tool_risk_mapper import classify_tool_risk


def test_safe_read_builtin_tools_do_not_require_confirmation() -> None:
    for name in [
        "read_file",
        "grep",
        "glob",
        "ls",
        "get_workspace",
        "web_search",
        "web_fetch",
        "list_emails",
        "read_email",
    ]:
        risk = classify_tool_risk(name)
        assert risk.level == "safe_read", name
        assert risk.requires_confirmation is False
        assert risk.proposal_only is False


def test_caution_builtin_mutation_tools_require_confirmation() -> None:
    for name in [
        "write_file",
        "edit_file",
        "manage_calendar",
        "manage_notes",
        "manage_memory",
        "send_email",
        "reply_to_email",
        "python",
        "bash",
        "ui_control",
    ]:
        risk = classify_tool_risk(name)
        assert risk.level == "caution", name
        assert risk.requires_confirmation is True
        assert risk.proposal_only is False


def test_destructive_builtin_tools_are_proposal_only() -> None:
    for name in [
        "delete_email",
        "archive_email",
        "manage_session_delete",
        "cancel_download",
        "stop_served_model",
    ]:
        risk = classify_tool_risk(name)
        assert risk.level == "destructive", name
        assert risk.requires_confirmation is True
        assert risk.proposal_only is True


def test_obsidian_mcp_read_tools_are_safe_read() -> None:
    for name in [
        "mcp__55585c47__obsidian_get_note",
        "mcp__55585c47__obsidian_list_notes",
        "mcp__55585c47__obsidian_open_in_ui",
    ]:
        risk = classify_tool_risk(name)
        assert risk.level == "safe_read", name
        assert risk.requires_confirmation is False


def test_obsidian_mcp_write_tools_are_caution_or_destructive() -> None:
    caution = classify_tool_risk("mcp__55585c47__obsidian_write_note")
    patch = classify_tool_risk("mcp__55585c47__obsidian_patch_note")
    append = classify_tool_risk("mcp__55585c47__obsidian_append_to_note")
    delete = classify_tool_risk("mcp__55585c47__obsidian_delete_note")

    assert caution.level == "caution"
    assert patch.level == "caution"
    assert append.level == "caution"
    assert delete.level == "destructive"
    assert delete.proposal_only is True


def test_schema_destructive_actions_override_benign_name() -> None:
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "search", "delete"],
            }
        },
    }

    risk = classify_tool_risk("manage_notes", schema=schema)

    assert risk.level == "destructive"
    assert risk.requires_confirmation is True
    assert risk.proposal_only is True


def test_unknown_tools_fail_closed() -> None:
    risk = classify_tool_risk("totally_custom_tool")

    assert risk.level == "unknown"
    assert risk.requires_confirmation is True
    assert risk.proposal_only is False


def test_empty_tool_name_is_unknown() -> None:
    risk = classify_tool_risk("")

    assert risk.level == "unknown"
    assert risk.requires_confirmation is True
