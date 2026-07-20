from __future__ import annotations

from pathlib import Path

from src.agent_runtime.tool_risk_mapper import classify_tool_risk

ROOT = Path(__file__).resolve().parents[1]
TOOL_SCHEMAS_TEXT = (ROOT / "src" / "tool_schemas.py").read_text(
    encoding="utf-8",
    errors="replace",
)


def _assert_tool_schema_name_present(name: str) -> None:
    assert f'"name": "{name}"' in TOOL_SCHEMAS_TEXT or f"'name': '{name}'" in TOOL_SCHEMAS_TEXT, name


def test_real_read_only_tool_schema_names_classify_safe_read() -> None:
    for name in [
        "read_file",
        "grep",
        "glob",
        "ls",
        "get_workspace",
        "web_search",
        "web_fetch",
        "list_models",
        "list_sessions",
        "search_chats",
        "list_email_accounts",
        "list_emails",
        "read_email",
    ]:
        _assert_tool_schema_name_present(name)
        risk = classify_tool_risk(name)
        assert risk.level == "safe_read", (name, risk)
        assert risk.requires_confirmation is False
        assert risk.proposal_only is False


def test_real_broad_or_mutating_tool_schema_names_are_not_silently_safe() -> None:
    expected = {
        "write_file": "caution",
        "edit_file": "caution",
        "create_document": "caution",
        "edit_document": "caution",
        "update_document": "caution",
        "suggest_document": "caution",
        "python": "caution",
        "bash": "caution",
        "send_email": "caution",
        "reply_to_email": "caution",
        "ui_control": "caution",
        "delete_email": "destructive",
        "archive_email": "destructive",
        "bulk_email": "caution",
        "stop_served_model": "destructive",
        "cancel_download": "destructive",
    }

    for name, level in expected.items():
        _assert_tool_schema_name_present(name)
        risk = classify_tool_risk(name)
        assert risk.level == level, (name, risk)
        assert risk.requires_confirmation is True
        if level == "destructive":
            assert risk.proposal_only is True


def test_real_manage_tool_schemas_with_destructive_actions_fail_closed() -> None:
    for name in [
        "manage_calendar",
        "manage_notes",
        "manage_memory",
        "manage_tasks",
        "manage_session",
        "manage_documents",
        "manage_mcp",
        "manage_settings",
        "manage_tokens",
        "manage_webhooks",
    ]:
        _assert_tool_schema_name_present(name)
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "update", "delete"],
                }
            },
        }
        risk = classify_tool_risk(name, schema=schema)
        assert risk.level == "destructive", (name, risk)
        assert risk.requires_confirmation is True
        assert risk.proposal_only is True



def test_bulk_email_schema_with_delete_action_is_destructive() -> None:
    _assert_tool_schema_name_present("bulk_email")

    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["mark_read", "mark_unread", "archive", "delete", "junk"],
            }
        },
    }

    risk = classify_tool_risk("bulk_email", schema=schema)

    assert risk.level == "destructive"
    assert risk.requires_confirmation is True
    assert risk.proposal_only is True

def test_real_atlas_tool_schema_is_not_silently_safe() -> None:
    _assert_tool_schema_name_present("manage_atlas")

    risk = classify_tool_risk("manage_atlas")

    assert risk.level in {"caution", "destructive"}
    assert risk.requires_confirmation is True


def test_model_and_server_lifecycle_tools_are_not_silently_safe() -> None:
    for name in [
        "download_model",
        "serve_model",
        "serve_preset",
        "adopt_served_model",
        "stop_served_model",
        "cancel_download",
    ]:
        _assert_tool_schema_name_present(name)
        risk = classify_tool_risk(name)
        assert risk.level in {"caution", "destructive"}, (name, risk)
        assert risk.requires_confirmation is True


def test_mcp_obsidian_patterns_remain_stable() -> None:
    assert classify_tool_risk("mcp__abc123__obsidian_get_note").level == "safe_read"
    assert classify_tool_risk("mcp__abc123__obsidian_list_notes").level == "safe_read"
    assert classify_tool_risk("mcp__abc123__obsidian_write_note").level == "caution"
    assert classify_tool_risk("mcp__abc123__obsidian_patch_note").level == "caution"
    assert classify_tool_risk("mcp__abc123__obsidian_append_to_note").level == "caution"
    assert classify_tool_risk("mcp__abc123__obsidian_delete_note").level == "destructive"
