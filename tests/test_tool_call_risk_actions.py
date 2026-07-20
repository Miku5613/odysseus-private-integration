from src.agent_runtime.tool_risk_mapper import decide_tool_call_policy_action


def test_manage_notes_list_is_safe_read() -> None:
    decision = decide_tool_call_policy_action(
        "manage_notes",
        {"action": "list"},
    )

    assert decision.risk.level == "safe_read"
    assert decision.action == "allow"


def test_manage_notes_add_requires_confirmation() -> None:
    decision = decide_tool_call_policy_action(
        "manage_notes",
        {"action": "add"},
    )

    assert decision.risk.level == "caution"
    assert decision.action == "confirm"


def test_manage_notes_delete_is_proposal_only() -> None:
    decision = decide_tool_call_policy_action(
        "manage_notes",
        {"action": "delete"},
    )

    assert decision.risk.level == "destructive"
    assert decision.action == "propose_only"
    assert decision.risk.proposal_only is True


def test_bulk_email_mark_read_is_not_misclassified_as_delete() -> None:
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["mark_read", "archive", "delete"],
            }
        },
    }

    decision = decide_tool_call_policy_action(
        "bulk_email",
        {"action": "mark_read"},
        schema=schema,
    )

    assert decision.risk.level == "caution"
    assert decision.action == "confirm"


def test_bulk_email_delete_is_proposal_only() -> None:
    decision = decide_tool_call_policy_action(
        "bulk_email",
        {"action": "delete"},
    )

    assert decision.action == "propose_only"


def test_manage_tasks_list_is_safe_even_if_schema_exposes_delete() -> None:
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "add", "update", "delete"],
            }
        },
    }

    decision = decide_tool_call_policy_action(
        "manage_tasks",
        {"action": "list"},
        schema=schema,
    )

    assert decision.risk.level == "safe_read"
    assert decision.action == "allow"


def test_intrinsically_destructive_name_remains_destructive() -> None:
    decision = decide_tool_call_policy_action(
        "delete_email",
        {"action": "read"},
    )

    assert decision.risk.level == "destructive"
    assert decision.action == "propose_only"


def test_normal_read_file_remains_allowed() -> None:
    decision = decide_tool_call_policy_action(
        "read_file",
        {"path": "notes/example.md"},
    )

    assert decision.action == "allow"


def test_unknown_tool_without_action_fails_closed() -> None:
    decision = decide_tool_call_policy_action(
        "brand_new_external_tool",
        {"payload": "x"},
    )

    assert decision.risk.level == "unknown"
    assert decision.action == "confirm"


def test_operation_alias_is_understood() -> None:
    decision = decide_tool_call_policy_action(
        "manage_documents",
        {"operation": "list"},
    )

    assert decision.action == "allow"
