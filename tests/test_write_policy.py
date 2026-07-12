from __future__ import annotations

from src.agent_runtime.write_policy import (
    WRITE_POLICY_VERSION,
    WritePolicyInput,
    build_write_policy_result,
    filename_has_ai_marker,
    suggested_ai_copy_filename,
)


def test_original_note_modify_is_never_allowed() -> None:
    result = build_write_policy_result(WritePolicyInput(
        action="modify_original",
        target="original_note",
        explicit_permission=True,
        source_path="AIOS/Plan.md",
    ))

    assert result["version"] == WRITE_POLICY_VERSION
    assert result["decision"] == "block"
    assert result["allowed"] is False
    assert result["may_modify_original"] is False
    assert result["deterministic_final_refusal_allowed"] is False
    assert result["must_use_llm_final_response"] is True


def test_original_document_append_is_never_allowed_even_with_permission() -> None:
    result = build_write_policy_result({
        "action": "append_original",
        "target": "original_document",
        "explicit_permission": True,
        "source_path": "Docs/Report.md",
    })

    assert result["decision"] == "block"
    assert result["allowed"] is False
    assert result["may_modify_original"] is False
    assert "never allowed" in result["reason"]


def test_session_note_write_allowed_by_default() -> None:
    result = build_write_policy_result(WritePolicyInput(
        action="write_session_note",
        target="session_note",
    ))

    assert result["decision"] == "allow"
    assert result["allowed"] is True
    assert result["may_write_session_note"] is True
    assert result["may_modify_original"] is False


def test_propose_changes_allowed_without_write_permission() -> None:
    result = build_write_policy_result(WritePolicyInput(
        action="propose_changes",
        target="chat_only",
        source_path="AIOS/Plan.md",
    ))

    assert result["decision"] == "allow"
    assert result["allowed"] is True
    assert result["may_write_session_note"] is False
    assert result["may_create_ai_marked_copy"] is False


def test_ai_marked_copy_requires_explicit_permission() -> None:
    result = build_write_policy_result(WritePolicyInput(
        action="create_ai_marked_copy",
        target="ai_marked_copy",
        explicit_permission=False,
        source_path="AIOS/Plan.md",
        proposed_filename="Plan__Odysseus_AI_Copy.md",
    ))

    assert result["decision"] == "needs_explicit_permission"
    assert result["allowed"] is False
    assert result["requires_explicit_permission"] is True


def test_ai_marked_copy_requires_clear_filename_marker() -> None:
    result = build_write_policy_result(WritePolicyInput(
        action="create_ai_marked_copy",
        target="ai_marked_copy",
        explicit_permission=True,
        source_path="AIOS/Plan.md",
        proposed_filename="Plan_Rewrite.md",
    ))

    assert result["decision"] == "block"
    assert result["allowed"] is False
    assert result["may_create_ai_marked_copy"] is False
    assert "Odysseus_AI_Copy" in result["suggested_filename"]


def test_ai_marked_copy_allowed_with_permission_and_marker() -> None:
    result = build_write_policy_result(WritePolicyInput(
        action="create_ai_marked_copy",
        target="ai_marked_copy",
        explicit_permission=True,
        source_path="AIOS/Plan.md",
        proposed_filename="Plan__CreatedByOdysseus.md",
    ))

    assert result["decision"] == "allow"
    assert result["allowed"] is True
    assert result["may_create_ai_marked_copy"] is True
    assert result["may_modify_original"] is False


def test_filename_marker_helper_accepts_odysseus_and_ai_markers() -> None:
    assert filename_has_ai_marker("Plan__Odysseus_AI_Copy.md")
    assert filename_has_ai_marker("Plan__CreatedByAI.md")
    assert not filename_has_ai_marker("Plan_Rewrite.md")


def test_suggested_ai_copy_filename_preserves_extension() -> None:
    assert suggested_ai_copy_filename("AIOS/Plan.md", date_stamp="20260713") == "Plan__Odysseus_AI_Copy__20260713.md"
    assert suggested_ai_copy_filename("Untitled", date_stamp="20260713") == "Untitled__Odysseus_AI_Copy__20260713.md"


def test_unknown_write_action_blocks_by_default() -> None:
    result = build_write_policy_result({
        "action": "unknown",
        "target": "unknown",
    })

    assert result["decision"] == "block"
    assert result["allowed"] is False
    assert result["must_use_llm_final_response"] is True
