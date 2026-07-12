from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


WRITE_POLICY_VERSION = "write_policy_v1"

WriteAction = Literal[
    "propose_changes",
    "write_session_note",
    "create_ai_marked_copy",
    "modify_original",
    "append_original",
    "delete_original",
    "rename_original",
    "move_original",
    "unknown",
]

WriteTarget = Literal[
    "session_note",
    "original_note",
    "original_document",
    "ai_marked_copy",
    "chat_only",
    "unknown",
]

Decision = Literal["allow", "block", "needs_explicit_permission"]


ORIGINAL_MUTATION_ACTIONS = {
    "modify_original",
    "append_original",
    "delete_original",
    "rename_original",
    "move_original",
}


AI_COPY_MARKERS = (
    "Odysseus",
    "AI_Copy",
    "AI_Generated",
    "CreatedByOdysseus",
    "Created_By_Odysseus",
    "CreatedByAI",
    "Created_By_AI",
)


@dataclass(frozen=True)
class WritePolicyInput:
    action: WriteAction
    target: WriteTarget = "unknown"
    explicit_permission: bool = False
    proposed_filename: str = ""
    source_path: str = ""
    reason: str = ""


def _norm(value: Any) -> str:
    return str(value or "").strip()


def filename_has_ai_marker(filename: str) -> bool:
    text = _norm(filename).lower()
    if not text:
        return False
    return any(marker.lower() in text for marker in AI_COPY_MARKERS)


def suggested_ai_copy_filename(source_path: str, *, date_stamp: str = "") -> str:
    """Return a conservative AI-marked copy name.

    This helper only names a copy. It does not write anything.
    """

    raw = _norm(source_path) or "Untitled.md"
    normalized = raw.replace("\\", "/")
    leaf = normalized.rsplit("/", 1)[-1] or "Untitled.md"

    if "." in leaf and not leaf.startswith("."):
        stem, ext = leaf.rsplit(".", 1)
        suffix = "." + ext
    else:
        stem, suffix = leaf, ".md"

    stamp = _norm(date_stamp)
    stamp_part = f"__{stamp}" if stamp else ""

    return f"{stem}__Odysseus_AI_Copy{stamp_part}{suffix}"


def build_write_policy_result(policy_input: WritePolicyInput | dict[str, Any]) -> dict[str, Any]:
    """Evaluate the runtime write policy.

    This function is deliberately deterministic and side-effect free.

    It decides what the runtime may do, but it does not produce the final
    user-facing prose. A later LLM bridge should turn blocked/allowed context
    into a natural response.
    """

    if isinstance(policy_input, dict):
        data = WritePolicyInput(
            action=policy_input.get("action", "unknown"),
            target=policy_input.get("target", "unknown"),
            explicit_permission=bool(policy_input.get("explicit_permission", False)),
            proposed_filename=_norm(policy_input.get("proposed_filename", "")),
            source_path=_norm(policy_input.get("source_path", "")),
            reason=_norm(policy_input.get("reason", "")),
        )
    else:
        data = policy_input

    action = data.action
    target = data.target

    result: dict[str, Any] = {
        "version": WRITE_POLICY_VERSION,
        "action": action,
        "target": target,
        "decision": "block",
        "allowed": False,
        "requires_explicit_permission": False,
        "may_write_session_note": False,
        "may_create_ai_marked_copy": False,
        "may_modify_original": False,
        "may_delete_original": False,
        "may_rename_original": False,
        "may_move_original": False,
        "must_use_llm_final_response": True,
        "deterministic_final_refusal_allowed": False,
        "reason": "",
        "safe_alternatives": [
            "write_session_note",
            "propose_changes_in_chat",
            "create_ai_marked_copy_with_explicit_permission",
        ],
        "required_filename_marker": "Odysseus_AI_Copy",
        "suggested_filename": "",
    }

    if action in ORIGINAL_MUTATION_ACTIONS or target in {"original_note", "original_document"} and action not in {"propose_changes"}:
        result.update({
            "decision": "block",
            "allowed": False,
            "reason": "Original note/document mutation is never allowed by the Write graph.",
        })
        return result

    if action == "write_session_note" or target == "session_note":
        result.update({
            "decision": "allow",
            "allowed": True,
            "may_write_session_note": True,
            "reason": "Session note writing is allowed by default.",
        })
        return result

    if action == "propose_changes" or target == "chat_only":
        result.update({
            "decision": "allow",
            "allowed": True,
            "reason": "Proposal-only output is allowed because it does not mutate source notes/documents.",
        })
        return result

    if action == "create_ai_marked_copy" or target == "ai_marked_copy":
        suggested = suggested_ai_copy_filename(data.source_path)
        filename = data.proposed_filename or suggested
        has_marker = filename_has_ai_marker(filename)

        result["requires_explicit_permission"] = True
        result["suggested_filename"] = filename if has_marker else suggested

        if not data.explicit_permission:
            result.update({
                "decision": "needs_explicit_permission",
                "allowed": False,
                "reason": "AI-marked copy creation requires explicit user permission.",
            })
            return result

        if not has_marker:
            result.update({
                "decision": "block",
                "allowed": False,
                "reason": "AI-created copies must have a filename clearly marked as Odysseus/AI-created.",
            })
            return result

        result.update({
            "decision": "allow",
            "allowed": True,
            "may_create_ai_marked_copy": True,
            "reason": "Explicitly permitted AI-marked copy creation is allowed.",
        })
        return result

    result.update({
        "decision": "block",
        "allowed": False,
        "reason": "Unknown write action; runtime blocks by default.",
    })
    return result


__all__ = [
    "WRITE_POLICY_VERSION",
    "WritePolicyInput",
    "filename_has_ai_marker",
    "suggested_ai_copy_filename",
    "build_write_policy_result",
]
