from __future__ import annotations

from typing import Any

from src.agent_runtime.tool_adapters import TOOL_MAP
from src.agent_runtime.tool_confirmation_store import (
    consume_approved_confirmation,
    get_pending_confirmation,
    resolve_pending_confirmation,
)
from src.agent_runtime.tool_risk_mapper import (
    decide_tool_call_policy_action,
)


def resolve_and_execute_tool_confirmation(
    *,
    session_id: str,
    call_id: str,
    fingerprint: str,
    approved: bool,
) -> dict[str, Any]:
    """Resolve one exact session-bound caution-level call."""

    record = get_pending_confirmation(session_id, call_id)

    if record is None:
        raise KeyError(
            "no pending confirmation for this session/call"
        )

    if record.fingerprint != str(fingerprint or "").strip():
        raise ValueError("confirmation fingerprint mismatch")

    arguments = record.arguments()

    # Re-evaluate policy at approval time.
    decision = decide_tool_call_policy_action(
        record.tool_name,
        arguments,
    )

    if decision.action != "confirm":
        raise ValueError(
            "tool call is no longer eligible for confirmation"
        )

    if not approved:
        rejected = resolve_pending_confirmation(
            session_id=session_id,
            call_id=call_id,
            fingerprint=fingerprint,
            approved=False,
        )

        return {
            "status": "rejected",
            "executed": False,
            "session_id": rejected.session_id,
            "call_id": rejected.call_id,
            "tool": rejected.tool_name,
        }

    resolve_pending_confirmation(
        session_id=session_id,
        call_id=call_id,
        fingerprint=fingerprint,
        approved=True,
    )

    consumed = consume_approved_confirmation(
        session_id=session_id,
        call_id=call_id,
        fingerprint=fingerprint,
        tool_name=record.tool_name,
        arguments=arguments,
    )

    tool_obj = TOOL_MAP.get(consumed.tool_name)

    if tool_obj is None:
        raise ValueError(
            f"tool is no longer available: {consumed.tool_name}"
        )

    try:
        output = tool_obj.invoke(arguments)
        output_text = str(output)

        return {
            "status": "executed",
            "executed": True,
            "ok": not output_text.startswith("ERROR:"),
            "session_id": consumed.session_id,
            "call_id": consumed.call_id,
            "tool": consumed.tool_name,
            "output": output_text,
        }

    except Exception as exc:
        return {
            "status": "executed",
            "executed": True,
            "ok": False,
            "session_id": consumed.session_id,
            "call_id": consumed.call_id,
            "tool": consumed.tool_name,
            "output": (
                f"ERROR executing {consumed.tool_name}: {exc}"
            ),
        }


__all__ = ["resolve_and_execute_tool_confirmation"]
