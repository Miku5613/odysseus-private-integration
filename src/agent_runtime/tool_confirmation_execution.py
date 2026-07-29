from __future__ import annotations

from typing import Any

from src.agent_runtime.tool_adapters import TOOL_MAP
from src.agent_runtime.tool_confirmation_store import (
    consume_approved_confirmation,
    get_confirmation,
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
    owner: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Resolve one exact session-bound caution-level call."""

    record = get_pending_confirmation(session_id, call_id)

    if record is None:
        durable = get_confirmation(session_id, call_id)
        if (
            durable is not None
            and durable.fingerprint == str(fingerprint or "").strip()
            and (owner is None or durable.owner == owner)
            and (run_id is None or durable.run_id == run_id)
            and durable.status in {"consumed", "rejected", "cancelled"}
        ):
            return {
                "status": (
                    "rejected"
                    if durable.status in {"rejected", "cancelled"}
                    else "already_processed"
                ),
                "executed": False,
                "idempotent_replay": True,
                "session_id": durable.session_id,
                "call_id": durable.call_id,
                "tool": durable.tool_name,
            }
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
            owner=owner,
            run_id=run_id,
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
        owner=owner,
        run_id=run_id,
    )

    consumed = consume_approved_confirmation(
        session_id=session_id,
        call_id=call_id,
        fingerprint=fingerprint,
        tool_name=record.tool_name,
        arguments=arguments,
        owner=owner,
        run_id=run_id,
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
