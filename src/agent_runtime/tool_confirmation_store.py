from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.agent_runtime.durable_checkpoint import get_default_checkpoint_store


CONFIRMATION_CONTRACT_VERSION = "tool_confirmation_v1"


@dataclass(frozen=True)
class PendingToolConfirmation:
    """Immutable identity for one exact caution-level tool call."""

    version: str
    session_id: str
    call_id: str
    tool_name: str
    arguments_json: str
    fingerprint: str
    reason: str
    status: str
    created_at: float
    record_id: str = ""
    run_id: str = ""
    owner: str = "anonymous"
    step_id: str | None = None
    attempt_id: str | None = None
    expires_at: float = 0.0
    consumed_at: float | None = None
    response: Any = None
    side_effect_risk: str = "guided"

    def arguments(self) -> dict[str, Any]:
        value = json.loads(self.arguments_json)
        return value if isinstance(value, dict) else {}


def _record_to_confirmation(record: dict[str, Any]) -> PendingToolConfirmation:
    metadata = dict(record.get("tool_metadata") or {})
    return PendingToolConfirmation(
        version=CONFIRMATION_CONTRACT_VERSION,
        session_id=str(record["session_id"]),
        call_id=str(metadata.get("call_id") or ""),
        tool_name=str(record.get("capability") or ""),
        arguments_json=canonicalize_tool_arguments(
            metadata.get("arguments")
            if isinstance(metadata.get("arguments"), dict)
            else {}
        ),
        fingerprint=str(record["fingerprint"]),
        reason=str(metadata.get("reason") or ""),
        status=str(record["state"]),
        created_at=float(record["created_at"]),
        record_id=str(record["record_id"]),
        run_id=str(record["run_id"]),
        owner=str(record["owner"]),
        step_id=record.get("step_id"),
        attempt_id=record.get("attempt_id"),
        expires_at=float(record["expires_at"]),
        consumed_at=record.get("consumed_at"),
        response=record.get("response"),
        side_effect_risk=str(record.get("side_effect_risk") or "guided"),
    )


def _decision_scope(
    *,
    session_id: str,
    run_id: str | None,
    owner: str | None,
) -> tuple[str, str]:
    store = get_default_checkpoint_store()
    if run_id:
        record = store.run_record(str(run_id))
    else:
        record = store.latest_run_record(
            session_id=session_id,
            owner=owner,
        )
    resolved_run = str(
        run_id
        or (record or {}).get("run_id")
        or f"single:{session_id}"
    )
    resolved_owner = str(
        owner
        or (record or {}).get("owner")
        or "anonymous"
    )
    return resolved_run, resolved_owner


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def canonicalize_tool_arguments(arguments: dict[str, Any] | None) -> str:
    """Return a deterministic immutable representation of tool arguments."""

    payload = arguments if isinstance(arguments, dict) else {}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def build_tool_confirmation_fingerprint(
    *,
    session_id: str,
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> str:
    """Bind approval to one session, call id, tool name, and exact arguments."""

    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")
    tool = _required_text(tool_name, "tool_name")
    args_json = canonicalize_tool_arguments(arguments)

    identity = json.dumps(
        {
            "version": CONFIRMATION_CONTRACT_VERSION,
            "session_id": session,
            "call_id": call,
            "tool_name": tool,
            "arguments_json": args_json,
            "policy_action": "confirm",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def register_pending_confirmation(
    *,
    session_id: str,
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
    reason: str = "",
    policy_action: str,
    run_id: str | None = None,
    owner: str | None = None,
    step_id: str | None = None,
    attempt_id: str | None = None,
    expires_in_seconds: float = 3600.0,
    side_effect_risk: str = "guided",
) -> PendingToolConfirmation:
    """Register one exact tool call that the risk mapper classified as confirm.

    allow/propose_only calls are rejected here so destructive actions never
    accidentally gain an approval path through this store.
    """

    if str(policy_action or "").strip() != "confirm":
        raise ValueError("only policy_action=confirm may enter confirmation flow")

    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")
    tool = _required_text(tool_name, "tool_name")
    args_json = canonicalize_tool_arguments(arguments)
    fingerprint = build_tool_confirmation_fingerprint(
        session_id=session,
        call_id=call,
        tool_name=tool,
        arguments=arguments,
    )

    resolved_run, resolved_owner = _decision_scope(
        session_id=session,
        run_id=run_id,
        owner=owner,
    )
    existing = get_default_checkpoint_store().decision_for_tool_call(
        session_id=session,
        call_id=call,
    )
    if existing is not None and existing["fingerprint"] != fingerprint:
        raise ValueError("call_id is already bound to a different tool call")
    record = get_default_checkpoint_store().create_decision(
        kind="approval",
        run_id=resolved_run,
        session_id=session,
        owner=resolved_owner,
        step_id=step_id,
        attempt_id=attempt_id,
        action_text=f"{tool}({args_json})",
        fingerprint=fingerprint,
        expires_in_seconds=expires_in_seconds,
        capability=tool,
        tool_metadata={
            "call_id": call,
            "arguments": json.loads(args_json),
            "reason": str(reason or "").strip(),
        },
        side_effect_risk=side_effect_risk,
    )
    return _record_to_confirmation(record)


def get_pending_confirmation(
    session_id: str,
    call_id: str,
) -> PendingToolConfirmation | None:
    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")

    record = get_default_checkpoint_store().decision_for_tool_call(
        session_id=session,
        call_id=call,
    )
    if record is None or record["state"] not in {"pending", "approved"}:
        return None
    return _record_to_confirmation(record)


def get_confirmation(
    session_id: str,
    call_id: str,
) -> PendingToolConfirmation | None:
    """Return a durable confirmation in any lifecycle state."""

    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")
    record = get_default_checkpoint_store().decision_for_tool_call(
        session_id=session,
        call_id=call,
    )
    return _record_to_confirmation(record) if record is not None else None


def resolve_pending_confirmation(
    *,
    session_id: str,
    call_id: str,
    fingerprint: str,
    approved: bool,
    owner: str | None = None,
    run_id: str | None = None,
) -> PendingToolConfirmation:
    """Approve or reject one exact pending call.

    Rejection consumes the pending record immediately.
    Approval does not execute anything; execution requires a separate exact,
    one-shot consume_approved_confirmation() call.
    """

    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")
    supplied_fingerprint = _required_text(fingerprint, "fingerprint")
    existing = get_default_checkpoint_store().decision_for_tool_call(
        session_id=session,
        call_id=call,
    )
    if existing is None:
        raise KeyError("no pending confirmation for this session/call")
    if existing["fingerprint"] != supplied_fingerprint:
        raise ValueError("confirmation fingerprint mismatch")
    resolved = get_default_checkpoint_store().resolve_decision(
        record_id=str(existing["record_id"]),
        owner=str(owner or existing["owner"]),
        session_id=session,
        run_id=str(run_id or existing["run_id"]),
        response={"approved": bool(approved)},
        state="approved" if approved else "rejected",
    )
    return _record_to_confirmation(resolved)


def consume_approved_confirmation(
    *,
    session_id: str,
    call_id: str,
    fingerprint: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
    owner: str | None = None,
    run_id: str | None = None,
) -> PendingToolConfirmation:
    """Consume approval exactly once after re-validating the exact call."""

    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")
    supplied_fingerprint = _required_text(fingerprint, "fingerprint")

    expected = build_tool_confirmation_fingerprint(
        session_id=session,
        call_id=call,
        tool_name=tool_name,
        arguments=arguments,
    )

    if expected != supplied_fingerprint:
        raise ValueError("tool call changed after approval")

    existing = get_default_checkpoint_store().decision_for_tool_call(
        session_id=session,
        call_id=call,
    )
    if existing is None:
        raise KeyError("approved confirmation is missing or already consumed")
    if existing["state"] == "consumed":
        return _record_to_confirmation(existing)
    if existing["state"] != "approved":
        raise ValueError("confirmation has not been approved")
    if existing["fingerprint"] != supplied_fingerprint:
        raise ValueError("confirmation fingerprint mismatch")
    consumed = get_default_checkpoint_store().resolve_decision(
        record_id=str(existing["record_id"]),
        owner=str(owner or existing["owner"]),
        session_id=session,
        run_id=str(run_id or existing["run_id"]),
        response={"approved": True, "consumed": True},
        state="consumed",
    )
    return _record_to_confirmation(consumed)


def clear_session_confirmations(session_id: str) -> int:
    """Remove confirmation state belonging only to one chat session."""

    session = _required_text(session_id, "session_id")

    return get_default_checkpoint_store().delete_session_decisions(
        session_id=session,
        kind="approval",
    )


def _clear_all_confirmations_for_tests() -> None:
    get_default_checkpoint_store().delete_session_decisions(kind="approval")


__all__ = [
    "CONFIRMATION_CONTRACT_VERSION",
    "PendingToolConfirmation",
    "build_tool_confirmation_fingerprint",
    "canonicalize_tool_arguments",
    "clear_session_confirmations",
    "consume_approved_confirmation",
    "get_confirmation",
    "get_pending_confirmation",
    "register_pending_confirmation",
    "resolve_pending_confirmation",
]
