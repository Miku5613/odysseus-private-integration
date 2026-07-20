from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any


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

    def arguments(self) -> dict[str, Any]:
        value = json.loads(self.arguments_json)
        return value if isinstance(value, dict) else {}


_LOCK = threading.RLock()
_PENDING: dict[tuple[str, str], PendingToolConfirmation] = {}


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

    record = PendingToolConfirmation(
        version=CONFIRMATION_CONTRACT_VERSION,
        session_id=session,
        call_id=call,
        tool_name=tool,
        arguments_json=args_json,
        fingerprint=fingerprint,
        reason=str(reason or "").strip(),
        status="pending",
        created_at=time.time(),
    )

    key = (session, call)

    with _LOCK:
        existing = _PENDING.get(key)

        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise ValueError(
                    "call_id is already bound to a different tool call"
                )

            return existing

        _PENDING[key] = record
        return record


def get_pending_confirmation(
    session_id: str,
    call_id: str,
) -> PendingToolConfirmation | None:
    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")

    with _LOCK:
        return _PENDING.get((session, call))


def resolve_pending_confirmation(
    *,
    session_id: str,
    call_id: str,
    fingerprint: str,
    approved: bool,
) -> PendingToolConfirmation:
    """Approve or reject one exact pending call.

    Rejection consumes the pending record immediately.
    Approval does not execute anything; execution requires a separate exact,
    one-shot consume_approved_confirmation() call.
    """

    session = _required_text(session_id, "session_id")
    call = _required_text(call_id, "call_id")
    supplied_fingerprint = _required_text(fingerprint, "fingerprint")
    key = (session, call)

    with _LOCK:
        record = _PENDING.get(key)

        if record is None:
            raise KeyError("no pending confirmation for this session/call")

        if record.fingerprint != supplied_fingerprint:
            raise ValueError("confirmation fingerprint mismatch")

        if record.status != "pending":
            raise ValueError("confirmation has already been resolved")

        if not approved:
            rejected = replace(record, status="rejected")
            del _PENDING[key]
            return rejected

        approved_record = replace(record, status="approved")
        _PENDING[key] = approved_record
        return approved_record


def consume_approved_confirmation(
    *,
    session_id: str,
    call_id: str,
    fingerprint: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
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

    key = (session, call)

    with _LOCK:
        record = _PENDING.get(key)

        if record is None:
            raise KeyError("approved confirmation is missing or already consumed")

        if record.status != "approved":
            raise ValueError("confirmation has not been approved")

        if record.fingerprint != supplied_fingerprint:
            raise ValueError("confirmation fingerprint mismatch")

        del _PENDING[key]
        return replace(record, status="consumed")


def clear_session_confirmations(session_id: str) -> int:
    """Remove confirmation state belonging only to one chat session."""

    session = _required_text(session_id, "session_id")

    with _LOCK:
        keys = [key for key in _PENDING if key[0] == session]

        for key in keys:
            del _PENDING[key]

        return len(keys)


def _clear_all_confirmations_for_tests() -> None:
    with _LOCK:
        _PENDING.clear()


__all__ = [
    "CONFIRMATION_CONTRACT_VERSION",
    "PendingToolConfirmation",
    "build_tool_confirmation_fingerprint",
    "canonicalize_tool_arguments",
    "clear_session_confirmations",
    "consume_approved_confirmation",
    "get_pending_confirmation",
    "register_pending_confirmation",
    "resolve_pending_confirmation",
]