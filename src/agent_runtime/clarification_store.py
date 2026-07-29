from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.agent_runtime.durable_checkpoint import get_default_checkpoint_store


@dataclass(frozen=True)
class PendingClarification:
    session_id: str
    original_request: str
    question: str
    record_id: str = ""
    run_id: str = ""
    owner: str = "anonymous"
    step_id: str | None = None
    attempt_id: str | None = None
    fingerprint: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0
    state: str = "pending"
    consumed_at: float | None = None
    response: Any = None


def _required(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _fingerprint(
    *,
    session_id: str,
    original_request: str,
    question: str,
) -> str:
    payload = json.dumps(
        {
            "version": "clarification_v2",
            "session_id": session_id,
            "original_request": original_request,
            "question": question,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_record(value: dict[str, Any]) -> PendingClarification:
    metadata = dict(value.get("tool_metadata") or {})
    return PendingClarification(
        session_id=str(value["session_id"]),
        original_request=str(metadata.get("original_request") or ""),
        question=str(value.get("action_text") or ""),
        record_id=str(value["record_id"]),
        run_id=str(value["run_id"]),
        owner=str(value["owner"]),
        step_id=value.get("step_id"),
        attempt_id=value.get("attempt_id"),
        fingerprint=str(value["fingerprint"]),
        created_at=float(value["created_at"]),
        expires_at=float(value["expires_at"]),
        state=str(value["state"]),
        consumed_at=value.get("consumed_at"),
        response=value.get("response"),
    )


def register_pending_clarification(
    *,
    session_id: str,
    original_request: str,
    question: str,
    run_id: str | None = None,
    owner: str | None = None,
    step_id: str | None = None,
    attempt_id: str | None = None,
    expires_in_seconds: float = 86400.0,
) -> PendingClarification:
    session = _required(session_id, "session_id")
    original = _required(original_request, "original_request")
    prompt = _required(question, "question")
    store = get_default_checkpoint_store()
    run = store.run_record(str(run_id)) if run_id else store.latest_run_record(
        session_id=session,
        owner=owner,
    )
    resolved_run = str(run_id or (run or {}).get("run_id") or f"single:{session}")
    resolved_owner = str(owner or (run or {}).get("owner") or "anonymous")
    fingerprint = _fingerprint(
        session_id=session,
        original_request=original,
        question=prompt,
    )
    value = store.create_decision(
        kind="clarification",
        run_id=resolved_run,
        session_id=session,
        owner=resolved_owner,
        step_id=step_id,
        attempt_id=attempt_id,
        action_text=prompt,
        fingerprint=fingerprint,
        expires_in_seconds=expires_in_seconds,
        tool_metadata={"original_request": original},
        side_effect_risk="none",
    )
    return _to_record(value)


def consume_pending_clarification(
    *,
    session_id: str,
    answer: str,
    owner: str | None = None,
    run_id: str | None = None,
) -> str | None:
    session = str(session_id or "").strip()
    reply = str(answer or "").strip()
    if not session or not reply:
        return None
    store = get_default_checkpoint_store()
    value = store.find_decision(
        kind="clarification",
        session_id=session,
        owner=owner,
        run_id=run_id,
    )
    if value is None or value["state"] != "pending":
        return None
    resolved = store.resolve_decision(
        record_id=str(value["record_id"]),
        owner=str(owner or value["owner"]),
        session_id=session,
        run_id=str(run_id or value["run_id"]),
        response={"answer": reply},
        state="answered",
    )
    record = _to_record(resolved)
    return (
        f"{record.original_request}\n\n"
        f"Clarification previously requested: {record.question}\n"
        f"User clarification: {reply}\n\n"
        "Re-evaluate the complete original request using this new "
        "clarification. Reconsider intent, available tools, target, "
        "source, and required arguments from scratch. If material "
        "ambiguity still remains, use ask_user again rather than guessing."
    )


def get_pending_clarification(
    session_id: str,
) -> PendingClarification | None:
    session = str(session_id or "").strip()
    if not session:
        return None
    value = get_default_checkpoint_store().find_decision(
        kind="clarification",
        session_id=session,
    )
    if value is None or value["state"] != "pending":
        return None
    return _to_record(value)


def clear_session_clarification(session_id: str) -> None:
    session = str(session_id or "").strip()
    if session:
        get_default_checkpoint_store().delete_session_decisions(
            session_id=session,
            kind="clarification",
        )


def _clear_all_clarifications_for_tests() -> None:
    get_default_checkpoint_store().delete_session_decisions(
        kind="clarification",
    )
