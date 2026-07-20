from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingClarification:
    session_id: str
    original_request: str
    question: str


_LOCK = threading.RLock()
_PENDING: dict[str, PendingClarification] = {}


def register_pending_clarification(
    *,
    session_id: str,
    original_request: str,
    question: str,
) -> PendingClarification:
    session = str(session_id or '').strip()
    original = str(original_request or '').strip()
    prompt = str(question or '').strip()

    if not session:
        raise ValueError('session_id is required')
    if not original:
        raise ValueError('original_request is required')
    if not prompt:
        raise ValueError('question is required')

    record = PendingClarification(
        session_id=session,
        original_request=original,
        question=prompt,
    )

    with _LOCK:
        _PENDING[session] = record

    return record


def consume_pending_clarification(
    *,
    session_id: str,
    answer: str,
) -> str | None:
    session = str(session_id or '').strip()
    reply = str(answer or '').strip()

    if not session or not reply:
        return None

    with _LOCK:
        record = _PENDING.pop(session, None)

    if record is None:
        return None

    return (
        f'{record.original_request}\n\n'
        f'Clarification previously requested: {record.question}\n'
        f'User clarification: {reply}\n\n'
        'Re-evaluate the complete original request using this new '
        'clarification. Reconsider intent, available tools, target, '
        'source, and required arguments from scratch. If material '
        'ambiguity still remains, use ask_user again rather than guessing.'
    )


def get_pending_clarification(
    session_id: str,
) -> PendingClarification | None:
    session = str(session_id or '').strip()

    with _LOCK:
        return _PENDING.get(session)


def clear_session_clarification(session_id: str) -> None:
    session = str(session_id or '').strip()

    with _LOCK:
        _PENDING.pop(session, None)


def _clear_all_clarifications_for_tests() -> None:
    with _LOCK:
        _PENDING.clear()
