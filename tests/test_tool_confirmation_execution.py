import pytest

import src.agent_runtime.tool_confirmation_execution as execution
from src.agent_runtime.tool_confirmation_store import (
    _clear_all_confirmations_for_tests,
    get_pending_confirmation,
    register_pending_confirmation,
)


class FakeTool:
    def __init__(self):
        self.calls = []

    def invoke(self, arguments):
        self.calls.append(dict(arguments))
        return "ok"


@pytest.fixture(autouse=True)
def clean_store():
    _clear_all_confirmations_for_tests()
    yield
    _clear_all_confirmations_for_tests()


def pending(tool_name="send_email"):
    return register_pending_confirmation(
        session_id="session-a",
        call_id="call-1",
        tool_name=tool_name,
        arguments={
            "to": "a@example.com",
            "body": "hello",
        },
        reason="confirmation required",
        policy_action="confirm",
    )


def test_reject_never_executes(monkeypatch):
    fake = FakeTool()
    monkeypatch.setitem(
        execution.TOOL_MAP,
        "send_email",
        fake,
    )

    record = pending()

    result = execution.resolve_and_execute_tool_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        approved=False,
    )

    assert result["executed"] is False
    assert fake.calls == []
    assert get_pending_confirmation(
        "session-a",
        "call-1",
    ) is None


def test_approve_executes_exact_call_once(monkeypatch):
    fake = FakeTool()
    monkeypatch.setitem(
        execution.TOOL_MAP,
        "send_email",
        fake,
    )

    record = pending()

    result = execution.resolve_and_execute_tool_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        approved=True,
    )

    assert result["executed"] is True
    assert fake.calls == [{
        "to": "a@example.com",
        "body": "hello",
    }]

    duplicate = execution.resolve_and_execute_tool_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        approved=True,
    )
    assert duplicate["status"] == "already_processed"
    assert duplicate["executed"] is False
    assert duplicate["idempotent_replay"] is True

    assert len(fake.calls) == 1


def test_wrong_fingerprint_never_executes(monkeypatch):
    fake = FakeTool()
    monkeypatch.setitem(
        execution.TOOL_MAP,
        "send_email",
        fake,
    )

    pending()

    with pytest.raises(ValueError):
        execution.resolve_and_execute_tool_confirmation(
            session_id="session-a",
            call_id="call-1",
            fingerprint="wrong",
            approved=True,
        )

    assert fake.calls == []
