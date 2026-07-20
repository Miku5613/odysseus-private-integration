import pytest

from src.agent_runtime.tool_confirmation_store import (
    _clear_all_confirmations_for_tests,
    build_tool_confirmation_fingerprint,
    canonicalize_tool_arguments,
    clear_session_confirmations,
    consume_approved_confirmation,
    get_pending_confirmation,
    register_pending_confirmation,
    resolve_pending_confirmation,
)


@pytest.fixture(autouse=True)
def _clean_store():
    _clear_all_confirmations_for_tests()
    yield
    _clear_all_confirmations_for_tests()


def _register(**overrides):
    values = {
        "session_id": "session-a",
        "call_id": "call-1",
        "tool_name": "send_email",
        "arguments": {"to": "a@example.com", "body": "hello"},
        "reason": "external mutation requires confirmation",
        "policy_action": "confirm",
    }
    values.update(overrides)
    return register_pending_confirmation(**values)


def test_argument_canonicalization_is_order_independent():
    left = canonicalize_tool_arguments({"b": 2, "a": 1})
    right = canonicalize_tool_arguments({"a": 1, "b": 2})
    assert left == right


def test_fingerprint_is_bound_to_session_call_tool_and_args():
    base = build_tool_confirmation_fingerprint(
        session_id="s1",
        call_id="c1",
        tool_name="send_email",
        arguments={"to": "a@example.com"},
    )

    changed = build_tool_confirmation_fingerprint(
        session_id="s1",
        call_id="c1",
        tool_name="send_email",
        arguments={"to": "b@example.com"},
    )

    assert base != changed


@pytest.mark.parametrize("action", ["allow", "propose_only", "", "ALLOW"])
def test_only_confirm_policy_can_enter_store(action):
    with pytest.raises(ValueError):
        _register(policy_action=action)


def test_pending_record_keeps_immutable_argument_snapshot():
    args = {"to": "a@example.com", "meta": {"x": 1}}
    record = _register(arguments=args)

    args["to"] = "changed@example.com"
    args["meta"]["x"] = 99

    assert record.arguments() == {
        "meta": {"x": 1},
        "to": "a@example.com",
    }


def test_same_exact_registration_is_idempotent():
    first = _register()
    second = _register()
    assert first.fingerprint == second.fingerprint


def test_call_id_cannot_be_rebound_to_mutated_arguments():
    _register()

    with pytest.raises(ValueError):
        _register(arguments={"to": "evil@example.com", "body": "hello"})


def test_confirmation_is_session_isolated():
    record = _register()

    assert get_pending_confirmation("session-a", "call-1") == record
    assert get_pending_confirmation("session-b", "call-1") is None


def test_wrong_fingerprint_cannot_approve():
    _register()

    with pytest.raises(ValueError):
        resolve_pending_confirmation(
            session_id="session-a",
            call_id="call-1",
            fingerprint="wrong",
            approved=True,
        )


def test_rejection_consumes_pending_call_permanently():
    record = _register()

    rejected = resolve_pending_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        approved=False,
    )

    assert rejected.status == "rejected"
    assert get_pending_confirmation("session-a", "call-1") is None

    with pytest.raises(KeyError):
        resolve_pending_confirmation(
            session_id=record.session_id,
            call_id=record.call_id,
            fingerprint=record.fingerprint,
            approved=True,
        )


def test_approval_does_not_execute_or_consume_by_itself():
    record = _register()

    approved = resolve_pending_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        approved=True,
    )

    assert approved.status == "approved"
    assert get_pending_confirmation("session-a", "call-1").status == "approved"


def test_approved_call_must_match_exact_tool_and_arguments():
    record = _register()

    resolve_pending_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        approved=True,
    )

    with pytest.raises(ValueError):
        consume_approved_confirmation(
            session_id=record.session_id,
            call_id=record.call_id,
            fingerprint=record.fingerprint,
            tool_name=record.tool_name,
            arguments={"to": "changed@example.com", "body": "hello"},
        )

    assert get_pending_confirmation("session-a", "call-1").status == "approved"


def test_exact_approval_is_consumable_once_only():
    arguments = {"to": "a@example.com", "body": "hello"}
    record = _register(arguments=arguments)

    resolve_pending_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        approved=True,
    )

    consumed = consume_approved_confirmation(
        session_id=record.session_id,
        call_id=record.call_id,
        fingerprint=record.fingerprint,
        tool_name=record.tool_name,
        arguments=arguments,
    )

    assert consumed.status == "consumed"

    with pytest.raises(KeyError):
        consume_approved_confirmation(
            session_id=record.session_id,
            call_id=record.call_id,
            fingerprint=record.fingerprint,
            tool_name=record.tool_name,
            arguments=arguments,
        )


def test_clear_session_does_not_touch_other_sessions():
    _register(session_id="session-a", call_id="call-a")
    _register(session_id="session-b", call_id="call-b")

    assert clear_session_confirmations("session-a") == 1
    assert get_pending_confirmation("session-a", "call-a") is None
    assert get_pending_confirmation("session-b", "call-b") is not None