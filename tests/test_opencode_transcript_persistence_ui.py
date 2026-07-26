from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_frontend_no_longer_owns_opencode_transcript_persistence():
    source = read("static/js/runtimeControl.js")

    assert "persistOpenCodeTranscriptMessage" not in source
    assert "/api/session/${encodeURIComponent(id)}/message" not in source
    assert "chatRenderer.addMessage('user', prompt" in source
    assert "fetch('/api/opencode/runtime/stream'" in source


def test_odysseus_route_injects_backend_persister_and_owner_guard():
    source = read("routes/opencode_runtime_routes.py")

    for marker in (
        "get_session_manager_instance",
        "effective_user",
        "_verify_opencode_session_owner(",
        "transcript_persister=_persist_transcript_message",
        'message.metadata or {}).get("_db_id")',
    ):
        assert marker in source


def test_route_persister_uses_existing_session_manager_path():
    from core.models import (
        get_session_manager_instance,
        set_session_manager_instance,
    )
    from routes import opencode_runtime_routes as routes

    class FakeSession:
        def __init__(self):
            self.id = "ody-1"
            self.owner = "owner"
            self.history = []
            self._history = self.history
            self.message_count = 0

    class FakeManager:
        def __init__(self):
            self.session = FakeSession()

        def get_session(self, session_id):
            if session_id != self.session.id:
                raise KeyError(session_id)
            return self.session

        def add_message(self, session_id, message):
            session = self.get_session(session_id)
            session.history.append(message)
            session._history = session.history
            session.message_count = len(session.history)
            message.metadata["_db_id"] = "db-message-1"

    previous = get_session_manager_instance()
    manager = FakeManager()
    set_session_manager_instance(manager)
    try:
        routes._persist_transcript_message(
            "ody-1",
            "user",
            "Inspect the project",
            {"runtime": "opencode"},
        )
    finally:
        set_session_manager_instance(previous)

    assert len(manager.session.history) == 1
    message = manager.session.history[0]
    assert message.role == "user"
    assert message.content == "Inspect the project"
    assert message.metadata["runtime"] == "opencode"
    assert message.metadata["_db_id"] == "db-message-1"


def test_route_persister_rolls_back_memory_when_database_commit_is_missing():
    from core.models import (
        get_session_manager_instance,
        set_session_manager_instance,
    )
    from routes import opencode_runtime_routes as routes

    class FakeSession:
        def __init__(self):
            self.id = "ody-1"
            self.owner = "owner"
            self.history = []
            self._history = self.history
            self.message_count = 0

    class FakeManager:
        def __init__(self):
            self.session = FakeSession()

        def get_session(self, session_id):
            return self.session

        def add_message(self, session_id, message):
            self.session.history.append(message)
            self.session._history = self.session.history
            self.session.message_count = len(self.session.history)

    previous = get_session_manager_instance()
    manager = FakeManager()
    set_session_manager_instance(manager)
    try:
        with pytest.raises(
            RuntimeError,
            match="did not commit",
        ):
            routes._persist_transcript_message(
                "ody-1",
                "assistant",
                "Final response",
                {"runtime": "opencode"},
            )
    finally:
        set_session_manager_instance(previous)

    assert manager.session.history == []
    assert manager.session.message_count == 0


def test_runtime_service_persister_callback_is_optional_and_content_safe(
    tmp_path,
):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService

    calls = []
    logs = []

    class FakeManager:
        log_path = tmp_path / "bridge.log"

        def log_event(self, event, **fields):
            logs.append((event, fields))

    def persist(session_id, role, content, metadata):
        calls.append((session_id, role, content, metadata))

    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(),
        transport=object(),
        server_manager=FakeManager(),
        transcript_persister=persist,
    )

    assert service._persist_transcript_message(
        "ody-1",
        "assistant",
        "private response body",
        {
            "run_id": "run-1",
            "opencode_session_id": "ses-1",
            "opencode_message_id": "msg-1",
        },
    ) is True

    assert len(calls) == 1
    assert calls[0][0:3] == (
        "ody-1",
        "assistant",
        "private response body",
    )
    serialized_logs = repr(logs)
    assert "private response body" not in serialized_logs
    assert "content_length" in serialized_logs
    assert "TRANSCRIPT_MESSAGE_PERSISTED" in serialized_logs

    standalone = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(),
        transport=object(),
        server_manager=FakeManager(),
    )
    assert standalone._persist_transcript_message(
        "ody-1",
        "user",
        "standalone message",
        {},
    ) is False


def test_runtime_service_sanitizes_persistence_failures(tmp_path):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService
    from src.opencode_runtime.transport import OpenCodeTransportError

    logs = []

    class FakeManager:
        log_path = tmp_path / "bridge.log"

        def log_event(self, event, **fields):
            logs.append((event, fields))

    def fail(*_args):
        raise RuntimeError("database password must stay hidden")

    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(),
        transport=object(),
        server_manager=FakeManager(),
        transcript_persister=fail,
    )

    with pytest.raises(
        OpenCodeTransportError,
        match="transcript persistence failed",
    ) as captured:
        service._persist_transcript_message(
            "ody-1",
            "assistant",
            "private response body",
            {"run_id": "run-1"},
        )

    assert "database password" not in str(captured.value)
    serialized_logs = repr(logs)
    assert "database password" not in serialized_logs
    assert "private response body" not in serialized_logs
    assert "TRANSCRIPT_MESSAGE_PERSIST_FAILED" in serialized_logs


def test_stream_orders_persistence_around_opencode_execution():
    source = read("src/opencode_runtime/service.py")

    user_marker = (
        "# Persist the user turn before OpenCode starts work."
    )
    assistant_marker = (
        "# Persist the final assistant turn before reporting completion."
    )

    assert source.index(user_marker) < source.index(
        "self.transport.submit_prompt",
        source.index(user_marker),
    )
    assert source.index(assistant_marker) < source.index(
        '"type": "runtime.completed"',
        source.index(assistant_marker),
    )
