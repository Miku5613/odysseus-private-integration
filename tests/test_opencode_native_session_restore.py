from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_route_resolver_uses_latest_opencode_metadata():
    from core.models import (
        ChatMessage,
        get_session_manager_instance,
        set_session_manager_instance,
    )
    from routes import opencode_runtime_routes as routes

    session = SimpleNamespace(
        id="ody-1",
        history=[
            ChatMessage(
                "assistant",
                "older",
                metadata={
                    "runtime": "opencode",
                    "opencode_session_id": "ses-old",
                    "workspace": r"D:\old",
                },
            ),
            ChatMessage(
                "assistant",
                "newer",
                metadata={
                    "runtime": "opencode",
                    "opencode_session_id": "ses-current",
                    "provider_id": "provider-1",
                    "model_id": "model-1",
                    "workspace": r"D:\AI\odysseus",
                },
            ),
        ],
    )

    class FakeManager:
        def get_session(self, session_id):
            assert session_id == "ody-1"
            return session

    previous = get_session_manager_instance()
    set_session_manager_instance(FakeManager())
    try:
        restored = routes._resolve_persisted_native_session("ody-1")
    finally:
        set_session_manager_instance(previous)

    assert restored["opencode_session_id"] == "ses-current"
    assert restored["provider_id"] == "provider-1"
    assert restored["model_id"] == "model-1"
    assert restored["workspace"] == r"D:\AI\odysseus"


@pytest.mark.asyncio
async def test_valid_native_session_is_restored_without_creation(tmp_path):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService

    calls = []
    logs = []

    class FakeTransport:
        def list_messages(self, session_id):
            calls.append(("list", session_id))
            return [{"info": {"id": "msg-1"}}]

        def create_session(self, title):
            calls.append(("create", title))
            return "ses-new"

    class FakeManager:
        log_path = tmp_path / "bridge.log"

        def log_event(self, event, **fields):
            logs.append((event, fields))

    workspace = str(tmp_path.resolve())
    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(
            workspace=workspace,
            provider_id="provider-1",
            model_id="model-1",
        ),
        transport=FakeTransport(),
        server_manager=FakeManager(),
        session_resolver=lambda _conversation_id: {
            "opencode_session_id": "ses-restored",
            "provider_id": "provider-1",
            "model_id": "model-1",
            "workspace": workspace,
        },
    )

    result = await service._get_or_create_session(
        "ody-1",
        title="Restored chat",
    )

    assert result == "ses-restored"
    assert calls == [("list", "ses-restored")]
    assert service._sessions["ody-1"] == "ses-restored"
    assert any(name == "SESSION_RESTORED" for name, _ in logs)


@pytest.mark.asyncio
async def test_workspace_mismatch_creates_fresh_session(tmp_path):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService

    calls = []

    class FakeTransport:
        def list_messages(self, session_id):
            calls.append(("list", session_id))
            return []

        def create_session(self, title):
            calls.append(("create", title))
            return "ses-new"

    class FakeManager:
        log_path = tmp_path / "bridge.log"

        def log_event(self, *_args, **_kwargs):
            pass

    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(
            workspace=str(tmp_path.resolve()),
            provider_id="provider-1",
            model_id="model-1",
        ),
        transport=FakeTransport(),
        server_manager=FakeManager(),
        session_resolver=lambda _conversation_id: {
            "opencode_session_id": "ses-stale",
            "provider_id": "provider-1",
            "model_id": "model-1",
            "workspace": str((tmp_path / "other").resolve()),
        },
    )

    result = await service._get_or_create_session(
        "ody-1",
        title="Fresh chat",
    )

    assert result == "ses-new"
    assert calls == [("create", "Fresh chat")]


@pytest.mark.asyncio
async def test_missing_native_session_creates_replacement(tmp_path):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService
    from src.opencode_runtime.transport import OpenCodeTransportError

    calls = []

    class FakeTransport:
        def list_messages(self, session_id):
            calls.append(("list", session_id))
            raise OpenCodeTransportError("missing")

        def create_session(self, title):
            calls.append(("create", title))
            return "ses-new"

    class FakeManager:
        log_path = tmp_path / "bridge.log"

        def log_event(self, *_args, **_kwargs):
            pass

    workspace = str(tmp_path.resolve())
    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(
            workspace=workspace,
            provider_id="provider-1",
            model_id="model-1",
        ),
        transport=FakeTransport(),
        server_manager=FakeManager(),
        session_resolver=lambda _conversation_id: {
            "opencode_session_id": "ses-missing",
            "provider_id": "provider-1",
            "model_id": "model-1",
            "workspace": workspace,
        },
    )

    result = await service._get_or_create_session(
        "ody-1",
        title="Replacement chat",
    )

    assert result == "ses-new"
    assert calls == [
        ("list", "ses-missing"),
        ("create", "Replacement chat"),
    ]


def test_stream_persists_workspace_for_future_restore():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src/opencode_runtime/service.py"
    ).read_text(encoding="utf-8")

    assert source.count(
        '"workspace": str(self._workspace_root())'
    ) >= 2
    assert "session_resolver: SessionResolver | None = None" in source
    assert "await self._restore_persisted_session(" in source
