from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


class FakeServerManager:
    owns_process = True

    def __init__(self, root: Path):
        self.log_path = root / "opencode-test.log"
        self.log_root = root
        self.opened = []

    def ensure_running(self):
        return {
            "mode": "managed",
            "owned": True,
            "pid": 101,
            "health": {"healthy": True},
        }

    def open_session_manager_terminal(self, session_id, **kwargs):
        self.opened.append((session_id, kwargs))
        return {"terminal": "test", "endpoint": "local"}


class FakeTransport:
    def __init__(self, config):
        self.config = config
        self.sessions = [
            {
                "id": "ses-old",
                "title": "Older work",
                "time": {"created": 1000, "updated": 2000},
            },
            {
                "id": "ses-active",
                "title": "Active work",
                "time": {"created": 3000, "updated": 4000},
            },
        ]
        self.created = []
        self.renamed = []
        self.deleted = []
        self.list_directories = []

    def list_sessions(self, directory=""):
        self.list_directories.append(directory)
        return list(self.sessions)

    def get_session(self, session_id):
        return next(
            (item for item in self.sessions if item.get("id") == session_id),
            None,
        )

    def create_session(self, title):
        self.created.append(title)
        session_id = f"ses-new-{len(self.created)}"
        self.sessions.append({"id": session_id, "title": title})
        return session_id

    def update_session_title(self, session_id, title):
        self.renamed.append((session_id, title))
        return {"id": session_id, "title": title}

    def delete_session(self, session_id):
        self.deleted.append(session_id)
        before = len(self.sessions)
        self.sessions = [item for item in self.sessions if item.get("id") != session_id]
        return len(self.sessions) != before


def make_service(tmp_path):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService

    config = OpenCodeRuntimeConfig(
        workspace=str(tmp_path.resolve()),
        endpoint_id="endpoint-a",
        provider_id="loaded-provider",
        provider_name="Local provider",
        provider_base_url="http://127.0.0.1:11434/v1",
        provider_models=("model-a",),
        model_id="model-a",
    )
    transport = FakeTransport(config)
    manager = FakeServerManager(tmp_path)
    transcript = []
    service = OpenCodeRuntimeService(
        config=config,
        transport=transport,
        server_manager=manager,
        transcript_persister=lambda *args: transcript.append(args),
    )
    return service, transport, manager, transcript


def test_native_session_routes_are_exposed():
    source = (ROOT / "routes" / "opencode_runtime_routes.py").read_text(
        encoding="utf-8"
    )
    assert '@router.get("/native-sessions")' in source
    assert '@router.post("/native-sessions")' in source
    assert '@router.post("/native-sessions/activate")' in source
    assert '@router.delete("/native-sessions/{opencode_session_id}")' in source
    assert '@router.patch("/native-sessions/{opencode_session_id}")' in source
    assert "create_if_missing=False" in source


def test_create_route_prepares_without_autocreate_then_creates(monkeypatch):
    from routes import opencode_runtime_routes as routes

    calls = []

    class Service:
        async def prepare_runtime(self, **kwargs):
            calls.append(("prepare", kwargs))
            return {"ready": False}

        async def create_conversation_session(self, conversation_id, *, title):
            calls.append(("create", {"conversation_id": conversation_id, "title": title}))
            return {"ok": True, "opencode_session_id": "ses-created"}

    selection = SimpleNamespace(
        endpoint_id="local-ollama",
        provider_id="ollama",
        provider_name="Ollama",
        base_url="http://127.0.0.1:11434/v1",
        models=("model-a",),
        api_key="",
        model_id="model-a",
    )
    monkeypatch.setattr(routes, "_service", Service())
    monkeypatch.setattr(routes, "_verify_opencode_session_owner", lambda *_: None)
    monkeypatch.setattr(routes, "resolve_model_selection", lambda **_: selection)
    request = routes.OpenCodeNativeSessionCreateRequest(
        odysseus_session_id="ody-1",
        title="Named session",
        endpoint_id="local-ollama",
        model_id="model-a",
        workspace=str(ROOT),
    )

    result = asyncio.run(routes.create_opencode_native_session(request, object()))

    assert result["opencode_session_id"] == "ses-created"
    assert [name for name, _ in calls] == ["prepare", "create"]
    assert calls[0][1]["create_if_missing"] is False


def test_activate_route_prepares_selected_provider_before_attach(monkeypatch):
    from routes import opencode_runtime_routes as routes

    calls = []

    class Service:
        async def prepare_runtime(self, **kwargs):
            calls.append(("prepare", kwargs))

        async def attach_conversation_session(self, conversation_id, native_id):
            calls.append(("attach", (conversation_id, native_id)))
            return {"ok": True, "opencode_session_id": native_id}

    selection = SimpleNamespace(
        endpoint_id="endpoint-b",
        provider_id="registry-provider",
        provider_name="Same local provider",
        base_url="http://127.0.0.1:11434/v1",
        models=("model-a", "model-b"),
        api_key="",
        model_id="model-b",
    )
    monkeypatch.setattr(routes, "_service", Service())
    monkeypatch.setattr(routes, "_verify_opencode_session_owner", lambda *_: None)
    monkeypatch.setattr(routes, "resolve_model_selection", lambda **_: selection)
    request = routes.OpenCodeNativeSessionRequest(
        odysseus_session_id="ody-1",
        opencode_session_id="ses-old",
        endpoint_id="endpoint-b",
        model_id="model-b",
        workspace=str(ROOT),
    )

    result = asyncio.run(routes.activate_opencode_native_session(request, object()))

    assert result["opencode_session_id"] == "ses-old"
    assert [name for name, _ in calls] == ["prepare", "attach"]
    assert calls[0][1]["create_if_missing"] is False
    assert calls[0][1]["provider_base_url"] == selection.base_url


def test_stream_route_preserves_explicit_provider_mismatch_detail(monkeypatch):
    from fastapi import HTTPException
    from routes import opencode_runtime_routes as routes

    class Service:
        def validate_turn_selection(self, **kwargs):
            raise RuntimeError(
                "The selected model belongs to a different OpenCode provider."
            )

    selection = SimpleNamespace(
        endpoint_id="endpoint-b",
        provider_id="other-provider",
        base_url="http://127.0.0.1:9999/v1",
        models=("model-a",),
        model_id="model-a",
    )
    monkeypatch.setattr(routes, "_service", Service())
    monkeypatch.setattr(routes, "_verify_opencode_session_owner", lambda *_: None)
    monkeypatch.setattr(routes, "resolve_model_selection", lambda **_: selection)
    request = routes.OpenCodeRuntimeRequest(
        message="hello",
        odysseus_session_id="ody-1",
        endpoint_id="endpoint-b",
        model_id="model-a",
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.stream_opencode_runtime(object(), request))

    assert error.value.status_code == 409
    assert error.value.detail == (
        "The selected model belongs to a different OpenCode provider."
    )


def test_list_shape_marks_only_the_current_session_active(tmp_path):
    service, transport, _, _ = make_service(tmp_path)
    service._sessions["ody-1"] = "ses-active"

    result = asyncio.run(
        service.list_native_sessions("ody-1", workspace="F:/project")
    )

    assert result["active_opencode_session_id"] == "ses-active"
    assert [item["id"] for item in result["sessions"]] == [
        "ses-active",
        "ses-old",
    ]
    assert sum(item["active"] for item in result["sessions"]) == 1
    assert transport.list_directories == ["F:/project"]
    assert result["sessions"][0]["updated_at"] == "1970-01-01T01:06:40Z"


def test_timestamp_normalization_sorts_seconds_milliseconds_and_iso(tmp_path):
    service, transport, _, _ = make_service(tmp_path)
    transport.sessions = [
        {"id": "seconds", "title": "Seconds", "time": {"updated": 1_785_696_825}},
        {"id": "milliseconds", "title": "Milliseconds", "time": {"updated": 1_785_696_826_000}},
        {"id": "iso", "name": "ISO", "updatedAt": "2026-08-02T18:53:47Z"},
        {"id": "missing-title", "time": {}},
    ]

    result = asyncio.run(service.list_native_sessions("ody-1", workspace="F:/project"))

    assert [item["id"] for item in result["sessions"]] == [
        "iso",
        "milliseconds",
        "seconds",
        "missing-title",
    ]
    assert len(result["sessions"]) == len(transport.sessions)
    assert result["sessions"][-1]["title"] == "OpenCode Session"
    by_id = {item["id"]: item for item in result["sessions"]}
    assert by_id["seconds"]["updated_at"] == "2026-08-02T18:53:45Z"
    assert by_id["milliseconds"]["updated_at"] == "2026-08-02T18:53:46Z"
    assert by_id["iso"]["updated_at"] == "2026-08-02T18:53:47Z"


def test_create_requires_title_and_preserves_previous_session(tmp_path):
    service, transport, _, transcript = make_service(tmp_path)
    service._sessions["ody-1"] = "ses-old"

    with pytest.raises(ValueError, match="title is required"):
        asyncio.run(service.create_conversation_session("ody-1", title="   "))

    result = asyncio.run(
        service.create_conversation_session("ody-1", title="New work")
    )

    assert result["message"] == 'OpenCode session created: "New work".'
    assert result["previous_opencode_session_id"] == "ses-old"
    assert service._sessions["ody-1"] == "ses-new-1"
    assert transport.deleted == []
    assert transcript[-1][3]["deterministic"] is True


def test_start_and_switch_messages_are_deterministic(tmp_path):
    service, _, _, transcript = make_service(tmp_path)

    started = asyncio.run(
        service.attach_conversation_session("ody-1", "ses-old")
    )
    switched = asyncio.run(
        service.attach_conversation_session("ody-1", "ses-active")
    )

    assert started["message"] == 'OpenCode session started: "Older work".'
    assert switched["message"] == 'OpenCode session switched to "Active work".'
    assert service._sessions["ody-1"] == "ses-active"
    assert all(item[3]["deterministic"] is True for item in transcript)


def test_rename_updates_the_native_opencode_title(tmp_path):
    service, transport, _, _ = make_service(tmp_path)

    result = asyncio.run(
        service.rename_native_session("ody-1", "ses-old", "Renamed work")
    )

    assert result["title"] == "Renamed work"
    assert transport.renamed == [("ses-old", "Renamed work")]


def test_same_endpoint_model_change_keeps_active_session(tmp_path):
    service, transport, _, _ = make_service(tmp_path)
    service._sessions["ody-1"] = "ses-active"

    result = asyncio.run(
        service.prepare_runtime(
            odysseus_session_id="ody-1",
            title="Ignored",
            permission_mode="ask",
            endpoint_id="endpoint-b",
            provider_id="registry-provider",
            provider_name="Same local provider",
            provider_base_url="http://127.0.0.1:11434/v1/",
            provider_models=("model-a", "model-b"),
            model_id="model-b",
            workspace=str(tmp_path.resolve()),
            create_if_missing=False,
        )
    )

    assert result["opencode_session_id"] == "ses-active"
    assert service._sessions["ody-1"] == "ses-active"
    assert service.config.model_id == "model-b"
    assert service.config.provider_id == "loaded-provider"
    assert set(service.config.provider_models) == {"model-a", "model-b"}
    assert transport.created == []
    assert transport.deleted == []


def test_turn_selection_accepts_same_provider_url_across_endpoint_records(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    service._sessions["ody-1"] = "ses-active"

    provider_id = service.validate_turn_selection(
        endpoint_id="endpoint-b",
        provider_id="registry-provider",
        provider_base_url="http://127.0.0.1:11434/v1/",
        provider_models=("model-a", "model-b"),
        model_id="model-b",
    )

    assert provider_id == "loaded-provider"
    assert service._sessions["ody-1"] == "ses-active"


def test_turn_selection_rejects_incompatible_provider_with_explicit_error(tmp_path):
    service, _, _, _ = make_service(tmp_path)

    with pytest.raises(
        Exception,
        match="selected model belongs to a different OpenCode provider",
    ):
        service.validate_turn_selection(
            endpoint_id="endpoint-b",
            provider_id="other-provider",
            provider_base_url="http://127.0.0.1:9999/v1",
            provider_models=("model-a",),
            model_id="model-a",
        )


def test_delete_inactive_session_keeps_active_mapping(tmp_path):
    service, transport, _, transcript = make_service(tmp_path)
    service._sessions["ody-1"] = "ses-active"

    result = asyncio.run(
        service.delete_native_session("ody-1", "ses-old", title="Older work")
    )

    assert result["deleted"] is True
    assert result["active_cleared"] is False
    assert service._sessions["ody-1"] == "ses-active"
    assert transport.deleted == ["ses-old"]
    assert result["message"] == 'OpenCode session deleted: "Older work".'
    assert transcript[-1][3]["deterministic"] is True


def test_delete_active_session_clears_mapping_and_handles_missing(tmp_path):
    service, transport, _, _ = make_service(tmp_path)
    service._sessions["ody-1"] = "ses-active"

    deleted = asyncio.run(service.delete_native_session("ody-1", "ses-active"))
    missing = asyncio.run(service.delete_native_session("ody-1", "ses-missing"))

    assert deleted["active_cleared"] is True
    assert "ody-1" not in service._sessions
    assert missing["deleted"] is False
    assert missing["not_found"] is True
    assert transport.deleted == ["ses-active", "ses-missing"]


def test_delete_is_blocked_during_an_active_run(tmp_path):
    service, transport, _, _ = make_service(tmp_path)
    service._runs["run-1"] = object()

    with pytest.raises(Exception, match="run is active"):
        asyncio.run(service.delete_native_session("ody-1", "ses-old"))

    assert transport.deleted == []


def test_tui_open_requires_an_active_session_and_never_creates(tmp_path):
    service, transport, manager, _ = make_service(tmp_path)

    with pytest.raises(Exception, match="Start or select"):
        asyncio.run(service.open_session_manager_terminal("ody-1"))

    assert transport.created == []
    assert manager.opened == []

    service._sessions["ody-1"] = "ses-active"
    result = asyncio.run(service.open_session_manager_terminal("ody-1"))
    assert result["opencode_session_id"] == "ses-active"
    assert manager.opened[0][0] == "ses-active"
    assert transport.created == []


def test_per_turn_model_selection_reaches_transport_payload():
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.transport import OpenCodeHttpTransport

    captured = {}
    transport = OpenCodeHttpTransport(
        OpenCodeRuntimeConfig(provider_id="ollama", model_id="model-a")
    )

    def request_json(method, path, payload=None, *, timeout=0):
        captured.update(method=method, path=path, payload=payload, timeout=timeout)
        return 204, None

    transport.request_json = request_json
    transport.submit_prompt(
        "ses-active",
        "hello",
        provider_id="ollama",
        model_id="model-b",
    )

    assert captured["payload"]["model"] == {
        "providerID": "ollama",
        "modelID": "model-b",
    }


def test_transport_scopes_native_list_and_reconciles_delete_404():
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.transport import (
        OpenCodeHttpTransport,
        OpenCodeTransportError,
    )

    transport = OpenCodeHttpTransport(OpenCodeRuntimeConfig())
    paths = []

    def request_json(method, path, payload=None, *, timeout=None):
        paths.append((method, path))
        if method == "DELETE":
            raise OpenCodeTransportError("missing", status_code=404)
        return 200, [{"id": "ses-1"}]

    transport.request_json = request_json

    assert transport.list_sessions("F:/Gauntlet of Gods") == [{"id": "ses-1"}]
    assert "directory=F%3A%2FGauntlet+of+Gods" in paths[0][1]
    assert transport.delete_session("ses-missing") is False


def test_composer_session_manager_and_tui_state_contracts():
    source = (ROOT / "static" / "js" / "runtimeControl.js").read_text(
        encoding="utf-8"
    )
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    assert "opencode-session-manager-btn" in source
    assert "right.insertBefore(button, tui)" in source
    assert "!ready || !!_activeRun || _openCodeSessionManagerBusy" in source
    assert "Start or select an OpenCode session first." in source
    assert "const ready = currentOpenCodeReadyState();\n  if (!ready) return null;" in source
    assert "openSessionNameDialog()" in source
    assert "if (!title) return;" in source
    assert "payload.active_opencode_session_id ? 'Switch' : 'Start'" in source
    assert "active ? '<span class=\"opencode-session-active-pill\">ACTIVE</span>' : ''" in source
    assert "active ? '' : `<button type=\"button\" data-action=\"activate\">" in source
    assert "data-action=\"select-mode\"" in source
    assert "data-action=\"select-all\"" in source
    assert "data-action=\"batch-delete\" disabled" in source
    assert "state.selected" in source
    assert "Permanently delete ${selectedIds.length}" in source
    assert "${failed} could not be deleted" in source
    assert "payload.active_cleared" in source
    assert "_openCodeReadySessions.delete" in source
    assert "async function openCodeResponseError" in source
    assert ".slice(0, 300)" in source
    assert "payload?.detail ?? payload?.error ?? payload?.message" in source
    assert ".opencode-session-row:hover .opencode-session-more" in style
    assert ".opencode-session-row:focus-within .opencode-session-more" in style
    assert ".opencode-session-active-pill" in style
    row_style = style[style.index(".opencode-session-row {"):]
    row_style = row_style[:row_style.index("}")]
    assert "background:" in row_style
    assert "border:" in row_style


def test_model_is_not_part_of_ready_session_identity():
    source = (ROOT / "static" / "js" / "runtimeControl.js").read_text(
        encoding="utf-8"
    )
    start = source.index("function openCodeReadyKey")
    end = source.index("\nfunction currentOpenCodeReadyState", start)
    ready_key = source[start:end]
    assert "_modelId" in ready_key
    assert "String(_modelId" not in ready_key
    assert "endpoint_id: selection.endpointId" in source
    assert "model_id: selection.modelId" in source
