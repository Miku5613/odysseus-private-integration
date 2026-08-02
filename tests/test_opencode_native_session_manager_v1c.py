from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_native_session_identity_exposes_model_count_preview_and_timestamp():
    from src.opencode_runtime.service import OpenCodeRuntimeService

    item = {
        "id": "ses_generic_12345678",
        "title": "OpenCode Session",
        "model": {"providerID": "ollama", "id": "gemma4:12b"},
        "time": {"updated": 1_785_696_825_887},
    }
    messages = [
        {
            "info": {"role": "user"},
            "parts": [{"type": "text", "text": "Inspect the render pipeline"}],
        },
        {
            "info": {"role": "assistant"},
            "parts": [{"type": "text", "text": "I inspected the pipeline."}],
        },
    ]

    view = OpenCodeRuntimeService._native_session_view(item, "", messages)

    assert view["title"] == "OpenCode Session"
    assert view["model_name"] == "gemma4:12b"
    assert view["message_count"] == 2
    assert view["preview"] == "Inspect the render pipeline"
    assert view["updated_at"] == "2026-08-02T18:53:45.887000Z"


def test_identity_falls_back_to_message_model_and_truncates_preview():
    from src.opencode_runtime.service import OpenCodeRuntimeService

    messages = [{
        "info": {
            "role": "user",
            "model": {"providerID": "provider", "modelID": "qwen3-vl:8b"},
        },
        "parts": [{"type": "text", "text": "x" * 240}],
    }]

    view = OpenCodeRuntimeService._native_session_view(
        {"id": "ses_generic", "title": "OpenCode Session"},
        "ses_generic",
        messages,
    )

    assert view["active"] is True
    assert view["model_name"] == "qwen3-vl:8b"
    assert len(view["preview"]) == 180
    assert view["preview"].endswith("...")


def test_transport_text_only_payload_is_unchanged_and_images_keep_order():
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.transport import OpenCodeHttpTransport

    payloads = []
    transport = OpenCodeHttpTransport(OpenCodeRuntimeConfig())
    transport.request_json = lambda method, path, payload=None, **_: (
        payloads.append(payload) or (204, None)
    )

    transport.submit_prompt("ses_text", "hello")
    transport.submit_prompt(
        "ses_image",
        "look",
        file_parts=(
            {
                "type": "file",
                "mime": "image/png",
                "filename": "one.png",
                "url": "data:image/png;base64,AAAA",
            },
            {
                "type": "file",
                "mime": "image/webp",
                "filename": "two.webp",
                "url": "data:image/webp;base64,BBBB",
            },
        ),
    )

    assert payloads[0]["parts"] == [{"type": "text", "text": "hello"}]
    assert [part["type"] for part in payloads[1]["parts"]] == [
        "text",
        "file",
        "file",
    ]
    assert [part.get("filename") for part in payloads[1]["parts"][1:]] == [
        "one.png",
        "two.webp",
    ]
    assert "tools" not in payloads[1]


def test_transport_suppresses_tools_for_ollama_media_turns():
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.transport import OpenCodeHttpTransport

    payloads = []
    transport = OpenCodeHttpTransport(OpenCodeRuntimeConfig(
        provider_base_url="http://127.0.0.1:11434/v1",
    ))
    transport.request_json = lambda method, path, payload=None, **_: (
        payloads.append(payload) or (204, None)
    )

    transport.submit_prompt(
        "ses_image",
        "look",
        file_parts=({
            "mime": "image/png",
            "filename": "one.png",
            "url": "data:image/png;base64,AAAA",
        },),
    )

    assert payloads[0]["tools"] == {"*": False}


def test_image_resolution_is_owner_checked_and_uses_documented_file_part(
    tmp_path,
    monkeypatch,
):
    from routes import opencode_runtime_routes as routes

    image = tmp_path / "source.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\npatch-1c")
    calls = []

    class UploadHandler:
        def resolve_upload(self, attachment_id, **kwargs):
            calls.append((attachment_id, kwargs))
            return {
                "id": attachment_id,
                "path": str(image),
                "mime": "image/png",
                "name": "source.png",
                "original_name": "source.png",
                "size": image.stat().st_size,
            }

        def _inside_upload_dir(self, path):
            return Path(path) == image

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(upload_handler=UploadHandler()))
    )
    monkeypatch.setattr(routes, "effective_user", lambda _request: "alice")

    parts, refs = routes._resolve_opencode_image_attachments(
        request,
        ["upload-one", "upload-one"],
    )

    assert len(parts) == 1
    assert parts[0]["type"] == "file"
    assert parts[0]["mime"] == "image/png"
    assert parts[0]["url"].startswith("data:image/png;base64,")
    assert refs[0]["attachment_id"] == "upload-one"
    assert calls == [("upload-one", {"owner": "alice", "allow_admin": False})]


@pytest.mark.parametrize(
    ("mime", "detail"),
    [
        ("image/gif", "do not support GIF"),
        ("application/pdf", "Use PNG, JPEG, or WebP"),
    ],
)
def test_unsupported_image_types_are_rejected(tmp_path, monkeypatch, mime, detail):
    from fastapi import HTTPException
    from routes import opencode_runtime_routes as routes

    source = tmp_path / "upload.bin"
    source.write_bytes(b"not-an-image")

    class UploadHandler:
        def resolve_upload(self, *_args, **_kwargs):
            return {"id": "upload", "path": str(source), "mime": mime}

        def _inside_upload_dir(self, _path):
            return True

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(upload_handler=UploadHandler()))
    )
    monkeypatch.setattr(routes, "effective_user", lambda _request: "alice")

    with pytest.raises(HTTPException, match=detail):
        routes._resolve_opencode_image_attachments(request, ["upload"])


def test_missing_or_foreign_image_is_rejected_clearly(monkeypatch):
    from fastapi import HTTPException
    from routes import opencode_runtime_routes as routes

    class UploadHandler:
        def resolve_upload(self, *_args, **_kwargs):
            return None

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(upload_handler=UploadHandler()))
    )
    monkeypatch.setattr(routes, "effective_user", lambda _request: "alice")

    with pytest.raises(HTTPException, match="does not belong to the current user"):
        routes._resolve_opencode_image_attachments(request, ["foreign-upload"])


def test_stream_route_rejects_known_non_vision_model_with_readable_detail(
    monkeypatch,
):
    from fastapi import HTTPException
    from routes import opencode_runtime_routes as routes

    class Service:
        def validate_turn_selection(self, **_kwargs):
            return "loaded-provider"

    selection = SimpleNamespace(
        endpoint_id="endpoint-a",
        provider_id="registry-provider",
        base_url="http://127.0.0.1:11434/v1",
        models=("qwen3:8b",),
        model_id="qwen3:8b",
    )
    monkeypatch.setattr(routes, "_service", Service())
    monkeypatch.setattr(routes, "_verify_opencode_session_owner", lambda *_: None)
    monkeypatch.setattr(routes, "resolve_model_selection", lambda **_: selection)
    monkeypatch.setattr(
        routes,
        "_resolve_opencode_image_attachments",
        lambda *_: ([{"type": "file"}], [{"attachment_id": "upload-one"}]),
    )
    monkeypatch.setattr(routes, "model_supports_vision", lambda *_: False)
    request = routes.OpenCodeRuntimeRequest(
        message="inspect",
        odysseus_session_id="ody-1",
        endpoint_id="endpoint-a",
        model_id="qwen3:8b",
        attachments=["upload-one"],
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.stream_opencode_runtime(object(), request))

    assert error.value.status_code == 400
    assert error.value.detail == (
        "The selected OpenCode model does not support image input. "
        "Select a vision-capable model and try again."
    )


def test_stream_route_forwards_vision_parts_to_same_service_stream(monkeypatch):
    from routes import opencode_runtime_routes as routes

    captured = {}

    class Service:
        def validate_turn_selection(self, **_kwargs):
            return "loaded-provider"

        def stream(self, **kwargs):
            captured.update(kwargs)

            async def frames():
                if False:
                    yield ""

            return frames()

    selection = SimpleNamespace(
        endpoint_id="endpoint-a",
        provider_id="registry-provider",
        base_url="http://127.0.0.1:11434/v1",
        models=("gemma4:12b",),
        model_id="gemma4:12b",
    )
    file_part = {
        "type": "file",
        "mime": "image/png",
        "filename": "image.png",
        "url": "data:image/png;base64,AAAA",
    }
    reference = {"type": "attachment_ref", "attachment_id": "upload-one"}
    monkeypatch.setattr(routes, "_service", Service())
    monkeypatch.setattr(routes, "_verify_opencode_session_owner", lambda *_: None)
    monkeypatch.setattr(routes, "resolve_model_selection", lambda **_: selection)
    monkeypatch.setattr(
        routes,
        "_resolve_opencode_image_attachments",
        lambda *_: ([file_part], [reference]),
    )
    monkeypatch.setattr(routes, "model_supports_vision", lambda *_: True)
    request = routes.OpenCodeRuntimeRequest(
        message="inspect",
        odysseus_session_id="ody-1",
        endpoint_id="endpoint-a",
        model_id="gemma4:12b",
        attachments=["upload-one"],
    )

    asyncio.run(routes.stream_opencode_runtime(object(), request))

    assert captured["attachment_parts"] == (file_part,)
    assert captured["attachment_refs"] == (reference,)
    assert captured["model_id"] == "gemma4:12b"


def test_image_stream_uses_existing_native_session_without_creating(tmp_path):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService

    submitted = []
    transcript = []

    class Transport:
        def __init__(self):
            self.reads = 0

        def health(self):
            return {"healthy": True}

        def create_session(self, _title):
            raise AssertionError("image send must not create a native session")

        def list_messages(self, _session_id):
            self.reads += 1
            if self.reads == 1:
                return []
            return [{
                "info": {
                    "id": "msg-final",
                    "role": "assistant",
                    "providerID": "loaded-provider",
                    "modelID": "gemma4:12b",
                    "finish": "stop",
                },
                "parts": [{"type": "text", "text": "Image received."}],
            }]

        def submit_prompt(self, session_id, message, **kwargs):
            submitted.append((session_id, message, kwargs))

        def iter_global_events(self, _stop_event):
            yield {
                "type": "session.idle",
                "properties": {"sessionID": "ses-existing"},
            }

        def abort_session(self, _session_id):
            return True

    class Manager:
        owns_process = True
        log_path = tmp_path / "server.log"
        log_root = tmp_path

        def ensure_running(self):
            return {
                "mode": "managed",
                "owned": True,
                "pid": 1,
                "health": {"healthy": True},
            }

    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(
            workspace=str(tmp_path),
            provider_id="loaded-provider",
            model_id="gemma4:12b",
        ),
        transport=Transport(),
        server_manager=Manager(),
        transcript_persister=lambda *args: transcript.append(args),
    )
    service._sessions["ody-1"] = "ses-existing"
    part = {
        "type": "file",
        "mime": "image/png",
        "filename": "image.png",
        "url": "data:image/png;base64,AAAA",
    }

    async def collect():
        frames = []
        async for frame in service.stream(
            message="Inspect this image",
            odysseus_session_id="ody-1",
            attachment_parts=(part,),
            attachment_refs=({
                "type": "attachment_ref",
                "attachment_id": "upload-one",
                "mime": "image/png",
            },),
        ):
            frames.append(json.loads(frame.removeprefix("data: ").strip()))
        return frames

    frames = asyncio.run(collect())

    assert service._sessions["ody-1"] == "ses-existing"
    assert submitted[0][0] == "ses-existing"
    assert submitted[0][2]["file_parts"] == (part,)
    assert any(frame["type"] == "runtime.accepted" for frame in frames)
    assert transcript[0][3]["attachments"][0]["attachment_id"] == "upload-one"
    assert "base64" not in json.dumps(transcript)


def test_process_manager_launches_real_opencode_attach_command(tmp_path, monkeypatch):
    import src.opencode_runtime.process_manager as process_module
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.process_manager import OpenCodeServerManager

    launched = []
    fake_process = SimpleNamespace(pid=77)
    executable = str(tmp_path / "opencode.exe")
    Path(executable).write_bytes(b"")
    monkeypatch.setattr(process_module.shutil, "which", lambda _name: None)
    manager = OpenCodeServerManager(
        OpenCodeRuntimeConfig(
            base_url="http://127.0.0.1:4096",
            workspace=str(tmp_path),
            executable=executable,
        ),
        health_probe=lambda: {"healthy": True},
        process_factory=lambda command, **kwargs: (
            launched.append((command, kwargs)) or fake_process
        ),
    )

    result = manager.open_native_tui("ses-existing", workspace=str(tmp_path))

    command_text = " ".join(launched[0][0])
    assert "attach" in command_text
    assert "http://127.0.0.1:4096" in command_text
    assert "--dir" in command_text
    assert str(tmp_path) in command_text
    assert "--session" in command_text
    assert "ses-existing" in command_text
    assert "session_manager_tui.py" not in command_text
    assert result["client"] == "opencode-native-tui"


def test_native_tui_missing_executable_and_session_errors_are_readable(
    tmp_path,
):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.process_manager import (
        OpenCodeServerLifecycleError,
        OpenCodeServerManager,
    )

    manager = OpenCodeServerManager(
        OpenCodeRuntimeConfig(workspace=str(tmp_path), executable="missing-opencode"),
        health_probe=lambda: {"healthy": True},
        which=lambda _name: None,
    )

    with pytest.raises(OpenCodeServerLifecycleError, match="valid OpenCode session ID"):
        manager.open_native_tui("", workspace=str(tmp_path))
    with pytest.raises(OpenCodeServerLifecycleError, match="executable was not found"):
        manager.open_native_tui("ses-existing", workspace=str(tmp_path))


def test_native_tui_service_does_not_switch_or_create_session(tmp_path):
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService

    class Transport:
        def health(self):
            return {"healthy": True}

        def get_session(self, session_id):
            if session_id == "ses-missing":
                return None
            return {"id": session_id, "title": "Existing"}

        def create_session(self, _title):
            raise AssertionError("opening TUI must not create a session")

    class Manager:
        owns_process = True
        log_path = tmp_path / "server.log"
        log_root = tmp_path

        def ensure_running(self):
            return {
                "mode": "managed",
                "owned": True,
                "pid": 1,
                "health": {"healthy": True},
            }

        def open_native_tui(self, session_id, **kwargs):
            return {
                "opened": True,
                "client": "opencode-native-tui",
                "opencode_session_id": session_id,
                "workspace": kwargs["workspace"],
            }

    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(workspace=str(tmp_path)),
        transport=Transport(),
        server_manager=Manager(),
    )
    service._sessions["ody-1"] = "ses-active"

    result = asyncio.run(
        service.open_native_tui(
            "ody-1",
            "ses-listed",
            workspace=str(tmp_path),
        )
    )

    assert result["opencode_session_id"] == "ses-listed"
    assert service._sessions["ody-1"] == "ses-active"
    with pytest.raises(KeyError, match="session was not found"):
        asyncio.run(
            service.open_native_tui(
                "ody-1",
                "ses-missing",
                workspace=str(tmp_path),
            )
        )


def test_patch_1c_frontend_contracts_preserve_retry_attachments():
    runtime = (ROOT / "static/js/runtimeControl.js").read_text(encoding="utf-8")
    chat = (ROOT / "static/js/chat.js").read_text(encoding="utf-8")
    files = (ROOT / "static/js/fileHandler.js").read_text(encoding="utf-8")
    style = (ROOT / "static/style.css").read_text(encoding="utf-8")

    assert "Open in OpenCode TUI" in runtime
    assert "data-action=\"open-native-tui\"" in runtime
    assert "/open-tui`" in runtime
    assert "item.model_name" in runtime
    assert "item.message_count" in runtime
    assert "item.preview" in runtime
    assert "nativeOpenCodeSessionMatches" in runtime
    assert "runtime.accepted" in runtime
    assert "if (accepted) return;" in runtime
    assert "attachments: attachmentIds" in runtime
    assert "retainOnSuccess: true" in chat
    assert "commitLastUploadedPending" in chat
    assert "onAccepted:" in chat
    assert "if (!opts.retainOnSuccess)" in files
    assert "export function commitLastUploadedPending" in files
    assert ".opencode-session-preview" in style
