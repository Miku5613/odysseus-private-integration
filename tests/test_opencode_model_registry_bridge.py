from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _fake_session(endpoint):
    class Query:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return endpoint

    class Session:
        def query(self, model):
            return Query()

        def close(self):
            return None

    return Session


def test_registry_resolves_visible_endpoint_models() -> None:
    from src.opencode_runtime.model_registry import resolve_model_selection

    endpoint = SimpleNamespace(
        id="local-ollama",
        name="Local Ollama",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        is_enabled=True,
        model_type="llm",
        cached_models=json.dumps([
            "qwen3:8b",
            "gemma4:12b",
            "hidden-model",
        ]),
        pinned_models=json.dumps(["qwen3.5:9b", "qwen3:8b"]),
        hidden_models=json.dumps(["hidden-model"]),
    )

    selection = resolve_model_selection(
        endpoint_id="local-ollama",
        model_id="qwen3:8b",
        session_factory=lambda: _fake_session(endpoint)(),
    )

    assert selection.provider_id.startswith("odysseus_")
    assert selection.provider_name == "Odysseus · Local Ollama"
    assert selection.base_url == "http://127.0.0.1:11434/v1"
    assert selection.models == (
        "qwen3:8b",
        "gemma4:12b",
        "qwen3.5:9b",
    )


def test_registry_rejects_unknown_model() -> None:
    from src.opencode_runtime.model_registry import (
        OpenCodeModelRegistryError,
        resolve_model_selection,
    )

    endpoint = SimpleNamespace(
        id="local-ollama",
        name="Local Ollama",
        base_url="http://127.0.0.1:11434/v1",
        api_key=None,
        is_enabled=True,
        model_type="llm",
        cached_models=json.dumps(["qwen3:8b"]),
        pinned_models=None,
        hidden_models=None,
    )

    with pytest.raises(OpenCodeModelRegistryError, match="not enabled"):
        resolve_model_selection(
            endpoint_id="local-ollama",
            model_id="missing",
            session_factory=lambda: _fake_session(endpoint)(),
        )


def test_managed_config_injects_registry_provider(tmp_path) -> None:
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.process_manager import OpenCodeServerManager

    manager = OpenCodeServerManager(
        OpenCodeRuntimeConfig(
            provider_id="odysseus_test",
            model_id="qwen3:8b",
            endpoint_id="local-ollama",
            provider_name="Odysseus · Local Ollama",
            provider_base_url="http://127.0.0.1:11434/v1",
            provider_models=("qwen3:8b", "gemma4:12b"),
        ),
        health_probe=lambda: {"healthy": True},
        log_root=tmp_path / "logs",
    )

    effective_path, changes = manager._prepare_managed_config()
    payload = json.loads(Path(effective_path).read_text(encoding="utf-8"))

    assert payload["model"] == "odysseus_test/qwen3:8b"
    provider = payload["provider"]["odysseus_test"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == (
        "http://127.0.0.1:11434/v1"
    )
    assert set(provider["models"]) == {"qwen3:8b", "gemma4:12b"}
    assert provider["models"]["qwen3:8b"]["modalities"] == {
        "input": ["text"],
        "output": ["text"],
    }
    assert provider["models"]["gemma4:12b"]["modalities"] == {
        "input": ["text", "image"],
        "output": ["text"],
    }
    assert provider["models"]["gemma4:12b"]["attachment"] is True
    assert provider["models"]["gemma4:12b"]["options"] == {
        "reasoningEffort": "none",
    }
    assert "attachment" not in provider["models"]["qwen3:8b"]
    assert "options" not in provider["models"]["qwen3:8b"]
    assert any(
        item["location"] == "provider.odysseus_test"
        for item in changes
    )


def test_managed_config_keeps_api_key_out_of_file(tmp_path) -> None:
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.process_manager import OpenCodeServerManager

    manager = OpenCodeServerManager(
        OpenCodeRuntimeConfig(
            provider_id="odysseus_secure",
            model_id="cloud-model",
            provider_name="Secure",
            provider_base_url="https://example.invalid/v1",
            provider_models=("cloud-model",),
            provider_api_key="must-not-be-written",
        ),
        health_probe=lambda: {"healthy": True},
        log_root=tmp_path / "logs",
    )

    effective_path, _ = manager._prepare_managed_config()
    text = Path(effective_path).read_text(encoding="utf-8")
    payload = json.loads(text)

    assert "must-not-be-written" not in text
    assert payload["provider"]["odysseus_secure"]["options"]["apiKey"] == (
        "{env:ODYSSEUS_OPENCODE_MODEL_API_KEY}"
    )


def test_service_restart_accepts_registry_provider_fields(
    tmp_path,
    monkeypatch,
) -> None:
    import src.opencode_runtime.service as service_module
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.service import OpenCodeRuntimeService

    class OldTransport:
        def health(self):
            return {"healthy": True}

        def delete_session(self, session_id):
            return True

    class OldManager:
        owns_process = True
        log_path = tmp_path / "old.log"
        log_root = tmp_path / "logs"

        def stop_owned(self):
            return True

    class NewTransport:
        def __init__(self, config):
            self.config = config

        def health(self):
            return {"healthy": True, "version": "test"}

    class NewManager:
        def __init__(self, config, *, health_probe, log_root=None):
            self.config = config
            self.health_probe = health_probe
            self.log_root = log_root
            self.owns_process = True
            self.log_path = tmp_path / "new.log"

        def ensure_running(self):
            return {
                "mode": "managed",
                "owned": True,
                "pid": 99,
                "health": self.health_probe(),
            }

    monkeypatch.setattr(service_module, "OpenCodeHttpTransport", NewTransport)
    monkeypatch.setattr(service_module, "OpenCodeServerManager", NewManager)

    service = OpenCodeRuntimeService(
        config=OpenCodeRuntimeConfig(workspace=str(tmp_path)),
        transport=OldTransport(),
        server_manager=OldManager(),
    )

    health = asyncio.run(
        service.restart_server(
            model_id="qwen3:8b",
            workspace=str(tmp_path),
            endpoint_id="local-ollama",
            provider_id="odysseus_test",
            provider_name="Odysseus · Local Ollama",
            provider_base_url="http://127.0.0.1:11434/v1",
            provider_models=("qwen3:8b", "gemma4:12b"),
            provider_api_key="",
        )
    )

    assert service.config.endpoint_id == "local-ollama"
    assert service.config.provider_id == "odysseus_test"
    assert health["model"]["endpoint_id"] == "local-ollama"


def test_runtime_ui_uses_registry_selection() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "js"
        / "runtimeControl.js"
    ).read_text(encoding="utf-8")

    assert '<select id="runtime-opencode-model"' in source
    assert "loadOpenCodeModelOptions" in source
    assert "endpoint_id:" in source
    assert "option.dataset.endpointId" in source
    assert 'placeholder="qwen3:8b"' not in source
