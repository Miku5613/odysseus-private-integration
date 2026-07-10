from __future__ import annotations

import pytest

from src.context_providers import AuthorizedRoot, CapabilityNotSupported
from src.context_providers.adapters import LocalDirectoryAdapter
from src.context_providers.registry import (
    ContextProviderRegistry,
    ProviderNotRegistered,
    build_default_registry,
)


def _root(tmp_path, **overrides):
    raw = {
        "id": "project",
        "provider_type": "local_dir",
        "display_name": "Project",
        "local_path": str(tmp_path),
        "allowed_capabilities": [
            "context.list",
            "context.read",
            "context.search",
            "context.status",
        ],
    }
    raw.update(overrides)
    return AuthorizedRoot.from_dict(raw)


def test_registry_registers_adapter_by_provider_type():
    registry = ContextProviderRegistry()
    adapter = LocalDirectoryAdapter()
    registry.register(adapter)

    assert registry.registered_provider_types() == ("local_dir",)
    assert registry.adapter_for(_root_pathless()) is adapter


def test_registry_fails_closed_for_unknown_provider(tmp_path):
    root = _root(
        tmp_path,
        provider_type="mcp_integration",
        connection_id="demo",
        allowed_capabilities=["context.read"],
    )
    registry = ContextProviderRegistry()

    with pytest.raises(ProviderNotRegistered):
        registry.read(root, "anything")


def test_registry_enforces_root_allowed_capabilities(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    root = _root(
        tmp_path,
        allowed_capabilities=["context.list"],
    )
    registry = ContextProviderRegistry()
    registry.register(LocalDirectoryAdapter())

    with pytest.raises(CapabilityNotSupported):
        registry.read(root, "README.md")


def test_registry_enforces_adapter_supported_capabilities(tmp_path):
    root = _root(
        tmp_path,
        allowed_capabilities=["notes.write_session"],
    )
    registry = ContextProviderRegistry()
    registry.register(LocalDirectoryAdapter())

    with pytest.raises(CapabilityNotSupported):
        registry.require_capability(root, "notes.write_session")


def test_registry_enforces_disabled_root(tmp_path):
    root = _root(tmp_path, enabled=False)
    registry = ContextProviderRegistry()
    registry.register(LocalDirectoryAdapter())

    with pytest.raises(CapabilityNotSupported):
        registry.list(root)


def test_registry_list_read_search_status(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("alpha beta", encoding="utf-8")

    root = _root(tmp_path)
    registry = ContextProviderRegistry()
    registry.register(LocalDirectoryAdapter())

    listing = registry.list(root)
    assert {item["path"] for item in listing} == {"docs", "README.md"}

    assert registry.read(root, "README.md") == "alpha beta"

    results = registry.search(root, "beta")
    assert len(results) == 1
    assert results[0]["path"] == "README.md"

    status = registry.status(root)
    assert status["readable"] is True


def test_registry_public_status_hides_local_paths(tmp_path):
    registry = ContextProviderRegistry()
    registry.register(LocalDirectoryAdapter())

    status = registry.public_status()

    assert "local_dir" in status["providers"]
    dumped = str(status)
    assert str(tmp_path) not in dumped
    assert "LocalDirectoryAdapter" in dumped


def test_default_registry_handles_local_dir_and_obsidian_roots(tmp_path):
    (tmp_path / "note.md").write_text("vault note", encoding="utf-8")
    registry = build_default_registry()

    local_root = _root(tmp_path)
    obsidian_root = _root(
        tmp_path,
        id="vault",
        provider_type="obsidian_vault",
        display_name="Vault",
    )

    assert "local_dir" in registry.registered_provider_types()
    assert "obsidian_vault" in registry.registered_provider_types()
    assert registry.read(local_root, "note.md") == "vault note"
    assert registry.read(obsidian_root, "note.md") == "vault note"


def _root_pathless():
    return AuthorizedRoot.from_dict(
        {
            "id": "project",
            "provider_type": "local_dir",
            "display_name": "Project",
            "local_path": ".",
            "allowed_capabilities": ["context.read"],
        }
    )
