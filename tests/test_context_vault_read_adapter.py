from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_runtime.context_vault_read_adapter import (
    ContextVaultReadAdapter,
    ContextVaultReadError,
    DEFAULT_VAULT_ROOT_ID,
)
from src.agent_runtime.knowledge_vault_index import build_structural_vault_index
from src.agent_runtime.obsidian_read_adapter import ObsidianReadError


VAULT_ITEMS = {
    ".": [
        {"path": "Atlas", "name": "Atlas", "type": "directory", "size": None},
        {"path": "AIOS", "name": "AIOS", "type": "directory", "size": None},
        {"path": "Long.md", "name": "Long.md", "type": "file", "size": 42},
    ],
    "Atlas": [
        {
            "path": "Atlas/Odysseus.md",
            "name": "Odysseus.md",
            "type": "file",
            "size": 100,
        },
    ],
    "AIOS": [
        {"path": "AIOS/WIP", "name": "WIP", "type": "directory", "size": None},
    ],
    "AIOS/WIP": [
        {
            "path": "AIOS/WIP/Runtime.md",
            "name": "Runtime.md",
            "type": "file",
            "size": 100,
        },
    ],
}

VAULT_TEXT = {
    "Atlas/Odysseus.md": "# Odysseus\n\nLangGraph evidence bridge notes.",
    "AIOS/WIP/Runtime.md": "# Runtime\n\nProvider-backed structural index note.",
    "Long.md": "# Long\n\n" + ("provider-backed content " * 400),
}


def _fake_context_tool_call(
    *,
    operation: str,
    root_id: str,
    selector: str = ".",
    config_path=None,
    limits=None,
    **kwargs,
):
    normalized = str(selector or ".").strip().replace("\\", "/")

    if root_id == "unknown_root":
        raise RuntimeError("Enabled root not found: unknown_root")

    if root_id == "disabled_root":
        raise RuntimeError("Enabled root not found: disabled_root")

    if root_id not in {"ideaverse_vault", "alternate_vault"}:
        raise RuntimeError(f"Enabled root not found: {root_id}")

    if normalized.startswith("/") or ":" in normalized:
        raise ValueError("Absolute selectors are not allowed.")

    if ".." in Path(normalized).parts:
        raise ValueError("Traversal selectors are not allowed.")

    if operation == "list":
        return {
            "ok": True,
            "operation": "list",
            "root": {"id": root_id},
            "selector": normalized,
            "items": list(VAULT_ITEMS.get(normalized, [])),
            "count": len(VAULT_ITEMS.get(normalized, [])),
        }

    if operation == "read":
        if normalized not in VAULT_TEXT:
            raise FileNotFoundError(normalized)

        text = VAULT_TEXT[normalized]
        return {
            "ok": True,
            "operation": "read",
            "root": {"id": root_id},
            "selector": normalized,
            "text": text,
            "characters": len(text),
            "truncated": False,
        }

    raise RuntimeError(f"unsupported operation: {operation}")


@pytest.fixture(autouse=True)
def fake_context_provider(monkeypatch):
    monkeypatch.setattr(
        "src.agent_runtime.context_vault_read_adapter.context_tool_call",
        _fake_context_tool_call,
    )


def test_default_root_id_is_ideaverse_vault():
    adapter = ContextVaultReadAdapter()

    assert DEFAULT_VAULT_ROOT_ID == "ideaverse_vault"
    assert adapter.root_id == "ideaverse_vault"


def test_constructor_root_id_override_uses_authorized_root():
    adapter = ContextVaultReadAdapter(root_id="alternate_vault")

    assert adapter.root_id == "alternate_vault"
    assert "Atlas/" in adapter.list_dir()


def test_error_subclasses_obsidian_read_error():
    assert issubclass(ContextVaultReadError, ObsidianReadError)


def test_list_dir_returns_child_names_and_directory_suffixes():
    adapter = ContextVaultReadAdapter()

    root_entries = adapter.list_dir()
    atlas_entries = adapter.list_dir("Atlas")
    aios_entries = adapter.list_dir("AIOS")

    assert root_entries == ["Atlas/", "AIOS/", "Long.md"]
    assert atlas_entries == ["Odysseus.md"]
    assert aios_entries == ["WIP/"]

    assert "Atlas/Odysseus.md" not in atlas_entries
    assert "AIOS/WIP/" not in aios_entries


def test_read_note_returns_provider_text():
    adapter = ContextVaultReadAdapter()

    text = adapter.read_note("Atlas/Odysseus.md")

    assert text == VAULT_TEXT["Atlas/Odysseus.md"]


def test_read_note_returns_long_text_without_default_clipping():
    adapter = ContextVaultReadAdapter()

    text = adapter.read_note("Long.md")

    assert text == VAULT_TEXT["Long.md"]
    assert len(text) > 6000


def test_traversal_selector_fails_closed():
    adapter = ContextVaultReadAdapter()

    with pytest.raises(ContextVaultReadError):
        adapter.read_note("../outside.md")


def test_absolute_selector_fails_closed():
    adapter = ContextVaultReadAdapter()

    with pytest.raises(ContextVaultReadError):
        adapter.read_note("D:/AI/Ideaverse_Test/Vault/Long.md")


def test_unknown_root_id_fails_closed():
    adapter = ContextVaultReadAdapter(root_id="unknown_root")

    with pytest.raises(ContextVaultReadError):
        adapter.list_dir()


def test_disabled_root_fails_closed():
    adapter = ContextVaultReadAdapter(root_id="disabled_root")

    with pytest.raises(ContextVaultReadError):
        adapter.list_dir()


def test_missing_unrelated_repo_does_not_break_ideaverse_vault():
    repo_adapter = ContextVaultReadAdapter(root_id="odysseus_repo")
    vault_adapter = ContextVaultReadAdapter(root_id="ideaverse_vault")

    with pytest.raises(ContextVaultReadError):
        repo_adapter.list_dir()

    assert "Atlas/" in vault_adapter.list_dir()
    assert "Odysseus" in vault_adapter.read_note("Atlas/Odysseus.md")


def test_adapter_exposes_no_mutation_methods():
    adapter = ContextVaultReadAdapter()

    forbidden_methods = {
        "write",
        "write_note",
        "create",
        "create_note",
        "update",
        "update_note",
        "delete",
        "delete_note",
        "move",
        "move_note",
        "rename",
        "rename_note",
        "append",
        "edit",
        "save",
    }

    assert all(not hasattr(adapter, name) for name in forbidden_methods)


def test_structural_index_builds_from_context_provider_adapter():
    adapter = ContextVaultReadAdapter()

    index = build_structural_vault_index(
        adapter=adapter,
        cache_path=None,
        max_depth=8,
        max_notes=20,
    )

    paths = {note.path for note in index.notes}
    folders = set(index.folders)

    assert index.note_count == 3
    assert "Atlas/Odysseus.md" in paths
    assert "AIOS/WIP/Runtime.md" in paths
    assert "Long.md" in paths

    assert "/" in folders
    assert "Atlas/" in folders
    assert "AIOS/" in folders
    assert "AIOS/WIP/" in folders

    assert all("Atlas/Atlas/" not in path for path in paths)
    assert all("AIOS/AIOS/" not in path for path in paths)