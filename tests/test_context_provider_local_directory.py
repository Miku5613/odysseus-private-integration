from __future__ import annotations

import os

import pytest

from src.context_providers import AuthorizedRoot
from src.context_providers.adapters import LocalDirectoryAdapter


def _root(tmp_path):
    return AuthorizedRoot.from_dict(
        {
            "id": "project",
            "provider_type": "local_dir",
            "display_name": "Project",
            "local_path": str(tmp_path),
            "allowed_capabilities": ["context.list", "context.read", "context.search"],
        }
    )


def test_local_directory_adapter_lists_files_and_folders(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"\x00\x01")

    adapter = LocalDirectoryAdapter()
    items = adapter.list(_root(tmp_path))

    paths = {item["path"] for item in items}
    assert "docs" in paths
    assert "README.md" in paths
    assert "image.bin" in paths


def test_local_directory_adapter_reads_text_file(tmp_path):
    (tmp_path / "README.md").write_text("Project notes", encoding="utf-8")

    adapter = LocalDirectoryAdapter()
    text = adapter.read(_root(tmp_path), "README.md")

    assert text == "Project notes"


def test_local_directory_adapter_rejects_traversal(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    adapter = LocalDirectoryAdapter()

    with pytest.raises(ValueError):
        adapter.read(_root(tmp_path), "../outside.txt")


def test_local_directory_adapter_rejects_absolute_selector(tmp_path):
    target = tmp_path / "README.md"
    target.write_text("Project notes", encoding="utf-8")

    adapter = LocalDirectoryAdapter()

    with pytest.raises(ValueError):
        adapter.read(_root(tmp_path), str(target))


def test_local_directory_adapter_rejects_non_text_file(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")

    adapter = LocalDirectoryAdapter()

    with pytest.raises(ValueError):
        adapter.read(_root(tmp_path), "blob.bin")


def test_local_directory_adapter_searches_text_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha beta gamma", encoding="utf-8")
    (tmp_path / "b.md").write_text("nothing here", encoding="utf-8")

    adapter = LocalDirectoryAdapter()
    results = adapter.search(_root(tmp_path), "beta")

    assert len(results) == 1
    assert results[0]["path"] == "a.md"
    assert "beta" in results[0]["snippet"]


def test_local_directory_adapter_skips_node_modules_in_search(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "bad.md").write_text("needle", encoding="utf-8")
    (tmp_path / "good.md").write_text("needle", encoding="utf-8")

    adapter = LocalDirectoryAdapter()
    results = adapter.search(_root(tmp_path), "needle")

    assert [r["path"] for r in results] == ["good.md"]


def test_local_directory_adapter_status(tmp_path):
    adapter = LocalDirectoryAdapter()
    status = adapter.status(_root(tmp_path))

    assert status["root_id"] == "project"
    assert status["exists"] is True
    assert status["is_dir"] is True
    assert status["readable"] is True
