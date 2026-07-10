
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.context_providers.runtime_tool import (
    RuntimeContextToolError,
    RuntimeContextToolLimits,
    context_tool_call,
)


def _write_config(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("alpha beta gamma", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "context.md").write_text(
        "This file explains context provider runtime usage.",
        encoding="utf-8",
    )

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Daily.md").write_text("Today we studied ChromaDB and Atlas.", encoding="utf-8")

    config = tmp_path / "local_roots.json"
    config.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "id": "repo",
                        "provider_type": "local_dir",
                        "display_name": "Repo",
                        "local_path": str(repo),
                        "allowed_capabilities": [
                            "context.list",
                            "context.read",
                            "context.search",
                            "context.status",
                        ],
                        "safety_scope": "read_only_sandbox",
                        "enabled": True,
                    },
                    {
                        "id": "vault",
                        "provider_type": "obsidian_vault",
                        "display_name": "Vault",
                        "local_path": str(vault),
                        "allowed_capabilities": [
                            "context.list",
                            "context.read",
                            "context.search",
                            "context.status",
                        ],
                        "safety_scope": "read_only_sandbox",
                        "enabled": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return config


def test_context_tool_roots_lists_public_roots(tmp_path):
    config = _write_config(tmp_path)

    result = context_tool_call(operation="roots", config_path=config)

    assert result["ok"] is True
    assert result["operation"] == "roots"
    assert {root["id"] for root in result["roots"]} == {"repo", "vault"}
    assert all("local_path" not in root for root in result["roots"])


def test_context_tool_list_requires_explicit_root(tmp_path):
    config = _write_config(tmp_path)

    with pytest.raises(RuntimeContextToolError):
        context_tool_call(operation="list", config_path=config)


def test_context_tool_list_root_items(tmp_path):
    config = _write_config(tmp_path)

    result = context_tool_call(operation="list", root_id="repo", selector=".", config_path=config)

    assert result["ok"] is True
    assert {item["path"] for item in result["items"]} == {"docs", "README.md"}


def test_context_tool_read_exact_selector(tmp_path):
    config = _write_config(tmp_path)

    result = context_tool_call(
        operation="read",
        root_id="repo",
        selector="README.md",
        config_path=config,
    )

    assert result["ok"] is True
    assert result["text"] == "alpha beta gamma"
    assert result["truncated"] is False


def test_context_tool_search_can_scan_all_enabled_roots(tmp_path):
    config = _write_config(tmp_path)

    result = context_tool_call(operation="search", query="ChromaDB", config_path=config)

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["results"][0]["root_id"] == "vault"
    assert result["results"][0]["path"] == "Daily.md"


def test_context_tool_search_read_finds_and_reads_without_user_path(tmp_path):
    config = _write_config(tmp_path)

    result = context_tool_call(operation="search_read", query="runtime", config_path=config)

    assert result["ok"] is True
    assert result["match_count"] == 1
    assert result["document_count"] == 1
    doc = result["documents"][0]
    assert doc["root_id"] == "repo"
    assert doc["path"] == "docs/context.md"
    assert "context provider runtime usage" in doc["text"]


def test_context_tool_search_read_respects_limits(tmp_path):
    config = _write_config(tmp_path)

    result = context_tool_call(
        operation="search_read",
        query="context",
        config_path=config,
        limits=RuntimeContextToolLimits(
            max_search_results=5,
            max_read_files=1,
            max_chars_per_file=10,
            max_total_chars=10,
        ),
    )

    assert result["ok"] is True
    assert result["document_count"] == 1
    assert len(result["documents"][0]["text"]) <= 10
    assert result["documents"][0]["truncated"] is True


def test_context_tool_blocks_traversal_through_adapter(tmp_path):
    config = _write_config(tmp_path)

    with pytest.raises(ValueError):
        context_tool_call(
            operation="read",
            root_id="repo",
            selector="../secret.txt",
            config_path=config,
        )


def test_context_tool_rejects_empty_search_query(tmp_path):
    config = _write_config(tmp_path)

    with pytest.raises(RuntimeContextToolError):
        context_tool_call(operation="search", query="", config_path=config)
