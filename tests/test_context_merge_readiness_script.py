
from __future__ import annotations

import json
from pathlib import Path

from scripts.context_merge_readiness import run_merge_readiness


def _write_config(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("Atlas context readiness smoke.", encoding="utf-8")

    config = tmp_path / "local_roots.json"
    config.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "id": "vault",
                        "provider_type": "obsidian_vault",
                        "display_name": "Vault",
                        "local_path": str(root),
                        "allowed_capabilities": [
                            "context.status",
                            "context.list",
                            "context.read",
                            "context.search",
                        ],
                        "safety_scope": "read_only_sandbox",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config


def test_merge_readiness_passes_for_valid_local_config(tmp_path):
    config = _write_config(tmp_path)

    result = run_merge_readiness(config_path=config, smoke_query="readiness")

    assert result["ok"] is True
    assert result["failed_count"] == 0
    names = {check["name"] for check in result["checks"]}
    assert "config_exists" in names
    assert "config_loads" in names
    assert "enabled_roots_present" in names
    assert "root_verify:vault" in names
    assert "adapter_status:vault" in names
    assert "runtime_tool_roots" in names
    assert "runtime_tool_search_read_smoke" in names


def test_merge_readiness_fails_for_missing_config(tmp_path):
    config = tmp_path / "missing.json"

    result = run_merge_readiness(config_path=config)

    assert result["ok"] is False
    assert result["failed_count"] == 1
    assert result["checks"][0]["name"] == "config_exists"


def test_merge_readiness_fails_for_missing_root_path(tmp_path):
    config = tmp_path / "local_roots.json"
    config.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "id": "missing",
                        "provider_type": "local_dir",
                        "display_name": "Missing",
                        "local_path": str(tmp_path / "does-not-exist"),
                        "allowed_capabilities": [
                            "context.status",
                            "context.list",
                            "context.read",
                            "context.search",
                        ],
                        "safety_scope": "read_only_sandbox",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_merge_readiness(config_path=config)

    assert result["ok"] is False
    assert any(check["name"] == "root_verify:missing" for check in result["checks"])
