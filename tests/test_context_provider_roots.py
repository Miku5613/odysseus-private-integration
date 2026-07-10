from __future__ import annotations

import json

import pytest

from src.context_providers import (
    AuthorizedRoot,
    BaseContextAdapter,
    CapabilityNotSupported,
    RootConfigError,
    load_authorized_roots,
    verify_authorized_root,
)


def test_missing_local_roots_config_returns_empty(tmp_path):
    assert load_authorized_roots(tmp_path / "missing.json") == []


def test_load_authorized_root_from_env_path(tmp_path, monkeypatch):
    root_dir = tmp_path / "vault"
    root_dir.mkdir()
    monkeypatch.setenv("TEST_VAULT_ROOT", str(root_dir))

    cfg = tmp_path / "local_roots.json"
    cfg.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "id": "vault_main",
                        "provider_type": "obsidian_vault",
                        "display_name": "Main Vault",
                        "path_env": "TEST_VAULT_ROOT",
                        "allowed_capabilities": ["context.read", "context.list"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    roots = load_authorized_roots(cfg)
    assert len(roots) == 1
    root = roots[0]
    assert root.id == "vault_main"
    assert root.resolve_path().exists()
    assert verify_authorized_root(root)["status"] == "ok"


def test_public_summary_does_not_expose_raw_local_path(tmp_path):
    root = AuthorizedRoot.from_dict(
        {
            "id": "project_main",
            "provider_type": "local_dir",
            "display_name": "Project",
            "local_path": str(tmp_path),
            "allowed_capabilities": ["context.read"],
        }
    )

    summary = root.public_summary()
    assert "local_path" not in summary
    assert str(tmp_path) not in json.dumps(summary)
    assert summary["has_local_path"] is True


def test_invalid_capability_fails_closed(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "id": "bad_root",
                        "provider_type": "local_dir",
                        "display_name": "Bad Root",
                        "local_path": str(tmp_path),
                        "allowed_capabilities": ["context.delete_everything"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RootConfigError):
        load_authorized_roots(cfg)


def test_duplicate_root_ids_fail(tmp_path):
    cfg = tmp_path / "dupes.json"
    entry = {
        "id": "dup_root",
        "provider_type": "local_dir",
        "display_name": "Duplicate",
        "local_path": str(tmp_path),
        "allowed_capabilities": ["context.read"],
    }
    cfg.write_text(json.dumps({"roots": [entry, entry]}), encoding="utf-8")

    with pytest.raises(RootConfigError):
        load_authorized_roots(cfg)


def test_adapter_capability_gate_fails_closed():
    class ReadOnlyAdapter(BaseContextAdapter):
        provider_type = "local_dir"
        supported_capabilities = frozenset({"context.read"})

    adapter = ReadOnlyAdapter()
    assert adapter.supports("context.read")
    assert not adapter.supports("context.write")

    with pytest.raises(CapabilityNotSupported):
        adapter.require("context.write")
