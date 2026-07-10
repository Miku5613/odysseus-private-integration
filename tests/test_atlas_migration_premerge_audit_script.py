
from __future__ import annotations

import json
from pathlib import Path

from scripts.atlas_migration_premerge_audit import run_premerge_audit


def _write_config(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("premerge audit context smoke", encoding="utf-8")

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


def test_premerge_audit_reports_missing_base_without_crashing(tmp_path):
    config = _write_config(tmp_path)

    report = run_premerge_audit(
        base_ref="refs/heads/definitely-missing-atlas-base",
        config_path=config,
        smoke_query="premerge",
    )

    assert report["ok"] is False
    assert any(check["name"] == "base_ref_exists" for check in report["checks"])
    assert any(check["name"] == "context_merge_readiness" for check in report["checks"])


def test_premerge_audit_report_shape_with_head_base(tmp_path):
    config = _write_config(tmp_path)

    report = run_premerge_audit(
        base_ref="HEAD",
        config_path=config,
        smoke_query="premerge",
    )

    names = {check["name"] for check in report["checks"]}

    assert "current_branch" in names
    assert "working_tree_clean" in names
    assert "base_ref_exists" in names
    assert "merge_base" in names
    assert "changed_files_since_base" in names
    assert "runtime_sensitive_files_touched_review" in names
    assert "outside_expected_atlas_context_scope_review" in names
    assert "local_roots_json_ignored" in names
    assert "local_roots_example_tracked" in names
    assert "context_merge_readiness" in names


def test_premerge_audit_context_readiness_can_pass_with_valid_config(tmp_path):
    config = _write_config(tmp_path)

    report = run_premerge_audit(
        base_ref="HEAD",
        config_path=config,
        smoke_query="premerge",
    )

    readiness = next(
        check
        for check in report["checks"]
        if check["name"] == "context_merge_readiness"
    )

    assert readiness["ok"] is True
