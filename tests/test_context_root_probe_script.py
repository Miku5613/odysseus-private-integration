
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_config(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / "README.md").write_text("alpha beta gamma", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "note.md").write_text("needle note", encoding="utf-8")

    config = tmp_path / "local_roots.json"
    config.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "id": "test_root",
                        "provider_type": "local_dir",
                        "display_name": "Test Root",
                        "local_path": str(root),
                        "allowed_capabilities": [
                            "context.list",
                            "context.read",
                            "context.search",
                            "context.status",
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


def test_context_root_probe_default_status_and_list(tmp_path):
    config = _write_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_root_probe.py",
            "--config",
            str(config),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["root"]["id"] == "test_root"
    assert [op["operation"] for op in payload["operations"]] == ["status", "list"]


def test_context_root_probe_read(tmp_path):
    config = _write_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_root_probe.py",
            "--config",
            str(config),
            "--root",
            "test_root",
            "--read",
            "README.md",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    op = payload["operations"][0]
    assert op["ok"] is True
    assert op["result"]["preview"] == "alpha beta gamma"


def test_context_root_probe_search(tmp_path):
    config = _write_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_root_probe.py",
            "--config",
            str(config),
            "--root",
            "test_root",
            "--search",
            "needle",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    op = payload["operations"][0]
    assert op["ok"] is True
    assert op["result"][0]["path"] == "docs/note.md"


def test_context_root_probe_missing_root_returns_error(tmp_path):
    config = _write_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_root_probe.py",
            "--config",
            str(config),
            "--root",
            "missing",
            "--json",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["requested_root"] == "missing"


def test_context_root_probe_blocks_traversal(tmp_path):
    config = _write_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_root_probe.py",
            "--config",
            str(config),
            "--root",
            "test_root",
            "--read",
            "../secret.txt",
            "--json",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["operations"][0]["ok"] is False
