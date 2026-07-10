
from __future__ import annotations

import json
import subprocess
import sys


def test_context_roots_status_missing_config_returns_no_roots():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_roots_status.py",
            "--config",
            "config/definitely_missing_local_roots.json",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["root_count"] == 0
    assert payload["roots"] == []
    assert "local_dir" in payload["registry"]["providers"]
    assert "obsidian_vault" in payload["registry"]["providers"]


def test_context_roots_status_example_config_is_readable():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_roots_status.py",
            "--config",
            "config/local_roots.example.json",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["root_count"] == 2
    ids = {root["id"] for root in payload["roots"]}
    assert ids == {"example_workspace", "example_obsidian_vault"}
    assert all("local_path" not in root for root in payload["roots"])


def test_context_roots_status_human_output_for_missing_config():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/context_roots_status.py",
            "--config",
            "config/definitely_missing_local_roots.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Context Roots Status" in result.stdout
    assert "No local roots configured." in result.stdout
