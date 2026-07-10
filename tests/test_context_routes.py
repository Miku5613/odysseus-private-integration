
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.context_routes import router


def _client(tmp_path: Path, monkeypatch):
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
                    },
                    {
                        "id": "disabled_root",
                        "provider_type": "local_dir",
                        "display_name": "Disabled Root",
                        "local_path": str(root),
                        "allowed_capabilities": ["context.list"],
                        "safety_scope": "read_only_sandbox",
                        "enabled": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ODYSSEUS_CONTEXT_ROOTS_CONFIG", str(config))

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_context_roots_endpoint_lists_public_roots(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/context/roots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["root_count"] == 2
    root = payload["roots"][0]
    assert root["id"] == "test_root"
    assert "local_path" not in root
    assert root["adapter_status"]["status"] == "ok"


def test_context_list_endpoint_lists_root_items(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/context/list",
        json={"root_id": "test_root", "selector": "."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["root"]["id"] == "test_root"
    assert {item["path"] for item in payload["items"]} == {"docs", "README.md"}


def test_context_read_endpoint_reads_text(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/context/read",
        json={"root_id": "test_root", "selector": "README.md"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selector"] == "README.md"
    assert payload["text"] == "alpha beta gamma"
    assert payload["characters"] == len("alpha beta gamma")


def test_context_search_endpoint_searches_text(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/context/search",
        json={"root_id": "test_root", "query": "needle"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["path"] == "docs/note.md"


def test_context_read_blocks_traversal(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/context/read",
        json={"root_id": "test_root", "selector": "../secret.txt"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "ValueError"


def test_context_missing_root_returns_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/context/list",
        json={"root_id": "missing", "selector": "."},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "root_not_found"


def test_context_disabled_root_returns_403(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/context/list",
        json={"root_id": "disabled_root", "selector": "."},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "root_disabled"
