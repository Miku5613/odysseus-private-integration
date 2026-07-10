
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.context_routes import router as context_router
from src.context_providers.runtime_tool import RuntimeContextToolError, context_tool_call




def _context_api_routes():
    return [
        route
        for route in context_router.routes
        if getattr(route, "path", "").startswith("/api/context")
    ]

def _write_config(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("read only contract", encoding="utf-8")

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


def test_context_api_exposes_only_read_only_routes():
    app = FastAPI()
    app.include_router(context_router)

    route_methods = {
        (route.path, tuple(sorted(route.methods or [])))
        for route in _context_api_routes()
    }

    assert route_methods == {
        ("/api/context/roots", ("GET",)),
        ("/api/context/list", ("POST",)),
        ("/api/context/read", ("POST",)),
        ("/api/context/search", ("POST",)),
    }


def test_context_api_has_no_write_like_endpoint_names():
    app = FastAPI()
    app.include_router(context_router)

    forbidden_words = {
        "write",
        "create",
        "update",
        "delete",
        "remove",
        "move",
        "rename",
        "mkdir",
        "touch",
        "save",
        "apply",
    }

    context_paths = [
        route.path.lower()
        for route in _context_api_routes()
    ]

    assert context_paths
    for path in context_paths:
        assert not any(word in path for word in forbidden_words), path


def test_context_api_write_like_paths_are_not_registered():
    app = FastAPI()
    app.include_router(context_router)
    client = TestClient(app)

    for path in [
        "/api/context/write",
        "/api/context/create",
        "/api/context/update",
        "/api/context/delete",
        "/api/context/move",
        "/api/context/rename",
        "/api/context/save",
    ]:
        response = client.post(path, json={})
        assert response.status_code == 404


@pytest.mark.parametrize(
    "operation",
    [
        "write",
        "create",
        "update",
        "delete",
        "remove",
        "move",
        "rename",
        "mkdir",
        "touch",
        "save",
        "apply",
    ],
)
def test_runtime_context_tool_rejects_write_like_operations(tmp_path, operation):
    config = _write_config(tmp_path)

    with pytest.raises(RuntimeContextToolError):
        context_tool_call(
            operation=operation,
            root_id="vault",
            selector="note.md",
            query="contract",
            config_path=config,
        )


def test_runtime_context_tool_supported_operations_remain_read_only(tmp_path):
    config = _write_config(tmp_path)

    roots = context_tool_call(operation="roots", config_path=config)
    listed = context_tool_call(
        operation="list",
        root_id="vault",
        selector=".",
        config_path=config,
    )
    read = context_tool_call(
        operation="read",
        root_id="vault",
        selector="note.md",
        config_path=config,
    )
    search = context_tool_call(
        operation="search",
        root_id="vault",
        query="contract",
        config_path=config,
    )
    search_read = context_tool_call(
        operation="search_read",
        root_id="vault",
        query="contract",
        config_path=config,
    )

    assert roots["operation"] == "roots"
    assert listed["operation"] == "list"
    assert read["operation"] == "read"
    assert search["operation"] == "search"
    assert search_read["operation"] == "search_read"

    assert read["text"] == "read only contract"
    assert search["count"] == 1
    assert search_read["document_count"] == 1
