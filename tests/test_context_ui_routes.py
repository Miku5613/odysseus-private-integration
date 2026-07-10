
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.context_ui_routes import router


def test_context_roots_page_renders_manual_inspector():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/context-roots")

    assert response.status_code == 200
    assert "Context Roots Inspector" in response.text
    assert "/context-roots.js" in response.text
    assert "does not call the model" in response.text


def test_context_roots_script_serves_javascript():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/context-roots.js")

    assert response.status_code == 200
    assert "application/javascript" in response.headers["content-type"]
    assert "window.contextRootsInspector" in response.text
    assert "loadRoots" in response.text
