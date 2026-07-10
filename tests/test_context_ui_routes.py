
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
    assert "/api/context/roots" in response.text
    assert "/api/context/list" in response.text
    assert "/api/context/read" in response.text
    assert "/api/context/search" in response.text
    assert "does not call the model" in response.text
