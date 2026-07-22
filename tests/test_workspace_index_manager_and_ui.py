from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_indexes_settings_ui_is_admin_only_and_has_svg_icon():
    html = _read("static/index.html")
    button_start = html.index(
        '<button class="settings-nav-item admin-only" data-settings-tab="indexes">'
    )
    button_end = html.index("</button>", button_start)
    button = html[button_start:button_end]

    assert "<svg" in button
    assert "<span>Indexes</span>" in button
    assert 'data-settings-panel="indexes" class="hidden admin-only"' in html
    assert 'data-settings-panel="indexes" style="display:none"' not in html
    assert 'id="indexes-use-active-btn"' in html
    assert 'id="indexes-workspace-path"' in html
    assert 'id="indexes-root-id"' in html
    assert '<script type="module" src="/static/js/indexes.js"></script>' in html


def test_indexes_frontend_uses_active_workspace_and_safe_dom_rendering():
    source = _read("static/js/indexes.js")

    assert "import { getWorkspace } from './workspace.js';" in source
    assert "getWorkspace()" in source
    assert "textContent" in source
    assert "encodeURIComponent(rootId)" in source
    assert "nomic-embed-text:latest" in source
    assert "window.confirm" in source
    assert 'data-index-action="toggle"' in source
    assert 'class="indexes-root-details" hidden' in source
    assert 'aria-expanded="false"' in source
    assert "Show index details" in source

    settings = _read("static/js/settings.js")
    assert "'indexes'" in settings.split("const ADMIN_TABS", 1)[1].split(";", 1)[0]


def test_indexes_controls_are_compact_aligned_and_described():
    html = _read("static/index.html")
    css = _read("static/style.css")

    refresh_start = html.index('id="indexes-refresh-btn"')
    refresh_end = html.index("</button>", refresh_start)
    refresh_button = html[refresh_start:refresh_end]

    assert "<svg" in refresh_button
    assert ">Refresh<" not in refresh_button
    assert 'title="Reload the saved workspace indexes"' in refresh_button
    assert 'aria-label="Reload saved indexes"' in refresh_button
    assert 'class="indexes-card-title"' in html
    assert 'id="indexes-check-stale-btn"' in html
    assert 'title="List saved indexes older than the selected period"' in html
    assert ".indexes-card-title" in css
    assert ".indexes-root-details[hidden]" in css
    assert '.indexes-expand-button[aria-expanded="true"]' in css


def test_long_running_index_operations_bypass_global_request_timeout():
    app = _read("app.py")
    timeout_block = app.split("_TIMEOUT_EXEMPT_PREFIXES = (", 1)[1].split(
        "class _RequestTimeoutMiddleware", 1
    )[0]

    assert '"/api/indexes/index"' in timeout_block
    assert '"/api/indexes/reindex"' in timeout_block


def test_indexes_javascript_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    completed = subprocess.run(
        [node, "--check", str(ROOT / "static/js/indexes.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_index_feature_python_files_parse():
    for relative in (
        "app.py",
        "routes/index_routes.py",
        "src/agent_runtime/knowledge_vault_index.py",
        "src/agent_runtime/knowledge_vault_semantic.py",
        "src/agent_runtime/workspace_index_manager.py",
    ):
        ast.parse(_read(relative), filename=relative)


def test_index_feature_files_have_no_mojibake():
    suspicious = ("Ã", "Â", "â€", "â†", "â•", "ï»¿", "�")
    for relative in (
        "app.py",
        "static/index.html",
        "static/js/indexes.js",
        "routes/index_routes.py",
        "src/agent_runtime/workspace_index_manager.py",
    ):
        source = _read(relative)
        found = [token for token in suspicious if token in source]
        assert not found, (relative, found)
        assert not source.startswith("\ufeff"), f"UTF-8 BOM: {relative}"


def test_indexes_backend_registered_once():
    app = _read("app.py")
    assert app.count("from routes.index_routes import setup_index_routes") == 1
    assert app.count("app.include_router(setup_index_routes())") == 1


def test_indexes_route_surface_is_complete():
    from routes.index_routes import setup_index_routes

    surface = {
        (route.path, method)
        for route in setup_index_routes().routes
        for method in (route.methods or set())
    }
    assert ("/api/indexes", "GET") in surface
    assert ("/api/indexes/suggest-root-id", "GET") in surface
    assert ("/api/indexes/register", "POST") in surface
    assert ("/api/indexes/index", "POST") in surface
    assert ("/api/indexes/reindex", "POST") in surface
    assert ("/api/indexes/stale", "GET") in surface
    assert ("/api/indexes/{root_id}/stats", "GET") in surface
    assert ("/api/indexes/{root_id}/index", "DELETE") in surface
    assert ("/api/indexes/{root_id}", "DELETE") in surface


def test_definitive_root_id_is_stable_and_path_specific(tmp_path):
    from src.agent_runtime import workspace_index_manager as manager

    first = tmp_path / "Notes"
    second = tmp_path / "Other" / "Notes"
    first.mkdir()
    second.mkdir(parents=True)

    first_id = manager.suggest_root_id(str(first))
    assert first_id == manager.suggest_root_id(str(first))
    assert first_id.startswith("workspace_notes_")
    assert first_id != manager.suggest_root_id(str(second))
    assert len(first_id) <= 128


def test_register_root_persists_authorized_structure(tmp_path, monkeypatch):
    from src.agent_runtime import workspace_index_manager as manager

    workspace = tmp_path / "Daily Notes"
    workspace.mkdir()
    roots_file = tmp_path / "local_roots.json"
    monkeypatch.setattr(manager, "ROOTS_FILE", roots_file)

    root_id = manager.suggest_root_id(str(workspace))
    result = manager.register_root(root_id, str(workspace))
    stored = json.loads(roots_file.read_text(encoding="utf-8"))["roots"]

    assert result["id"] == root_id
    assert result["root_id"] == root_id
    assert result["index_namespace"] == root_id
    assert result["associated_index_id"] == root_id
    assert result["provider_type"] == "local_dir"
    assert result["safety_scope"] == "read_only_sandbox"
    assert result["allowed_capabilities"] == ["context.list", "context.read"]
    assert result["embedding_model"] == "nomic-embed-text:latest"
    assert stored == [result]

    with pytest.raises(ValueError, match="definitive ID"):
        manager.register_root("different_root", str(workspace))


def test_index_root_uses_explicit_registry_namespace_and_nomic(
    tmp_path,
    monkeypatch,
):
    from src.agent_runtime import workspace_index_manager as manager

    workspace = tmp_path / "Knowledge"
    workspace.mkdir()
    (workspace / "note.md").write_text("# Dentist\nFriday at 10.", encoding="utf-8")
    roots_file = tmp_path / "local_roots.json"
    monkeypatch.setattr(manager, "ROOTS_FILE", roots_file)

    root_id = manager.suggest_root_id(str(workspace))
    manager.register_root(root_id, str(workspace))
    captured: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, *, root_id, config_path):
            captured["adapter_root_id"] = root_id
            captured["config_path"] = config_path

        def list_dir(self, selector=""):
            captured["probe_selector"] = selector
            return ["note.md"]

    def fake_structural(**kwargs):
        captured["structural"] = kwargs
        return SimpleNamespace(note_count=1, notes=[object()])

    class FakeStats:
        embedding_model = "nomic-embed-text:latest"
        note_count = 1
        chunk_count = 1

        def to_dict(self):
            return {
                "embedding_model": self.embedding_model,
                "note_count": self.note_count,
                "chunk_count": self.chunk_count,
            }

    def fake_semantic(**kwargs):
        captured["semantic"] = kwargs
        return FakeStats()

    monkeypatch.setattr(manager, "ContextVaultReadAdapter", FakeAdapter)
    monkeypatch.setattr(manager, "build_structural_vault_index", fake_structural)
    monkeypatch.setattr(manager, "build_semantic_vault_index", fake_semantic)

    result = manager.index_root(root_id, rebuild=True)
    semantic = captured["semantic"]

    assert captured["adapter_root_id"] == root_id
    assert captured["config_path"] == roots_file
    assert captured["probe_selector"] == ""
    assert semantic["index_namespace"] == root_id
    assert semantic["embedding_model"] == "nomic-embed-text:latest"
    assert semantic["strict_reads"] is True
    assert result["stats"]["chunk_count"] == 1


def test_structural_workspace_index_accepts_supported_text_files():
    from src.agent_runtime.knowledge_vault_index import build_structural_vault_index

    class FakeAdapter:
        def list_dir(self, selector=""):
            if not selector:
                return ["note.md", "settings.json", "image.png"]
            return []

        def read_note(self, selector):
            return f"content for {selector}"

    index = build_structural_vault_index(
        adapter=FakeAdapter(),
        cache_path=None,
        file_extensions=frozenset({".md", ".json"}),
    )

    assert index.note_count == 2
    assert {note.path for note in index.notes} == {"note.md", "settings.json"}


def test_workspace_knowledge_worker_has_no_global_root_fallback():
    worker = _read("src/agent_runtime/knowledge_worker.py")

    assert "find_root_by_path(workspace)" in worker
    assert "search_root(" in worker
    assert 'or "ideaverse_vault"' not in worker
