from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_panel_drag_matches_original_pre_optimization_logic():
    source = _read("static/js/windowDrag.js")

    assert "requestAnimationFrame(_paintDragTransform)" not in source
    assert "translate3d(${dragDx}px, ${dragDy}px, 0)" not in source
    assert "content.style.willChange = 'transform'" not in source
    windowed_move = source.split("// Windowed: just follow the cursor.", 1)[1].split(
        "// Corner guard:", 1
    )[0]
    assert "content.style.left = (startLeft + cx - startX) + 'px'" in windowed_move
    assert "content.style.top = (startTop + cy - startY) + 'px'" in windowed_move


def test_saved_roots_and_atlas_index_affordances_are_present():
    workspace = _read("static/js/workspace.js")
    atlas = _read("static/js/atlas.js")
    indexes = _read("static/js/indexes.js")
    css = _read("static/style.css")

    assert "loadIndexRoots({ force: true })" in workspace
    assert "workspace-saved-roots-list" in workspace
    assert "This saved root has a searchable index" in workspace
    assert ".workspace-saved-roots-list" in css
    assert "overflow-y: auto" in css
    assert "overflow-x: hidden" in css
    assert ".workspace-saved-roots-list::-webkit-scrollbar { width: 5px; height: 0; }" in css
    assert "text-overflow: ellipsis" in css
    assert "`${r.label || _basename(r.path)} — ${r.path}`" in workspace
    assert "Embedding size:" in indexes
    assert "embedding_size_mb" in indexes
    assert "findIndexForPath" in atlas
    assert "This root has a searchable index" in atlas
    assert "data-menu-trigger=\"view\"" in atlas
    assert "data-menu-trigger=\"more\"" in atlas
    assert "data-el=\"workspace-tree\"" in atlas
    assert "openWorkspaceNote(entry.path)" in atlas
    assert "data-el=\"workspace-tree-resizer\"" not in atlas
    assert "WORKSPACE_TREE_HEIGHT_KEY" not in atlas
    assert ".atlas-root-section { flex:1 1 auto; min-height:0; display:flex" in atlas
    assert ".atlas-worktree { margin-top:7px; flex:1 1 auto; min-height:0" in atlas
    assert ".atlas-editor { display:flex; flex-direction:column; overflow:hidden" in atlas
    assert "height:100%; min-height:0; max-height:none; box-sizing:border-box; overflow:auto" in atlas
    assert "_setAtlasPane('edit');" not in atlas
    assert "workspaceApiGet('/graph', { root })" in atlas
    assert "_loadWorkspaceBacklinks(_currentWorkspaceRoot, data.path)" in atlas
    assert "Session ${sessionCount} · Root ${rootCount}" in atlas
    assert "_currentSource === 'workspace'" in atlas
    assert "fetch('/api/workspace/file'" in atlas
    assert 'class="atlas-btn atlas-refresh-btn"' in atlas
    assert "<span>Refresh</span>" in atlas
    assert "#atlas-modal .modal-content::after { display:none; }" in atlas
    assert "font-size:13px" in atlas
    assert "file tree is not implemented" not in atlas


def test_dock_resize_seam_stays_active_without_covering_floating_controls():
    source = _read("static/js/modalSnap.js")
    css = _read("static/style.css")

    assert "handle.style.width = '4px'" in source
    assert "_setStyle(handle, 'left', (x - 2) + 'px')" in source
    assert "handle.setAttribute('aria-label', 'Drag to resize docked window')" in source
    assert "handle.title = 'Drag to resize docked window'" not in source
    assert "transparent 3px 4px" in css


def test_workspace_tree_is_sorted_confined_and_skips_hidden(tmp_path, monkeypatch):
    from routes import workspace_routes
    from src import tool_execution

    root = tmp_path / "Vault"
    root.mkdir()
    (root / "B-folder").mkdir()
    (root / "a-folder").mkdir()
    (root / ".hidden").mkdir()
    (root / "z.md").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(tool_execution, "vet_workspace", lambda path: str(root))

    result = workspace_routes._workspace_tree_entries(str(root))

    assert [item["name"] for item in result["entries"]] == [
        "a-folder", "B-folder", "z.md"
    ]
    assert result["entries"][-1]["kind"] == "file"
    assert result["entries"][-1]["size"] == 5

    monkeypatch.setattr(tool_execution, "vet_workspace", lambda path: str(root))
    with pytest.raises(ValueError, match="escapes"):
        workspace_routes._resolve_workspace_tree_path(str(root), "../outside")


def test_workspace_tree_route_is_registered():
    from routes.workspace_routes import setup_workspace_routes

    surface = {
        (route.path, method)
        for route in setup_workspace_routes().routes
        for method in (route.methods or set())
    }
    assert ("/api/workspace/tree", "GET") in surface
    assert ("/api/workspace/file", "GET") in surface
    assert ("/api/workspace/file", "PUT") in surface
    assert ("/api/workspace/graph", "GET") in surface
    assert ("/api/workspace/backlinks", "GET") in surface


def test_workspace_note_read_write_is_confined(tmp_path, monkeypatch):
    from routes import workspace_routes
    from src import tool_execution

    root = tmp_path / "Vault"
    root.mkdir()
    note = root / "note.md"
    note.write_text("# Original\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"png")
    monkeypatch.setattr(tool_execution, "vet_workspace", lambda path: str(root))

    loaded = workspace_routes._read_workspace_note(str(root), "note.md")
    assert loaded["path"] == "note.md"
    assert loaded["content"] == "# Original\n"

    saved = workspace_routes._write_workspace_note(
        str(root), "note.md", "# Edited in Atlas\n"
    )
    assert saved["saved"] is True
    assert note.read_text(encoding="utf-8") == "# Edited in Atlas\n"

    with pytest.raises(ValueError, match="escapes"):
        workspace_routes._read_workspace_note(str(root), "../outside.md")
    with pytest.raises(ValueError, match="Markdown or text"):
        workspace_routes._read_workspace_note(str(root), "image.png")


def test_workspace_graph_and_backlinks_include_root_notes(tmp_path, monkeypatch):
    from routes import workspace_routes
    from src import tool_execution

    root = tmp_path / "Vault"
    root.mkdir()
    (root / "Projects").mkdir()
    (root / "Projects" / "Alpha.md").write_text(
        "# Alpha\n\nLinks to [[Beta]].\n", encoding="utf-8"
    )
    (root / "Beta.md").write_text("# Beta\n", encoding="utf-8")
    (root / "venv").mkdir()
    (root / "venv" / "ignored.md").write_text("# Ignore\n", encoding="utf-8")
    monkeypatch.setattr(tool_execution, "vet_workspace", lambda path: str(root))

    graph = workspace_routes._workspace_note_graph(str(root))
    assert {node["id"] for node in graph["nodes"]} == {
        "Beta.md", "Projects/Alpha.md"
    }
    assert graph["links"] == [
        {"source": "Projects/Alpha.md", "target": "Beta.md"}
    ]
    assert graph["note_count"] == 2
    assert graph["truncated"] is False

    backlinks = workspace_routes._workspace_note_backlinks(str(root), "Beta.md")
    assert backlinks["backlinks"] == [
        {"path": "Projects/Alpha.md", "title": "Alpha"}
    ]


def test_docked_resize_handle_stays_available_with_floating_panels():
    source = _read("static/js/modalSnap.js")

    assert "_dockHandleZIndex(owner)" in source
    assert "_hasVisibleFloatingModal(owner)" not in source
    assert "opening a second tool never disables resizing the docked one" in source


def test_changed_javascript_parses_and_has_no_nul_or_mojibake():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    suspicious = ("Ãƒ", "Ã‚", "Ã¢â‚¬", "Ã¢â€", "ï¿½", "\x00")
    for relative in (
        "static/js/atlas.js",
        "static/js/indexStatus.js",
        "static/js/indexes.js",
        "static/js/modalSnap.js",
        "static/js/windowDrag.js",
        "static/js/workspace.js",
    ):
        source = _read(relative)
        assert not [token for token in suspicious if token in source], relative
        completed = subprocess.run(
            [node, "--check", str(ROOT / relative)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
