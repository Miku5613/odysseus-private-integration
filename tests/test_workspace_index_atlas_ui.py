from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_panel_drag_is_frame_batched_and_compositor_driven():
    source = _read("static/js/windowDrag.js")

    assert "requestAnimationFrame(_paintDragTransform)" in source
    assert "translate3d(${dragDx}px, ${dragDy}px, 0)" in source
    assert "content.style.willChange = 'transform'" in source
    assert "_commitDragTransform();" in source
    windowed_move = source.split("// Windowed: just follow the cursor.", 1)[1].split(
        "// Corner guard:", 1
    )[0]
    assert "content.style.left" not in windowed_move
    assert "content.style.top" not in windowed_move


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
    assert "Embedding size:" in indexes
    assert "embedding_size_mb" in indexes
    assert "findIndexForPath" in atlas
    assert "This root has a searchable index" in atlas
    assert "data-menu-trigger=\"view\"" in atlas
    assert "data-menu-trigger=\"more\"" in atlas
    assert "data-el=\"workspace-tree\"" in atlas
    assert "file tree is not implemented" not in atlas


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


def test_changed_javascript_parses_and_has_no_nul_or_mojibake():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    suspicious = ("Ãƒ", "Ã‚", "Ã¢â‚¬", "Ã¢â€", "ï¿½", "\x00")
    for relative in (
        "static/js/atlas.js",
        "static/js/indexStatus.js",
        "static/js/indexes.js",
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
