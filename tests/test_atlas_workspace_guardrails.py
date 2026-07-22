from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_static(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def test_atlas_ui_displays_active_workspace_root_card() -> None:
    atlas_js = _read_static("static/js/atlas.js")

    assert "function _activeWorkspaceRoot()" in atlas_js
    assert "getWorkspace" in atlas_js
    assert "atlas-root-card" in atlas_js
    assert "Workspace is linked" in atlas_js
    assert "Open Markdown or text notes below to view and edit them in Atlas" in atlas_js
    assert "Atlas file tree is not implemented yet" not in atlas_js
    assert "clear Workspace to switch Atlas roots or import" in atlas_js


def test_atlas_root_switching_is_blocked_when_workspace_is_active() -> None:
    atlas_js = _read_static("static/js/atlas.js")

    assert "_activeWorkspaceRoot()" in atlas_js
    assert "Can't switch Atlas root while an active Workspace is selected" in atlas_js
    assert "Clear Workspace first" in atlas_js
    assert "_syncAtlasWorkspaceTitle()" in atlas_js
    assert "return;" in atlas_js


def test_atlas_import_is_disabled_when_workspace_is_active() -> None:
    atlas_js = _read_static("static/js/atlas.js")

    assert "_activeWorkspaceRoot()" in atlas_js
    assert "Import is disabled while an active Workspace root is selected" in atlas_js
    assert "importBtn.classList.toggle('disabled', !!root)" in atlas_js


def test_atlas_workspace_status_syncs_on_overflow_changes() -> None:
    atlas_js = _read_static("static/js/atlas.js")

    assert "function _syncAtlasWorkspaceTitle()" in atlas_js
    assert "document.addEventListener('overflow-state-change', _syncAtlasWorkspaceTitle)" in atlas_js
    assert "document.removeEventListener('overflow-state-change', _syncAtlasWorkspaceTitle)" in atlas_js


def test_workspace_ui_keeps_saved_roots_wording() -> None:
    workspace_js = _read_static("static/js/workspace.js")

    assert "Saved Roots" in workspace_js
    assert "No saved roots yet" in workspace_js
    assert "Authorized Roots" not in workspace_js
    assert "No authorized roots yet" not in workspace_js


def test_backend_authorized_context_root_names_are_preserved() -> None:
    adapter = _read_static("src/agent_runtime/context_vault_read_adapter.py")
    config = _read_static("config/local_roots.json")

    assert "authorized context-provider vault root" in adapter
    assert "authorized root" in adapter
    assert "allowed_capabilities" in config
