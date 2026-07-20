from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_ui_uses_saved_roots_wording() -> None:
    workspace_js = ROOT / "static" / "js" / "workspace.js"
    index_html = ROOT / "static" / "index.html"

    ui_text = ""
    if workspace_js.exists():
        ui_text += workspace_js.read_text(encoding="utf-8")
    if index_html.exists():
        ui_text += "\n" + index_html.read_text(encoding="utf-8")

    assert "Saved Roots" in ui_text or "saved roots" in ui_text
    assert "Authorized Roots" not in ui_text
    assert "No authorized roots yet" not in ui_text


def test_backend_authorized_root_internals_are_not_renamed() -> None:
    # Backend/provider internals can keep authorization terminology. This patch is UI wording only.
    expected_backend_files = [
        ROOT / "src" / "agent_runtime" / "context_vault_read_adapter.py",
        ROOT / "config" / "local_roots.json",
    ]

    existing = [p for p in expected_backend_files if p.exists()]
    assert existing

    combined = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in existing)
    assert "authorized" in combined.lower() or "allowed_capabilities" in combined
