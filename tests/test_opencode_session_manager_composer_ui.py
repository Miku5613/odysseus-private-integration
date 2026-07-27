from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_composer_button_prepares_then_opens_terminal_session_manager():
    source = _source("static/js/runtimeControl.js")
    assert "function handleOpenCodeSessionManagerClick()" in source
    assert "prepareOpenCodeForCurrentChat()" in source
    assert "/api/opencode/runtime/prepare" in source
    assert "/api/opencode/runtime/server/session-manager-terminal" in source
    assert "sessionModule.materializePendingSession()" in source
    assert "getWorkspace()" in source
    assert "OpenCode Session Manager" in source
    assert "runtime-quick-reset-btn" in source
    assert "Start OpenCode or open OpenCode Session Manager" in source


def test_composer_ready_state_is_bound_to_chat_model_and_workspace():
    source = _source("static/js/runtimeControl.js")
    assert "openCodeReadyKey(sessionId, modelId, workspace)" in source
    assert "_openCodeReadySessions" in source
    assert "currentOpenCodeReadyState()" in source
    assert "modelId" in source
    assert "workspace" in source
    assert "classList.toggle(" in source
    assert "'is-ready'" in source




def test_prepare_persists_deterministic_ready_message():
    source = _source("src/opencode_runtime/service.py")
    assert 'kind": "runtime_ready"' in source
    assert '"deterministic": True' in source
    assert '"message": ready_message' in source
    assert "self._persist_transcript_message(" in source

def test_composer_controls_match_native_size_and_backend_ready_highlight():
    runtime_js = _source("static/js/runtimeControl.js")
    style = _source("static/style.css")

    assert "_indicator.classList.toggle(" in runtime_js
    assert "'active'," in runtime_js
    assert "runtime === 'opencode'" in runtime_js
    assert "_quickResetButton.classList.toggle('active', ready)" in runtime_js
    assert ".runtime-indicator-btn," in style
    assert ".runtime-quick-reset-btn {" in style
    assert "height: 32px !important" in style
    assert "width: 32px !important" in style

def test_prepare_uses_pending_model_identity_and_one_inflight_promise():
    source = _source("static/js/runtimeControl.js")
    assert "sessionModule.getPendingChat" in source
    assert "pending?.modelId" in source
    assert "pending?.endpointId" in source
    assert "activeOpenCodeSelection()" in source
    assert "let _openCodePreparePromise = null" in source
    assert "if (_openCodePreparePromise) return _openCodePreparePromise" in source


def test_prepare_failure_cards_are_deduplicated():
    source = _source("static/js/runtimeControl.js")
    assert "let _lastOpenCodeFailure" in source
    assert "now - _lastOpenCodeFailure.timestamp < 5000" in source
