from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent_runtime.runtime_settings import (
    DEFAULT_LANGGRAPH_RUNTIME_SETTINGS,
    LangGraphRuntimeSettingsError,
    runtime_policy,
    validate_langgraph_runtime_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def test_missing_settings_migrate_to_safe_defaults():
    config = validate_langgraph_runtime_settings(None)
    assert config == DEFAULT_LANGGRAPH_RUNTIME_SETTINGS
    assert config["permission_level"] == "guided"
    assert config["legacy_step_fallback_policy"] == "ask_first"
    assert config["interrupted_run_policy"] == "ask_before_resume"
    assert config is not DEFAULT_LANGGRAPH_RUNTIME_SETTINGS


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"permission_level": "unrestricted"}, "permission_level"),
        ({"parallel_worker_limit": 50}, "parallel_worker_limit"),
        ({"completed_run_retention": 0}, "completed_run_retention"),
        ({"unknown": True}, "Unknown LangGraph runtime setting"),
        (
            {"runtime_model_policy": {"mode": "anything"}},
            "runtime_model_policy.mode",
        ),
    ],
)
def test_invalid_backend_settings_are_rejected(patch, message):
    with pytest.raises(LangGraphRuntimeSettingsError, match=message):
        validate_langgraph_runtime_settings(patch)


def test_runtime_policy_changes_real_permission_fallback_and_recovery_gates():
    config = validate_langgraph_runtime_settings({
        "execution_mode": "direct_only",
        "permission_level": "read_only",
        "legacy_step_fallback_policy": "never",
        "interrupted_run_policy": "never_auto_resume",
        "research_approval_policy": "disabled",
        "parallel_worker_limit": 2,
        "completed_run_retention": 7,
    })
    policy = runtime_policy(config)
    assert policy["allow_multi_graph"] is False
    assert policy["allow_mutation"] is False
    assert policy["allow_legacy_step_fallback"] is False
    assert policy["allow_research_start"] is False
    assert policy["allow_safe_auto_resume"] is False
    assert policy["parallel_worker_limit"] == 2
    assert policy["completed_run_retention"] == 7


def test_settings_persist_after_cache_reset(tmp_path, monkeypatch):
    import src.settings as settings

    target = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(target))
    settings._invalidate_caches()
    configured = validate_langgraph_runtime_settings({
        "effort_preset": "thorough",
        "completed_run_retention": 90,
    })
    settings.save_settings({"langgraph_runtime": configured})
    settings._invalidate_caches()
    loaded = settings.load_settings()["langgraph_runtime"]
    assert loaded["effort_preset"] == "thorough"
    assert loaded["completed_run_retention"] == 90
    assert json.loads(target.read_text(encoding="utf-8"))[
        "langgraph_runtime"
    ] == configured


def test_conditional_runtime_panels_and_session_assignment_contract():
    source = (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")
    assert 'id="runtime-opencode-panel"' in source
    assert 'id="runtime-langgraph-panel"' in source
    assert "openCodePanel.classList.toggle('hidden', inactive)" in source
    assert "openCodePanel.hidden = inactive" in source
    assert "langGraphPanel.hidden = inactive" in source
    assert "map[id] = normalizeRuntime(_pendingRuntime, getDefaultRuntime())" in source
    assert "safeStorageSet(DEFAULT_RUNTIME_KEY, selected)" in source
    default_slice = source[
        source.index("export async function setDefaultRuntime"):
        source.index("export async function setCurrentRuntime")
    ]
    assert "setCurrentRuntime(" not in default_slice
    assert "restartOpenCodeServer()" not in source[
        source.index("export async function setDefaultRuntime"):
        source.index("export async function setCurrentRuntime")
    ]


def test_ui_controls_are_backed_by_typed_contract_and_accessible():
    source = (
        ROOT / "static" / "js" / "runtimeControl.js"
    ).read_text(encoding="utf-8")
    for setting in (
        "execution_mode",
        "permission_level",
        "runtime_model_policy",
        "legacy_step_fallback_policy",
        "interrupted_run_policy",
        "effort_preset",
        "research_approval_policy",
        "research_reuse_policy",
        "context_source_policy",
        "parallel_worker_limit",
        "task_planner_visibility",
        "completed_run_retention",
    ):
        assert setting in source
    assert 'role="switch"' in source
    assert 'aria-live="polite"' in source
    assert "data-langgraph-role" in source
    assert "(endpoint.model_type || 'llm') !== 'llm'" in source
    assert "body: JSON.stringify({ langgraph_runtime: config })" in source


def test_task_planner_visibility_is_presentation_only():
    settings_source = (
        ROOT / "src" / "agent_runtime" / "runtime_settings.py"
    ).read_text(encoding="utf-8")
    coordinator = (
        ROOT / "src" / "agent_runtime" / "multi_graph_coordinator.py"
    ).read_text(encoding="utf-8")
    assert "task_planner_visibility" in settings_source
    assert "task_planner_visibility" not in coordinator


def test_settings_are_wired_to_runtime_enforcement_seams():
    coordinator = (
        ROOT / "src" / "agent_runtime" / "multi_graph_coordinator.py"
    ).read_text(encoding="utf-8")
    chat_routes = (
        ROOT / "routes" / "chat_routes.py"
    ).read_text(encoding="utf-8")
    checkpoint = (
        ROOT / "src" / "agent_runtime" / "durable_checkpoint.py"
    ).read_text(encoding="utf-8")
    assert 'configured_policy["allow_legacy_step_fallback"]' in coordinator
    assert 'configured_policy["allow_research_start"]' in coordinator
    assert 'configured_policy["allow_mutation"]' in coordinator
    assert 'runtime_settings["context_source_policy"]["workspace"]' in coordinator
    assert 'runtime_settings["effort_preset"]' in coordinator
    assert "_langgraph_multi_graph_allowed_by_settings()" in chat_routes
    assert '["completed_run_retention"]' in checkpoint


def test_opencode_runtime_remains_a_separate_peer():
    coordinator = (
        ROOT / "src" / "agent_runtime" / "multi_graph_coordinator.py"
    ).read_text(encoding="utf-8")
    settings_source = (
        ROOT / "src" / "agent_runtime" / "runtime_settings.py"
    ).read_text(encoding="utf-8")
    assert "opencode_runtime" not in coordinator
    assert "opencode_runtime" not in settings_source
