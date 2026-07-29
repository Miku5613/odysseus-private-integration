from __future__ import annotations

from copy import deepcopy
from typing import Any


LANGGRAPH_RUNTIME_SETTINGS_VERSION = 1
LANGGRAPH_RUNTIME_SETTINGS_SCOPE = "global_admin"

DEFAULT_LANGGRAPH_RUNTIME_SETTINGS: dict[str, Any] = {
    "version": LANGGRAPH_RUNTIME_SETTINGS_VERSION,
    "execution_mode": "automatic",
    "permission_level": "guided",
    "runtime_model_policy": {
        "mode": "single_compatible",
        "model": "",
        "role_models": {},
    },
    "legacy_step_fallback_policy": "ask_first",
    "interrupted_run_policy": "ask_before_resume",
    "effort_preset": "balanced",
    "research_approval_policy": "ask_before_long_research",
    "research_reuse_policy": "prefer_existing",
    "context_source_policy": {
        "conversation": True,
        "workspace": True,
        "indexed_knowledge": True,
        "completed_research": True,
        "personal_memory": False,
    },
    "parallel_worker_limit": "automatic",
    "task_planner_visibility": "show",
    "completed_run_retention": 30,
}

_CHOICES = {
    "execution_mode": {
        "automatic", "direct_only", "multi_step_allowed",
        "ask_before_multi_step",
    },
    "permission_level": {"read_only", "guided", "active"},
    "legacy_step_fallback_policy": {"never", "ask_first", "allow_bounded"},
    "interrupted_run_policy": {
        "ask_before_resume", "never_auto_resume",
        "auto_resume_safe_read_only",
    },
    "effort_preset": {"quick", "balanced", "thorough", "custom"},
    "research_approval_policy": {
        "ask_before_long_research", "allow_within_budget", "disabled",
    },
    "research_reuse_policy": {
        "prefer_existing", "ask", "always_new_when_requested",
    },
    "task_planner_visibility": {"show", "show_collapsed", "hidden"},
}
_CONTEXT_KEYS = frozenset(
    DEFAULT_LANGGRAPH_RUNTIME_SETTINGS["context_source_policy"]
)
_MODEL_ROLES = frozenset({
    "planner", "general_worker", "knowledge_worker", "code_worker",
    "research_worker", "final_synthesis",
})


class LangGraphRuntimeSettingsError(ValueError):
    pass


def validate_langgraph_runtime_settings(
    value: Any,
) -> dict[str, Any]:
    if value is None:
        return deepcopy(DEFAULT_LANGGRAPH_RUNTIME_SETTINGS)
    if not isinstance(value, dict):
        raise LangGraphRuntimeSettingsError(
            "langgraph_runtime must be an object"
        )
    unknown = set(value) - set(DEFAULT_LANGGRAPH_RUNTIME_SETTINGS)
    if unknown:
        raise LangGraphRuntimeSettingsError(
            f"Unknown LangGraph runtime setting: {sorted(unknown)[0]}"
        )
    result = deepcopy(DEFAULT_LANGGRAPH_RUNTIME_SETTINGS)
    result.update(value)
    for key, choices in _CHOICES.items():
        if result[key] not in choices:
            raise LangGraphRuntimeSettingsError(
                f"{key} must be one of: {', '.join(sorted(choices))}"
            )

    parallel = result["parallel_worker_limit"]
    if parallel != "automatic":
        if isinstance(parallel, bool):
            raise LangGraphRuntimeSettingsError(
                "parallel_worker_limit must be automatic or an integer"
            )
        try:
            parallel = int(parallel)
        except (TypeError, ValueError) as exc:
            raise LangGraphRuntimeSettingsError(
                "parallel_worker_limit must be automatic or an integer"
            ) from exc
        if not 1 <= parallel <= 4:
            raise LangGraphRuntimeSettingsError(
                "parallel_worker_limit must be between 1 and 4"
            )
        result["parallel_worker_limit"] = parallel

    try:
        retention = int(result["completed_run_retention"])
    except (TypeError, ValueError) as exc:
        raise LangGraphRuntimeSettingsError(
            "completed_run_retention must be an integer"
        ) from exc
    if not 1 <= retention <= 365:
        raise LangGraphRuntimeSettingsError(
            "completed_run_retention must be between 1 and 365 days"
        )
    result["completed_run_retention"] = retention

    context = result["context_source_policy"]
    if not isinstance(context, dict) or set(context) - _CONTEXT_KEYS:
        raise LangGraphRuntimeSettingsError(
            "context_source_policy contains unsupported sources"
        )
    merged_context = dict(
        DEFAULT_LANGGRAPH_RUNTIME_SETTINGS["context_source_policy"]
    )
    for key, enabled in context.items():
        if not isinstance(enabled, bool):
            raise LangGraphRuntimeSettingsError(
                f"context_source_policy.{key} must be boolean"
            )
        merged_context[key] = enabled
    result["context_source_policy"] = merged_context

    model_policy = result["runtime_model_policy"]
    if not isinstance(model_policy, dict):
        raise LangGraphRuntimeSettingsError(
            "runtime_model_policy must be an object"
        )
    mode = model_policy.get("mode", "single_compatible")
    if mode not in {"single_compatible", "advanced_roles"}:
        raise LangGraphRuntimeSettingsError(
            "runtime_model_policy.mode is unsupported"
        )
    model = str(model_policy.get("model") or "").strip()
    role_models = model_policy.get("role_models") or {}
    if not isinstance(role_models, dict) or set(role_models) - _MODEL_ROLES:
        raise LangGraphRuntimeSettingsError(
            "runtime_model_policy.role_models contains unsupported roles"
        )
    result["runtime_model_policy"] = {
        "mode": mode,
        "model": model,
        "role_models": {
            key: str(item or "").strip()
            for key, item in role_models.items()
            if str(item or "").strip()
        },
    }
    result["version"] = LANGGRAPH_RUNTIME_SETTINGS_VERSION
    return result


def get_langgraph_runtime_settings() -> dict[str, Any]:
    from src.settings import get_setting

    return validate_langgraph_runtime_settings(
        get_setting("langgraph_runtime", None)
    )


def runtime_policy(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    config = validate_langgraph_runtime_settings(settings)
    return {
        "allow_multi_graph": config["execution_mode"] in {
            "automatic", "multi_step_allowed",
        },
        "ask_before_multi_graph": (
            config["execution_mode"] == "ask_before_multi_step"
        ),
        "allow_mutation": config["permission_level"] == "active",
        "require_mutation_approval": config["permission_level"] == "guided",
        "allow_legacy_step_fallback": (
            config["legacy_step_fallback_policy"] != "never"
        ),
        "require_legacy_fallback_approval": (
            config["legacy_step_fallback_policy"] == "ask_first"
        ),
        "allow_research_start": (
            config["research_approval_policy"] != "disabled"
        ),
        "require_research_approval": (
            config["research_approval_policy"]
            == "ask_before_long_research"
        ),
        "allow_safe_auto_resume": (
            config["interrupted_run_policy"]
            == "auto_resume_safe_read_only"
        ),
        "parallel_worker_limit": (
            1
            if config["parallel_worker_limit"] == "automatic"
            else config["parallel_worker_limit"]
        ),
        "completed_run_retention": config["completed_run_retention"],
    }


__all__ = [
    "DEFAULT_LANGGRAPH_RUNTIME_SETTINGS",
    "LANGGRAPH_RUNTIME_SETTINGS_SCOPE",
    "LangGraphRuntimeSettingsError",
    "get_langgraph_runtime_settings",
    "runtime_policy",
    "validate_langgraph_runtime_settings",
]
