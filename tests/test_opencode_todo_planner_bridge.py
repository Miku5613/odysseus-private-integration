from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "static" / "js" / "openCodeTaskPlannerAdapter.js"
TRACKER = ROOT / "static" / "js" / "multiGraphTracker.js"


def _run_adapter_node(
    tmp_path: Path,
    script: str,
) -> subprocess.CompletedProcess:
    (tmp_path / "multiGraphTracker.mjs").write_text(
        TRACKER.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    adapter_source = ADAPTER.read_text(encoding="utf-8").replace(
        "./multiGraphTracker.js",
        "./multiGraphTracker.mjs",
    )
    (tmp_path / "openCodeTaskPlannerAdapter.mjs").write_text(
        adapter_source,
        encoding="utf-8",
    )
    return subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_normalizer_maps_native_todos_to_plan_steps() -> None:
    from src.opencode_runtime.normalizer import OpenCodeEventNormalizer

    normalizer = OpenCodeEventNormalizer("ses-todo")
    events = normalizer.normalize({
        "type": "todo.updated",
        "properties": {
            "sessionID": "ses-todo",
            "todos": [
                {
                    "id": "inspect",
                    "content": "Inspect runtime routes",
                    "status": "completed",
                    "priority": "high",
                },
                {
                    "id": "service",
                    "content": "Trace service calls",
                    "status": "in_progress",
                    "priority": "medium",
                },
                {
                    "id": "report",
                    "content": "Prepare cited report",
                    "status": "pending",
                    "priority": "low",
                },
            ],
        },
    })

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "runtime.plan.updated"
    assert [step["id"] for step in event["steps"]] == [
        "inspect",
        "service",
        "report",
    ]
    assert [step["status"] for step in event["steps"]] == [
        "completed",
        "running",
        "pending",
    ]
    assert event["steps"][1]["objective"] == "Trace service calls"


def test_adapter_uses_call_id_for_distinct_fallback_tools(
    tmp_path: Path,
) -> None:
    result = _run_adapter_node(
        tmp_path,
        r"""
        import {
          getAdapterStateSnapshot,
          resetAdapterStateForTests,
          translateOpenCodeRuntimeEvent,
        } from "./openCodeTaskPlannerAdapter.mjs";

        resetAdapterStateForTests();
        translateOpenCodeRuntimeEvent({
          type: "runtime.started",
          run_id: "run-fallback",
          objective: "Fallback tool plan",
        });

        for (const [callId, path] of [
          ["call-routes", "routes/opencode_runtime_routes.py"],
          ["call-service", "src/opencode_runtime/service.py"],
          ["call-manager", "src/opencode_runtime/process_manager.py"],
        ]) {
          translateOpenCodeRuntimeEvent({
            type: "tool.started",
            call_id: callId,
            tool: "read",
            summary: path,
          });
        }

        const state = getAdapterStateSnapshot();
        for (const callId of [
          "call-routes",
          "call-service",
          "call-manager",
        ]) {
          if (!state.steps.some(step => step.id === callId)) {
            throw new Error(`Missing unique tool step ${callId}`);
          }
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "PASS"


def test_native_todo_plan_replaces_fallback_and_owns_checklist(
    tmp_path: Path,
) -> None:
    result = _run_adapter_node(
        tmp_path,
        r"""
        import {
          getAdapterStateSnapshot,
          resetAdapterStateForTests,
          translateOpenCodeRuntimeEvent,
        } from "./openCodeTaskPlannerAdapter.mjs";

        resetAdapterStateForTests();
        translateOpenCodeRuntimeEvent({
          type: "runtime.started",
          run_id: "run-native",
          objective: "Audit OpenCode runtime",
        });
        translateOpenCodeRuntimeEvent({
          type: "tool.started",
          call_id: "call-before-plan",
          tool: "read",
          summary: "first.py",
        });

        const update = translateOpenCodeRuntimeEvent({
          type: "runtime.plan.updated",
          detail: "Checklist updated",
          steps: [
            {
              id: "inspect-routes",
              objective: "Inspect runtime routes",
              status: "completed",
            },
            {
              id: "trace-service",
              objective: "Trace service methods",
              status: "running",
            },
            {
              id: "prepare-report",
              objective: "Prepare cited report",
              status: "pending",
            },
          ],
        });

        if (update.length !== 1 || update[0].kind !== "plan") {
          throw new Error("Native todo update did not emit one plan update");
        }

        let state = getAdapterStateSnapshot();
        if (!state.hasNativePlan) {
          throw new Error("Native plan ownership was not recorded");
        }
        if (
          state.steps.length !== 3 ||
          state.steps.some(step => step.id === "call-before-plan") ||
          state.steps.some(step => step.id === "opencode-execution")
        ) {
          throw new Error("Native todos did not replace fallback steps");
        }

        const activity = translateOpenCodeRuntimeEvent({
          type: "tool.started",
          call_id: "call-after-plan",
          tool: "read",
          summary: "src/opencode_runtime/service.py",
        });

        state = getAdapterStateSnapshot();
        if (state.steps.length !== 3) {
          throw new Error("Tool activity polluted native checklist");
        }
        if (
          activity.length !== 1 ||
          activity[0].kind !== "plan" ||
          !activity[0].payload.data.detail.includes("service.py")
        ) {
          throw new Error("Tool activity was not shown as planner detail");
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "PASS"


def test_managed_config_allows_internal_todowrite(
    tmp_path: Path,
) -> None:
    from src.opencode_runtime.config import OpenCodeRuntimeConfig
    from src.opencode_runtime.process_manager import OpenCodeServerManager

    base = tmp_path / "opencode.json"
    base.write_text(
        json.dumps({
            "permission": {"webfetch": "deny"},
            "agent": {
                "build": {
                    "permission": {"webfetch": "deny"},
                },
            },
        }),
        encoding="utf-8",
    )

    manager = OpenCodeServerManager(
        OpenCodeRuntimeConfig(
            config_path=str(base),
            provider_id="odysseus_test",
            model_id="qwen3:8b",
            provider_base_url="http://127.0.0.1:11434/v1",
            provider_models=("qwen3:8b",),
        ),
        health_probe=lambda: {"healthy": True},
        log_root=tmp_path / "logs",
    )

    effective_path, _ = manager._prepare_managed_config()
    payload = json.loads(Path(effective_path).read_text(encoding="utf-8"))

    assert payload["permission"]["todowrite"] == "allow"
    assert (
        payload["agent"]["build"]["permission"]["todowrite"]
        == "allow"
    )
    assert "todowrite" not in manager.MANAGED_PERMISSION_ACTIONS


def test_runtime_contract_requests_internal_todowrite() -> None:
    from src.opencode_runtime.config import OpenCodeRuntimeConfig

    prompt = " ".join(OpenCodeRuntimeConfig().system_prompt.split())
    assert "todowrite tool before the first work tool" in prompt
    assert "Do not output a task plan" in prompt


def test_service_exposes_objective_and_plan_updates() -> None:
    source = (
        ROOT / "src" / "opencode_runtime" / "service.py"
    ).read_text(encoding="utf-8")

    assert '"objective": (' in source
    assert '"runtime.plan.updated"' in source
    assert '"todo.updated"' in source
