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
          objective: [
            "1. Review runtime setup.",
            "2. Explain session handling.",
            "3. Prepare the final answer.",
          ].join("\\n"),
        });

        const initial = getAdapterStateSnapshot();
        const initialIds = initial.steps.map(step => step.id);

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
        if (
          JSON.stringify(state.steps.map(step => step.id))
          !== JSON.stringify(initialIds)
        ) {
          throw new Error("Tool calls created planner objectives");
        }
        if (
          state.steps.some(step => (
            /[\\\\/]|\\.py\\b|qwen|\\d{1,2}:\\d{2}/i.test(step.objective)
          ))
        ) {
          throw new Error(
            `Technical planner objective leaked: ${JSON.stringify(state.steps)}`
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout

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
            { id: "inspect-routes", objective: "Inspect runtime routes", status: "completed" },
            { id: "trace-service", objective: "Trace service methods", status: "running" },
            { id: "prepare-report", objective: "Prepare cited report", status: "pending" },
          ],
        });

        if (update.length !== 1 || update[0].kind !== "plan") {
          throw new Error("Native todo update did not emit one plan update");
        }

        let state = getAdapterStateSnapshot();
        if (!state.hasNativePlan || state.steps.length !== 3) {
          throw new Error("Native plan ownership was not recorded");
        }
        if (state.steps.some(step => step.id === "call-before-plan")) {
          throw new Error("Fallback tool step survived native-plan adoption");
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

        const detail = activity?.[0]?.payload?.data?.detail || "";
        if (
          activity.length !== 1 ||
          activity[0].kind !== "plan" ||
          !detail ||
          /service\\.py|src[\\\\/]|\\bread\\b/i.test(detail)
        ) {
          throw new Error(`Technical planner detail leaked: ${detail}`);
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout

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

def test_fallback_planner_uses_user_objectives_not_tool_paths(
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
          run_id: "run-user-facing",
          objective: [
            "1. Inspect src/opencode_runtime/config.py and summarize how settings are selected.",
            "2. Inspect src/opencode_runtime/transport.py and explain session handling.",
            "3. Give me a concise final verdict.",
          ].join("\\n"),
        });

        let state = getAdapterStateSnapshot();
        const labels = state.steps.map(step => step.objective);
        if (labels.length !== 3) throw new Error(JSON.stringify(labels));
        if (labels.some(label => /[A-Za-z]:\\\\|src[\\\\/]|\\.py\\b|qwen|\\d{1,2}:\\d{2}/i.test(label))) {
          throw new Error(`Technical detail leaked: ${JSON.stringify(labels)}`);
        }

        translateOpenCodeRuntimeEvent({
          type: "tool.completed",
          call_id: "read-one",
          tool: "read",
          summary: "D:\\\\AI\\\\odysseus\\\\src\\\\opencode_runtime\\\\config.py",
          status: "completed",
        });
        state = getAdapterStateSnapshot();
        if (state.steps[0].status !== "completed") {
          throw new Error("First objective did not complete");
        }

        translateOpenCodeRuntimeEvent({
          type: "tool.started",
          call_id: "read-two",
          tool: "read",
          summary: "D:\\\\AI\\\\odysseus\\\\src\\\\opencode_runtime\\\\transport.py",
        });
        state = getAdapterStateSnapshot();
        if (state.currentStepId !== state.steps[1].id) {
          throw new Error("Planner did not advance");
        }

        const expected = [
          "Summarize how settings are selected.",
          "Explain session handling.",
          "Give me a concise final verdict.",
        ];
        if (JSON.stringify(labels) !== JSON.stringify(expected)) {
          throw new Error(`Unexpected labels: ${JSON.stringify(labels)}`);
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, result.stderr or result.stdout

def test_runtime_completion_closes_every_planner_item(tmp_path: Path) -> None:
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
          run_id: "run-completion",
          objective: [
            "1. Review the available endpoints.",
            "2. Explain the runtime lifecycle.",
            "3. Prepare the final answer.",
          ].join("\\n"),
        });
        translateOpenCodeRuntimeEvent({
          type: "tool.completed",
          call_id: "one",
          tool: "read",
          status: "completed",
        });
        const actions = translateOpenCodeRuntimeEvent({
          type: "runtime.completed",
          run_id: "run-completion",
        });
        const state = getAdapterStateSnapshot();
        if (actions.length !== 1 || actions[0].kind !== "plan") {
          throw new Error("Completion did not emit one planner action");
        }
        if (actions[0].payload.data.status !== "completed") {
          throw new Error("Overall planner did not complete");
        }
        if (state.steps.some(step => step.status !== "completed")) {
          throw new Error(`Incomplete steps remained: ${JSON.stringify(state.steps)}`);
        }
        if (state.currentStepId) {
          throw new Error("Current step remained selected after completion");
        }
        console.log("PASS");
        """,
    )
    assert result.returncode == 0, result.stderr or result.stdout
