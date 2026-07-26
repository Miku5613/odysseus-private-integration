from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "static" / "js" / "runtimeControl.js"
ADAPTER = ROOT / "static" / "js" / "openCodeTaskPlannerAdapter.js"
TRACKER = ROOT / "static" / "js" / "multiGraphTracker.js"


def _read(path: Path) -> str:
    return path.read_text(
        encoding="utf-8-sig",
        errors="strict",
    )


def _run_node(
    tmp_path: Path,
    body: str,
) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node.exe") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is required.")

    tracker_copy = tmp_path / "multiGraphTracker.mjs"
    tracker_copy.write_text(
        _read(TRACKER),
        encoding="utf-8",
        newline="\n",
    )

    adapter_source = _read(ADAPTER).replace(
        "./multiGraphTracker.js",
        tracker_copy.as_uri(),
    )
    adapter_copy = tmp_path / "openCodeTaskPlannerAdapter.mjs"
    adapter_copy.write_text(
        adapter_source,
        encoding="utf-8",
        newline="\n",
    )

    script = tmp_path / "opencode_task_planner_test.mjs"
    script.write_text(
        textwrap.dedent(body).replace(
            "./openCodeTaskPlannerAdapter.mjs",
            adapter_copy.as_uri(),
        ),
        encoding="utf-8",
        newline="\n",
    )

    return subprocess.run(
        [node, str(script)],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_runtime_control_wires_adapter_once() -> None:
    source = _read(RUNTIME)

    assert source.count(
        "import openCodeTaskPlannerAdapter "
        "from './openCodeTaskPlannerAdapter.js';"
    ) == 1
    assert source.count(
        "openCodeTaskPlannerAdapter.handleRuntimeEvent(event, {"
    ) == 1
    assert source.count(
        "openCodeTaskPlannerAdapter.handleRuntimeChanged(runtime, {"
    ) == 1


def test_permission_events_never_activate_planner(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          getAdapterStateSnapshot,
          resetAdapterStateForTests,
          translateOpenCodeRuntimeEvent,
        } from "./openCodeTaskPlannerAdapter.mjs";

        resetAdapterStateForTests();

        for (const type of [
          "runtime.permission_requested",
          "runtime.permission_resolved",
        ]) {
          const actions = translateOpenCodeRuntimeEvent({
            type,
            action: "write",
          });

          if (actions.length) {
            throw new Error(
              "Permission-only event produced planner actions",
            );
          }
        }

        if (getAdapterStateSnapshot().ownsTracker) {
          throw new Error(
            "Permission-only event claimed tracker ownership",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_started_status_and_completed_translate_to_contract(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          resetAdapterStateForTests,
          translateOpenCodeRuntimeEvent,
        } from "./openCodeTaskPlannerAdapter.mjs";

        resetAdapterStateForTests();

        const started = translateOpenCodeRuntimeEvent({
          type: "runtime.started",
          run_id: "run-1",
          title: "Patch project",
          steps: [
            { id: "inspect", objective: "Inspect" },
            { id: "patch", objective: "Patch" },
          ],
        }, {
          sessionId: "session-1",
        });

        if (
          started.length !== 2 ||
          started[0].kind !== "plan" ||
          started[1].event.type !== "task.step.started"
        ) {
          throw new Error(
            "runtime.started did not create plan + running step",
          );
        }

        const status = translateOpenCodeRuntimeEvent({
          type: "runtime.status",
          step_id: "inspect",
          status: "completed",
          message: "inspection complete",
        });

        if (
          status.at(-1)?.event?.type !==
          "task.step.completed"
        ) {
          throw new Error(
            "runtime.status did not complete the step",
          );
        }

        const completed = translateOpenCodeRuntimeEvent({
          type: "runtime.completed",
        });

        if (
          completed[0]?.event?.type !==
          "task.plan.completed"
        ) {
          throw new Error(
            "runtime.completed did not complete the plan",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_tool_events_failed_and_cancelled_translate(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          getAdapterStateSnapshot,
          resetAdapterStateForTests,
          translateOpenCodeRuntimeEvent,
        } from "./openCodeTaskPlannerAdapter.mjs";

        resetAdapterStateForTests();

        translateOpenCodeRuntimeEvent({
          type: "runtime.started",
          run_id: "run-2",
          title: "Tool run",
        });

        const toolStarted = translateOpenCodeRuntimeEvent({
          type: "tool.started",
          tool_call_id: "tool-1",
          tool: "read_file",
        });

        if (
          toolStarted.at(-1)?.event?.type !==
          "task.step.started"
        ) {
          throw new Error(
            "tool.started did not start a planner step",
          );
        }

        if (
          !getAdapterStateSnapshot().steps.some(
            step => step.id === "tool-1",
          )
        ) {
          throw new Error(
            "tool.started did not add its dynamic step",
          );
        }

        const toolCompleted = translateOpenCodeRuntimeEvent({
          type: "tool.completed",
          tool_call_id: "tool-1",
          tool: "read_file",
        });

        if (
          toolCompleted.at(-1)?.event?.type !==
          "task.step.completed"
        ) {
          throw new Error(
            "tool.completed did not complete the planner step",
          );
        }

        const failed = translateOpenCodeRuntimeEvent({
          type: "runtime.failed",
          error: "boom",
        });

        if (
          failed[0]?.event?.type !==
          "task.plan.failed"
        ) {
          throw new Error(
            "runtime.failed did not fail the plan",
          );
        }

        resetAdapterStateForTests();
        translateOpenCodeRuntimeEvent({
          type: "runtime.started",
          run_id: "run-3",
        });

        const cancelled = translateOpenCodeRuntimeEvent({
          type: "runtime.cancelled",
        });

        if (
          cancelled[0]?.event?.type !==
          "task.plan.cancelled"
        ) {
          throw new Error(
            "runtime.cancelled did not cancel the plan",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"
