from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "static" / "js" / "chat.js"
TRACKER = ROOT / "static" / "js" / "multiGraphTracker.js"
STYLE = ROOT / "static" / "style.css"


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
        pytest.skip(
            "Node.js is required for Task Planner "
            "frontend contract tests."
        )

    tracker_copy = tmp_path / "multiGraphTracker.mjs"
    tracker_copy.write_text(
        _read(TRACKER),
        encoding="utf-8",
        newline="\n",
    )

    script = tmp_path / "task_planner_contract_test.mjs"
    script.write_text(
        textwrap.dedent(body).replace(
            "./multiGraphTracker.mjs",
            tracker_copy.as_uri(),
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


def test_chat_wires_only_langgraph_plan_stream_events() -> None:
    source = _read(CHAT)

    assert source.count(
        "import multiGraphTracker "
        "from './multiGraphTracker.js';"
    ) == 1
    assert source.count(
        "multiGraphTracker.handlePlanUpdate(json);"
    ) == 1
    assert source.count(
        "multiGraphTracker.handleToolStart(json);"
    ) == 1
    assert source.count(
        "multiGraphTracker.handleToolOutput(json);"
    ) == 1
    assert source.count(
        "multiGraphTracker.handleMetrics("
        "json.data, { background: _isBg });"
    ) == 1


def test_tracker_exposes_runtime_agnostic_event_contract() -> None:
    source = _read(TRACKER)

    assert (
        "const MULTI_GRAPH_RUNTIME = "
        "'p3c5b2-multi-graph-v1';"
    ) in source

    for event_type in (
        "task.plan.updated",
        "task.plan.completed",
        "task.plan.failed",
        "task.plan.cancelled",
        "task.step.started",
        "task.step.completed",
        "task.step.failed",
        "task.step.partial",
        "task.step.cancelled",
        "task.reset",
    ):
        assert event_type in source

    assert "handleEvent," in source
    assert "reduceTaskEvent," in source
    assert "normalizePlanUpdate," in source
    assert "reset," in source


def test_legacy_payload_cannot_activate_task_planner(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          createInitialState,
          normalizePlanUpdate,
        } from "./multiGraphTracker.mjs";

        const plan = normalizePlanUpdate({
          data: {
            runtime: "legacy",
            objective: "Legacy request",
            steps: [
              {
                id: "s1",
                objective: "Do work",
              },
            ],
          },
        });
        const state = createInitialState();

        if (plan !== null) {
          throw new Error("Legacy payload was accepted");
        }
        if (
          state.active ||
          state.visible ||
          state.steps.length
        ) {
          throw new Error(
            "Initial state is not inactive",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_mid_run_updates_and_stale_revision_guard(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          MULTI_GRAPH_RUNTIME,
          TASK_EVENT_TYPES,
          createInitialState,
          normalizePlanUpdate,
          reduceTaskEvent,
        } from "./multiGraphTracker.mjs";

        const plan = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "plan-1",
            revision: 2,
            status: "planned",
            objective: "Current plan",
            steps: [
              {
                id: "s1",
                objective: "Inspect",
              },
              {
                id: "s2",
                objective: "Patch",
              },
            ],
          },
        });

        let state = createInitialState();
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan,
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "s1",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_COMPLETED,
          stepId: "s1",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "s2",
        });

        const stale = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "plan-1",
            revision: 1,
            status: "planned",
            objective: "Stale plan",
            steps: [
              {
                id: "s1",
                objective: "Old inspect",
              },
              {
                id: "s2",
                objective: "Old patch",
              },
            ],
          },
        });

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan: stale,
        });

        if (state.objective !== "Current plan") {
          throw new Error(
            "Stale revision replaced the active plan",
          );
        }
        if (
          state.statuses.get("s1") !== "completed"
        ) {
          throw new Error(
            "Completed status was not preserved",
          );
        }
        if (
          state.statuses.get("s2") !== "running"
        ) {
          throw new Error(
            "Current step was not preserved",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_failure_cancellation_and_reset_transitions(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          MULTI_GRAPH_RUNTIME,
          TASK_EVENT_TYPES,
          createInitialState,
          normalizePlanUpdate,
          reduceTaskEvent,
        } from "./multiGraphTracker.mjs";

        const plan = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "plan-2",
            revision: 1,
            objective: "Failure path",
            steps: [
              {
                id: "s1",
                objective: "Inspect",
              },
              {
                id: "s2",
                objective: "Patch",
              },
            ],
          },
        });

        let state = createInitialState();
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan,
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "s1",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_FAILED,
          detail: "boom",
        });

        if (state.overallStatus !== "failed") {
          throw new Error(
            "Plan failure was not recorded",
          );
        }
        if (
          state.statuses.get("s1") !== "failed"
        ) {
          throw new Error(
            "Running step was not marked failed",
          );
        }
        if (state.detail !== "boom") {
          throw new Error(
            "Failure detail was not retained",
          );
        }

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.RESET,
        });

        if (
          state.active ||
          state.visible ||
          state.steps.length
        ) {
          throw new Error(
            "Reset did not clear the planner",
          );
        }

        const cancelledPlan = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "plan-3",
            revision: 1,
            objective: "Cancellation path",
            steps: [
              {
                id: "s1",
                objective: "Wait",
              },
            ],
          },
        });

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan: cancelledPlan,
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "s1",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_CANCELLED,
        });

        if (
          state.overallStatus !== "cancelled"
        ) {
          throw new Error(
            "Plan cancellation was not recorded",
          );
        }
        if (
          state.statuses.get("s1") !== "cancelled"
        ) {
          throw new Error(
            "Running step was not marked cancelled",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_task_planner_styles_cover_terminal_states() -> None:
    source = _read(STYLE)

    assert source.count(
        "/* MULTI_GRAPH_TRACKER_UI_V1_START */"
    ) == 1
    assert source.count(
        "/* MULTI_GRAPH_TRACKER_UI_V1_END */"
    ) == 1

    for selector in (
        ".multi-graph-tracker-status",
        "#multi-graph-tracker.is-complete",
        "#multi-graph-tracker.is-failed",
        "#multi-graph-tracker.is-cancelled",
        ".multi-graph-tracker-step.is-failed",
        ".multi-graph-tracker-step.is-cancelled",
    ):
        assert selector in source

# TASK_PLANNER_UI_EXPLICIT_STREAMING_CONTRACTS_V2_START
#
# Frontend/runtime-boundary tests only. These use the tracker's exported pure
# reducer contract and do not execute LangGraph or launch a browser.


def test_legacy_tool_events_do_not_mount_without_valid_plan(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          TASK_EVENT_TYPES,
          createInitialState,
          reduceTaskEvent,
        } from "./multiGraphTracker.mjs";

        let state = createInitialState();

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "legacy-ghost-step",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_COMPLETED,
          stepId: "legacy-ghost-step",
        });

        if (
          state.active ||
          state.visible ||
          state.steps.length ||
          state.statuses.size ||
          state.currentStepId !== null
        ) {
          throw new Error(
            "Legacy-like tool events mounted or mutated the planner",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_duplicate_events_are_idempotent_and_stale_revision_safe(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          MULTI_GRAPH_RUNTIME,
          TASK_EVENT_TYPES,
          createInitialState,
          normalizePlanUpdate,
          reduceTaskEvent,
        } from "./multiGraphTracker.mjs";

        const plan = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "duplicate-plan",
            revision: 3,
            objective: "Current duplicate-safe plan",
            steps: [
              {
                id: "s1",
                objective: "Inspect",
              },
            ],
          },
        });

        let state = createInitialState();

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan,
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan,
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "s1",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "s1",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_COMPLETED,
          stepId: "s1",
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_COMPLETED,
          stepId: "s1",
        });

        const stale = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "duplicate-plan",
            revision: 2,
            objective: "Stale duplicate plan",
            steps: [
              {
                id: "s1",
                objective: "Old inspect",
              },
            ],
          },
        });

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan: stale,
        });

        if (state.steps.length !== 1) {
          throw new Error("Duplicate plan event duplicated steps");
        }
        if (state.statuses.get("s1") !== "completed") {
          throw new Error(
            "Duplicate step events did not remain idempotent",
          );
        }
        if (
          state.objective !== "Current duplicate-safe plan"
        ) {
          throw new Error(
            "Stale revision replaced the active duplicate-safe plan",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_unknown_plan_step_id_is_ignored_safely(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          MULTI_GRAPH_RUNTIME,
          TASK_EVENT_TYPES,
          createInitialState,
          normalizePlanUpdate,
          reduceTaskEvent,
        } from "./multiGraphTracker.mjs";

        const plan = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "unknown-step-plan",
            revision: 1,
            objective: "Ignore unknown step IDs",
            steps: [
              {
                id: "s1",
                objective: "Known step",
              },
            ],
          },
        });

        let state = createInitialState();
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan,
        });

        for (const type of [
          TASK_EVENT_TYPES.STEP_STARTED,
          TASK_EVENT_TYPES.STEP_COMPLETED,
          TASK_EVENT_TYPES.STEP_FAILED,
          TASK_EVENT_TYPES.STEP_PARTIAL,
          TASK_EVENT_TYPES.STEP_CANCELLED,
        ]) {
          state = reduceTaskEvent(state, {
            type,
            stepId: "unknown-step",
          });
        }

        if (state.statuses.has("unknown-step")) {
          throw new Error(
            "Unknown plan_step_id was inserted into tracker state",
          );
        }
        if (state.statuses.get("s1") !== "pending") {
          throw new Error(
            "Unknown plan_step_id mutated a known step",
          );
        }
        if (state.currentStepId !== null) {
          throw new Error(
            "Unknown plan_step_id became the current step",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_tool_start_and_tool_output_transition_running_to_terminal(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          MULTI_GRAPH_RUNTIME,
          TASK_EVENT_TYPES,
          createInitialState,
          normalizePlanUpdate,
          reduceTaskEvent,
        } from "./multiGraphTracker.mjs";

        const plan = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "transition-plan",
            revision: 1,
            objective: "Exercise streaming transitions",
            steps: [
              {
                id: "completed-step",
                objective: "Complete",
              },
              {
                id: "partial-step",
                objective: "Return partial",
              },
              {
                id: "failed-step",
                objective: "Fail safely",
              },
            ],
          },
        });

        let state = createInitialState();
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan,
        });

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "completed-step",
        });
        if (
          state.statuses.get("completed-step") !== "running"
        ) {
          throw new Error(
            "tool_start did not transition the step to running",
          );
        }
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_COMPLETED,
          stepId: "completed-step",
        });
        if (
          state.statuses.get("completed-step") !== "completed"
        ) {
          throw new Error(
            "tool_output completion did not reach completed",
          );
        }

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "partial-step",
        });
        if (
          state.statuses.get("partial-step") !== "running"
        ) {
          throw new Error(
            "Partial step did not enter running",
          );
        }
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_PARTIAL,
          stepId: "partial-step",
        });
        if (
          state.statuses.get("partial-step") !== "partial"
        ) {
          throw new Error(
            "tool_output partial did not reach partial",
          );
        }

        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_STARTED,
          stepId: "failed-step",
        });
        if (
          state.statuses.get("failed-step") !== "running"
        ) {
          throw new Error(
            "Failed step did not enter running",
          );
        }
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.STEP_FAILED,
          stepId: "failed-step",
          detail: "expected failure",
        });
        if (
          state.statuses.get("failed-step") !== "failed"
        ) {
          throw new Error(
            "tool_output failure did not reach failed",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"


def test_final_completion_auto_hide_contract(
    tmp_path: Path,
) -> None:
    result = _run_node(
        tmp_path,
        """
        import {
          MULTI_GRAPH_RUNTIME,
          TASK_EVENT_TYPES,
          createInitialState,
          normalizePlanUpdate,
          reduceTaskEvent,
        } from "./multiGraphTracker.mjs";

        const plan = normalizePlanUpdate({
          data: {
            runtime: MULTI_GRAPH_RUNTIME,
            plan_id: "completion-plan",
            revision: 1,
            objective: "Complete and hide",
            steps: [
              {
                id: "s1",
                objective: "Finish",
              },
            ],
          },
        });

        let state = createInitialState();
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_UPDATED,
          plan,
        });
        state = reduceTaskEvent(state, {
          type: TASK_EVENT_TYPES.PLAN_COMPLETED,
        });

        if (state.overallStatus !== "completed") {
          throw new Error(
            "Final completion was not recorded",
          );
        }
        if (state.currentStepId !== null) {
          throw new Error(
            "Final completion retained a current step",
          );
        }

        console.log("PASS");
        """,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
    assert result.stdout.strip() == "PASS"

    source = _read(TRACKER)

    assert (
        "const TERMINAL_PLAN_STATUSES = "
        "new Set(['completed', 'failed', 'cancelled']);"
    ) in source
    assert "function scheduleHide(ms = 7000) {" in source
    assert "state.hideTimer = setTimeout(() => {" in source
    assert "scheduleHide();" in source

# TASK_PLANNER_UI_EXPLICIT_STREAMING_CONTRACTS_V2_END
