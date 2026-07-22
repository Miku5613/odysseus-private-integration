from dataclasses import FrozenInstanceError

import pytest

from src.agent_runtime.supervisor_contracts import (
    CapabilityDescriptor,
    WorkerHandoff,
    WorkerResult,
)


def test_capability_descriptor_supports_worker_capability():
    capability = CapabilityDescriptor(
        name="code_inspection",
        description="Inspect arbitrary programming codebases.",
        kind="worker",
        provider="code_worker",
        discoverable_inputs=("workspace", "code_target"),
        supported_actions=("inspect", "trace", "explain"),
        tool_names=("grep", "read_file", "read_file_around"),
    )

    assert capability.name == "code_inspection"
    assert capability.kind == "worker"
    assert "code_target" in capability.discoverable_inputs


def test_capability_descriptor_is_immutable():
    capability = CapabilityDescriptor(
        name="knowledge_search",
        description="Search indexed semantic knowledge.",
        kind="worker",
        provider="knowledge_worker",
    )

    with pytest.raises(FrozenInstanceError):
        capability.name = "changed"


def test_worker_handoff_is_domain_neutral():
    handoff = WorkerHandoff(
        capability="knowledge_search",
        objective="Find relevant project decisions.",
        context={"query": "Atlas write mode"},
    )

    assert handoff.context["query"] == "Atlas write mode"
    assert not hasattr(handoff, "claim_support")
    assert not hasattr(handoff, "code_location")


def test_worker_result_does_not_require_code_evidence():
    result = WorkerResult(
        capability="deep_research",
        status="completed",
        summary="Research completed.",
        data={"research_id": "example"},
    )

    assert result.status == "completed"
    assert result.evidence == ()


def test_worker_result_can_request_clarification():
    result = WorkerResult(
        capability="mcp_specialized",
        status="needs_clarification",
        summary="More information is required.",
        missing_information=("target_service",),
    )

    assert result.missing_information == ("target_service",)


def test_worker_result_supports_specialist_evidence():
    result = WorkerResult(
        capability="code_inspection",
        status="completed",
        summary="Implementation located.",
        evidence=(
            {
                "kind": "code_location",
                "file": "routes/chat_routes.py",
                "line": 100,
            },
        ),
    )

    assert result.evidence[0]["kind"] == "code_location"



def test_phase5ft_contract_contextual_match_ranking(
    monkeypatch,
) -> None:
    from src.agent_runtime import contract_runtime
    from src.agent_runtime.contract_models import SubTask

    output = "\n".join(
        [
            (
                "Exclusions/Backups/routes/chat_routes.py.before_patch:"
                "48: tool_output"
            ),
            (
                "Exclusions/Scripts/prompt_harness.py:"
                "10: tool_output"
            ),
            (
                "routes/chat_routes.py:"
                "500: tool_output"
            ),
        ]
    )

    task = SubTask(
        id="task_001",
        kind="exact_lookup",
        target="tool_output",
        source_scope=["."],
        dependencies=[],
        is_lookup_only=True,
        required_evidence="Find an exact source match for tool_output.",
    )

    bonuses = {
        "Exclusions/Backups/routes/chat_routes.py.before_patch": -29.0,
        "Exclusions/Scripts/prompt_harness.py": -44.0,
        "routes/chat_routes.py": 80.0,
    }

    monkeypatch.setattr(
        contract_runtime,
        "_contextual_match_bonus",
        lambda workspace, path, repository_context=None: bonuses.get(
            path,
            0.0,
        ),
    )

    match = contract_runtime._first_match(
        output,
        task,
        r"D:\AI\odysseus",
    )

    assert match is not None
    assert match[0] == "routes/chat_routes.py"


def test_phase5ft_unusual_exact_sources_remain_eligible(
    monkeypatch,
) -> None:
    from src.agent_runtime import contract_runtime
    from src.agent_runtime.contract_models import SubTask

    output = (
        "Exclusions/Backups/routes/chat_routes.py.before_patch:"
        "48: unique_backup_only_symbol"
    )

    task = SubTask(
        id="task_001",
        kind="exact_lookup",
        target="unique_backup_only_symbol",
        source_scope=["."],
        dependencies=[],
        is_lookup_only=True,
        required_evidence=(
            "Find an exact source match for unique_backup_only_symbol."
        ),
    )

    monkeypatch.setattr(
        contract_runtime,
        "_contextual_match_bonus",
        lambda workspace, path, repository_context=None: -29.0,
    )

    match = contract_runtime._first_match(
        output,
        task,
        r"D:\AI\odysseus",
    )

    assert match is not None
    assert (
        match[0]
        == "Exclusions/Backups/routes/chat_routes.py.before_patch"
    )


def test_phase5ft_contextual_ranking_without_workspace() -> None:
    from src.agent_runtime import contract_runtime
    from src.agent_runtime.contract_models import SubTask

    output = "\n".join(
        [
            "routes/a.py:20: target_symbol",
            "routes/b.py:30: target_symbol",
        ]
    )

    task = SubTask(
        id="task_001",
        kind="exact_lookup",
        target="target_symbol",
        source_scope=["."],
        dependencies=[],
        is_lookup_only=True,
        required_evidence="Find target_symbol.",
    )

    match = contract_runtime._first_match(
        output,
        task,
    )

    assert match is not None



def test_phase5ft_contract_state_owns_workspace() -> None:
    from src.agent_runtime.contract_models import ContractState

    state = ContractState(
        raw_query="test",
        workspace=r"D:\Example\Workspace",
    )

    assert state.workspace == r"D:\Example\Workspace"


def test_phase5ft_repository_context_built_once(
    monkeypatch,
) -> None:
    from src.agent_runtime import contract_runtime
    from src.agent_runtime.contract_models import SubTask

    output = "\n".join(
        [
            "routes/a.py:10: target_symbol",
            "routes/b.py:20: target_symbol",
            "routes/c.py:30: target_symbol",
        ]
    )

    task = SubTask(
        id="task_001",
        kind="exact_lookup",
        target="target_symbol",
        source_scope=["."],
        dependencies=[],
        is_lookup_only=True,
        required_evidence="Find target_symbol.",
    )

    builds = []

    def fake_build(workspace):
        builds.append(workspace)
        return {
            "tracked": set(),
            "ignored": set(),
            "modified": set(),
        }

    from src.agent_runtime import general_graph

    monkeypatch.setattr(
        general_graph,
        "_build_repository_context",
        fake_build,
    )

    monkeypatch.setattr(
        contract_runtime,
        "_contextual_match_bonus",
        lambda workspace, path, repository_context=None: 0.0,
    )

    match = contract_runtime._first_match(
        output,
        task,
        r"D:\AI\odysseus",
    )

    assert match is not None
    assert builds == [r"D:\AI\odysseus"]



def test_phase5ftw_workspace_context_is_explicit() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]

    critical = [
        repo / "src" / "agent_runtime" / "tool_adapters.py",
        repo / "src" / "agent_runtime" / "runner.py",
        repo / "src" / "agent_runtime" / "streaming.py",
        repo / "src" / "agent_runtime" / "general_bridge.py",
        repo / "src" / "agent_runtime" / "contract_bridge.py",
        repo / "src" / "agent_runtime" / "contract_runtime.py",
        repo / "routes" / "chat_routes.py",
    ]

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in critical
    )

    assert "DEFAULT_WORKSPACE" not in combined
    assert "odysseus-upstream-test" not in combined

    assert (
        "workspace=str(Path(__file__).resolve().parents[1])"
        not in combined
    )


def test_phase5ftw_missing_workspace_fails_closed() -> None:
    from src.agent_runtime import tool_adapters

    token = tool_adapters._ACTIVE_WORKSPACE.set(None)

    try:
        try:
            tool_adapters.active_workspace()
        except RuntimeError as exc:
            assert "No active workspace" in str(exc)
        else:
            raise AssertionError(
                "Missing workspace unexpectedly resolved."
            )
    finally:
        tool_adapters._ACTIVE_WORKSPACE.reset(token)


def test_phase5ftw_bound_workspace_is_authoritative(
    tmp_path,
) -> None:
    from src.agent_runtime import tool_adapters

    token = tool_adapters.bind_workspace(tmp_path)

    try:
        assert (
            tool_adapters.active_workspace()
            == tmp_path.resolve()
        )

        assert (
            tool_adapters._safe_path(".")
            == tmp_path.resolve()
        )
    finally:
        tool_adapters.reset_workspace(token)



def test_phase5ftw_contract_eval_workspace_is_explicit() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    source = (
        repo
        / "src"
        / "agent_runtime"
        / "evals"
        / "run_contract_evals.py"
    ).read_text(
        encoding="utf-8"
    )

    assert 'parser.add_argument(' in source
    assert '"--workspace"' in source
    assert "required=True" in source
    assert "run_case(query: str, workspace: str)" in source
    assert "EVAL_WORKSPACE" not in source


def test_phase5ftw_contract_eval_does_not_rewrite_queries() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    source = (
        repo
        / "src"
        / "agent_runtime"
        / "evals"
        / "run_contract_evals.py"
    ).read_text(
        encoding="utf-8"
    ).lower()

    assert "odysseus-upstream-test" not in source
    assert "d:\\\\ai\\\\odysseus" not in source


def test_phase5ftw_contract_tool_call_binds_state_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    from src.agent_runtime import contract_runtime
    from src.agent_runtime.contract_models import ContractState

    observed = {}

    class FakeTool:
        def invoke(self, args):
            from src.agent_runtime.tool_adapters import active_workspace

            observed["workspace"] = active_workspace()
            return "ok"

    monkeypatch.setitem(
        contract_runtime.TOOL_MAP,
        "phase5ftw_fake_tool",
        FakeTool(),
    )

    state = ContractState(
        raw_query="test",
        workspace=str(tmp_path),
    )

    output = contract_runtime._call_tool(
        "phase5ftw_fake_tool",
        {},
        "phase5ftw_call",
        state,
    )

    assert output == "ok"
    assert observed["workspace"] == tmp_path.resolve()


def test_phase5ftw_contract_tool_call_resets_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    from src.agent_runtime import contract_runtime
    from src.agent_runtime.contract_models import ContractState
    from src.agent_runtime import tool_adapters

    class FakeTool:
        def invoke(self, args):
            return "ok"

    monkeypatch.setitem(
        contract_runtime.TOOL_MAP,
        "phase5ftw_fake_reset_tool",
        FakeTool(),
    )

    token = tool_adapters._ACTIVE_WORKSPACE.set(None)

    try:
        state = ContractState(
            raw_query="test",
            workspace=str(tmp_path),
        )

        contract_runtime._call_tool(
            "phase5ftw_fake_reset_tool",
            {},
            "phase5ftw_reset_call",
            state,
        )

        assert tool_adapters._ACTIVE_WORKSPACE.get() is None
    finally:
        tool_adapters._ACTIVE_WORKSPACE.reset(token)

def test_phase5fw_glob_files_is_safe_read() -> None:
    from src.agent_runtime.tool_risk_mapper import (
        classify_tool_risk,
        decide_tool_policy_action,
    )

    risk = classify_tool_risk("glob_files")
    decision = decide_tool_policy_action("glob_files")

    assert risk.level == "safe_read"
    assert risk.requires_confirmation is False
    assert decision.action == "allow"