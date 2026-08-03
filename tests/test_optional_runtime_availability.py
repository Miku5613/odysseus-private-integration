from pathlib import Path
import subprocess
import sys

import pytest
from fastapi import HTTPException

from routes.opencode_runtime_routes import _require_opencode_installed
from src import runtime_availability as availability


ROOT = Path(__file__).resolve().parents[1]


def test_langgraph_missing_is_optional_and_legacy_remains_available(
    monkeypatch,
) -> None:
    real_import = availability.importlib.import_module

    def import_with_missing_checkpoint(name: str):
        if name == "langgraph.checkpoint.sqlite":
            raise ModuleNotFoundError("simulated missing optional checkpoint")
        return real_import(name)

    monkeypatch.setattr(
        availability.importlib,
        "import_module",
        import_with_missing_checkpoint,
    )
    monkeypatch.setattr(availability, "_opencode_executable", lambda: "")

    statuses = availability.runtime_availability()
    assert statuses["legacy"]["available"] is True
    assert statuses["langgraph"]["available"] is False
    assert statuses["langgraph"]["mandatory"] is False
    assert statuses["opencode"]["available"] is False


def test_opencode_route_guard_returns_controlled_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "routes.opencode_runtime_routes.runtime_is_available",
        lambda runtime: False,
    )
    with pytest.raises(HTTPException) as caught:
        _require_opencode_installed()
    assert caught.value.status_code == 503
    assert "not installed" in str(caught.value.detail).lower()


def test_optional_packages_are_not_core_requirements() -> None:
    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    langgraph = (ROOT / "requirements-langgraph.txt").read_text(
        encoding="utf-8"
    )
    assert "langgraph==" not in core
    assert "langchain-core" not in core
    assert "langgraph-checkpoint-sqlite" not in core
    assert "langgraph==1.2.7" in langgraph
    assert "langchain-core" in langgraph
    assert "langgraph-checkpoint-sqlite" in langgraph


def test_runtime_ui_disables_missing_choices_and_installer_excludes_models() -> None:
    runtime_js = (ROOT / "static" / "js" / "runtimeControl.js").read_text(
        encoding="utf-8"
    )
    installer = (ROOT / "Express Installation.bat").read_text(
        encoding="utf-8"
    ).lower()
    assert "/api/runtime/availability" in runtime_js
    assert "option.disabled = !available" in runtime_js
    assert "Not Installed" in runtime_js
    assert "filter(runtimeAvailable)" in runtime_js
    assert "ollama.ollama" not in installer
    assert "lm studio" in installer  # Explicitly documents that it is excluded.


def test_dependency_status_distinguishes_current_missing_and_update(tmp_path) -> None:
    helper = ROOT / "scripts" / "dependency_status.py"
    cases = (
        ("pip", 0, "[CURRENT]"),
        ("odysseus-package-that-does-not-exist-987654", 10, "[MISSING]"),
        ("pip<0", 20, "[UPDATE REQUIRED]"),
    )
    for index, (requirement, expected_code, marker) in enumerate(cases):
        requirements = tmp_path / f"requirements-{index}.txt"
        requirements.write_text(requirement + "\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(helper),
                "requirements",
                str(requirements),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == expected_code, result.stdout + result.stderr
        assert marker in result.stdout


def test_express_installer_and_portable_launcher_are_state_aware() -> None:
    installer = (ROOT / "Express Installation.bat").read_text(
        encoding="utf-8"
    )
    launcher = (
        ROOT / "scripts" / "portable-quick-launch-template.bat"
    ).read_text(encoding="utf-8")

    assert "dependency_status.py requirements requirements-langgraph.txt" in installer
    assert "already installed and compatible" in installer
    assert "Do you want to update the current version of LangGraph/LangChain?" in installer
    assert "Do you want to update the current version of OpenCode?" in installer
    assert "Would you like to create Quick Launch Odysseus.bat?" in installer
    assert "fc /b" in installer

    assert "dependency_status.py requirements requirements.txt" in launcher
    assert "runtime_reachability.py chroma" in launcher
    assert "runtime_reachability.py model" in launcher
    assert "Do you still want to boot Odysseus without them?" in launcher
    assert "goto :required_failed" in launcher
    assert "-m uvicorn app:app --host 127.0.0.1 --port 7000" in launcher
