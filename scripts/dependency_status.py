"""Small dependency-state probe used by the Windows batch entry points."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from packaging.requirements import Requirement
except ImportError:  # A brand-new venv still has pip's vendored copy.
    from pip._vendor.packaging.requirements import Requirement


CURRENT = 0
MISSING = 10
UPDATE = 20
UNKNOWN = 30


def requirement_state(path: Path) -> int:
    missing: list[str] = []
    incompatible: list[str] = []
    installed: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            nested = line.split(maxsplit=1)[1]
            nested_status = requirement_state(path.parent / nested)
            if nested_status == MISSING:
                missing.append(nested)
            elif nested_status != CURRENT:
                incompatible.append(nested)
            continue
        try:
            requirement = Requirement(line)
        except Exception:
            print(f"[UNKNOWN] Unsupported requirement entry: {line}")
            return UNKNOWN
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(requirement.name)
            continue
        installed.append(f"{requirement.name}=={version}")
        if requirement.specifier and version not in requirement.specifier:
            incompatible.append(
                f"{requirement.name} {version} requires {requirement.specifier}"
            )

    if missing:
        print("[MISSING] " + ", ".join(sorted(set(missing))))
        if incompatible:
            print("[UPDATE REQUIRED] " + ", ".join(incompatible))
        return MISSING
    if incompatible:
        print("[UPDATE REQUIRED] " + ", ".join(incompatible))
        return UPDATE
    print(f"[CURRENT] {path.name}: {len(installed)} package(s) satisfy requirements")
    return CURRENT


def _npm_executable() -> str:
    found = shutil.which("npm.cmd") or shutil.which("npm")
    if found:
        return found
    candidate = Path(r"C:\Program Files\nodejs\npm.cmd")
    return str(candidate) if candidate.is_file() else ""


def _run_npm(npm: str, arguments: str, timeout: float) -> str:
    process = subprocess.Popen(
        f'"{npm}" {arguments}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
        return stdout if process.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        else:
            process.kill()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        return ""


def _installed_opencode_version(npm: str) -> str:
    if not npm:
        return ""
    try:
        npm_root = _run_npm(npm, "root -g", 8).strip()
        package_file = Path(npm_root) / "opencode-ai" / "package.json"
        if package_file.is_file():
            return str(
                json.loads(package_file.read_text(encoding="utf-8")).get(
                    "version", ""
                )
            ).strip()
    except Exception:
        pass
    return ""


def opencode_state(*, check_updates: bool = True) -> int:
    executable = (
        shutil.which("opencode.cmd")
        or shutil.which("opencode.exe")
        or shutil.which("opencode")
    )
    if not executable:
        candidate = Path.home() / "AppData" / "Roaming" / "npm" / "opencode.cmd"
        executable = str(candidate) if candidate.is_file() else ""
    if not executable:
        print("[MISSING] OpenCode CLI")
        return MISSING

    npm = _npm_executable()
    installed = _installed_opencode_version(npm)
    if not check_updates:
        version_text = f" {installed}" if installed else ""
        print(f"[CURRENT] OpenCode{version_text}: {executable}")
        return CURRENT
    if not npm or not installed:
        print(f"[CURRENT] OpenCode CLI found at {executable}; update state unavailable")
        return UNKNOWN

    try:
        output = _run_npm(npm, "view opencode-ai version --json", 8)
        latest = str(json.loads(output or '""')).strip()
    except Exception:
        latest = ""
    if not latest:
        print(f"[CURRENT] OpenCode {installed}; registry update check unavailable")
        return UNKNOWN
    if installed != latest:
        print(f"[UPDATE REQUIRED] OpenCode {installed} -> {latest}")
        return UPDATE
    print(f"[CURRENT] OpenCode {installed}")
    return CURRENT


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    requirements = subparsers.add_parser("requirements")
    requirements.add_argument("path", type=Path)
    opencode = subparsers.add_parser("opencode")
    opencode.add_argument("--installed-only", action="store_true")
    args = parser.parse_args()
    if args.command == "requirements":
        return requirement_state(args.path.resolve())
    return opencode_state(check_updates=not args.installed_only)


if __name__ == "__main__":
    raise SystemExit(main())
