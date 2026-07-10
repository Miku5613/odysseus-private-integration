"""Atlas migration pre-merge audit.

This script produces a factual merge-readiness report for the Atlas/context
migration branch.

It does not enforce artificial runtime isolation. Its job is to help review:
- branch/base state
- changed files since base
- working tree cleanliness
- local config hygiene
- context provider readiness
- potentially risky touched files

It does not call the model, does not invoke LangGraph, and does not write to any
authorized root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.context_merge_readiness import run_merge_readiness


DEFAULT_BASE = "local/f3-upstream-dev-green-20260705"
DEFAULT_CONFIG_PATH = Path("config/local_roots.json")


RUNTIME_SENSITIVE_PATH_PREFIXES = [
    "routes/chat_routes.py",
    "src/agent_loop.py",
    "src/agent_runtime/",
    "src/tool_execution.py",
    "src/tool_index.py",
    "src/tool_policy.py",
    "src/tool_security.py",
    "src/tool_schemas.py",
]

EXPECTED_ATLAS_CONTEXT_PREFIXES = [
    "app.py",
    "config/local_roots.example.json",
    ".gitignore",
    "routes/atlas_routes.py",
    "routes/context_routes.py",
    "routes/context_ui_routes.py",
    "scripts/atlas_migration_premerge_audit.py",
    "scripts/context_",
    "src/agent_tools/",
    "src/atlas_links.py",
    "src/constants.py",
    "src/context_providers/",
    "static/",
    "tests/test_atlas_",
    "tests/test_context_",
]


def run_premerge_audit(
    *,
    base_ref: str = DEFAULT_BASE,
    config_path: Path = DEFAULT_CONFIG_PATH,
    smoke_query: str | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": True,
        "base_ref": base_ref,
        "checks": [],
    }

    def add_check(name: str, ok: bool, detail: str, **extra: Any) -> None:
        report["checks"].append(
            {
                "name": name,
                "ok": ok,
                "detail": detail,
                **extra,
            }
        )

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    add_check(
        "current_branch",
        branch.returncode == 0,
        branch.stdout.strip() or branch.stderr.strip(),
    )

    status = _git(["status", "--porcelain=v1", "--untracked-files=all"])
    status_lines = [line for line in status.stdout.splitlines() if line.strip()]
    add_check(
        "working_tree_clean",
        status.returncode == 0 and len(status_lines) == 0,
        "clean" if not status_lines else f"{len(status_lines)} pending item(s)",
        items=status_lines,
    )

    base_exists = _git(["rev-parse", "--verify", base_ref])
    add_check(
        "base_ref_exists",
        base_exists.returncode == 0,
        base_exists.stdout.strip() or base_exists.stderr.strip(),
    )

    if base_exists.returncode == 0:
        mb = _git(["merge-base", "HEAD", base_ref])
        merge_base = mb.stdout.strip() if mb.returncode == 0 else None
        add_check(
            "merge_base",
            mb.returncode == 0,
            merge_base or mb.stderr.strip(),
        )

    if base_exists.returncode == 0:
        diff = _git(["diff", "--name-only", f"{base_ref}...HEAD"])
        changed_files = sorted(
            line.strip().replace("\\", "/")
            for line in diff.stdout.splitlines()
            if line.strip()
        )
        add_check(
            "changed_files_since_base",
            diff.returncode == 0,
            f"{len(changed_files)} changed file(s)",
            files=changed_files,
        )

        runtime_sensitive = [
            path
            for path in changed_files
            if any(
                path == prefix.rstrip("/") or path.startswith(prefix)
                for prefix in RUNTIME_SENSITIVE_PATH_PREFIXES
            )
        ]
        add_check(
            "runtime_sensitive_files_touched_review",
            True,
            (
                "none"
                if not runtime_sensitive
                else f"{len(runtime_sensitive)} runtime-sensitive file(s) touched; review diff before merge"
            ),
            files=runtime_sensitive,
        )

        outside_expected = [
            path
            for path in changed_files
            if not any(
                path == prefix.rstrip("/") or path.startswith(prefix)
                for prefix in EXPECTED_ATLAS_CONTEXT_PREFIXES
            )
        ]
        add_check(
            "outside_expected_atlas_context_scope_review",
            True,
            f"{len(outside_expected)} file(s) outside expected Atlas/context prefixes",
            files=outside_expected,
        )

    ignored = _git(["check-ignore", "-v", "config/local_roots.json"])
    add_check(
        "local_roots_json_ignored",
        ignored.returncode == 0,
        ignored.stdout.strip() or ignored.stderr.strip(),
    )

    example_tracked = _git(["ls-files", "--error-unmatch", "config/local_roots.example.json"])
    add_check(
        "local_roots_example_tracked",
        example_tracked.returncode == 0,
        example_tracked.stdout.strip() or example_tracked.stderr.strip(),
    )

    try:
        readiness = run_merge_readiness(
            config_path=config_path,
            smoke_query=smoke_query,
            json_output=False,
        )
        add_check(
            "context_merge_readiness",
            bool(readiness.get("ok")),
            f"checks={readiness.get('check_count')} failed={readiness.get('failed_count')}",
            readiness=readiness,
        )
    except Exception as exc:
        add_check(
            "context_merge_readiness",
            False,
            f"{exc.__class__.__name__}: {exc}",
        )

    failed = [check for check in report["checks"] if not check["ok"]]
    report["ok"] = len(failed) == 0
    report["failed_count"] = len(failed)
    report["check_count"] = len(report["checks"])

    if json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)

    return report


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _print_human(report: dict[str, Any]) -> None:
    print("Atlas migration pre-merge audit")
    print("=" * 40)
    print(f"ok: {report['ok']}")
    print(f"base: {report['base_ref']}")
    print(f"checks: {report['check_count']}")
    print(f"failed: {report['failed_count']}")
    print("")

    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"[{mark}] {check['name']}")
        print(f"       {check['detail']}")

        files = check.get("files")
        if files:
            for item in files[:40]:
                print(f"       - {item}")
            if len(files) > 40:
                print(f"       ... {len(files) - 40} more")

        items = check.get("items")
        if items:
            for item in items[:40]:
                print(f"       - {item}")
            if len(items) > 40:
                print(f"       ... {len(items) - 40} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Atlas migration pre-merge audit.")
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help="Base ref to compare against.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Local authorized roots config path.",
    )
    parser.add_argument(
        "--smoke-query",
        default=None,
        help="Optional context search_read smoke query.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report.",
    )

    args = parser.parse_args()

    report = run_premerge_audit(
        base_ref=args.base,
        config_path=Path(args.config),
        smoke_query=args.smoke_query,
        json_output=args.json,
    )

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
