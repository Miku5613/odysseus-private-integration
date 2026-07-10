"""Atlas/context provider merge-readiness check.

This script is intentionally boring.

It validates the local authorized-root/context-provider stack before merging the
Atlas migration branch. It does not call the model, does not use LangGraph, and
does not write to any configured root.

Checks:
- local root config can be loaded
- enabled roots verify
- default registry can resolve adapters
- runtime context wrapper can expose roots
- optional read-only search smoke does not crash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.context_providers.registry import build_default_registry
from src.context_providers.roots import load_authorized_roots, verify_authorized_root
from src.context_providers.runtime_tool import RuntimeContextToolLimits, context_tool_call


DEFAULT_CONFIG_PATH = Path("config/local_roots.json")


def run_merge_readiness(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    smoke_query: str | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    config_path = Path(config_path)

    def add_check(name: str, ok: bool, detail: str, **extra: Any) -> None:
        checks.append(
            {
                "name": name,
                "ok": ok,
                "detail": detail,
                **extra,
            }
        )

    if not config_path.exists():
        add_check(
            "config_exists",
            False,
            f"Missing local root config: {config_path}",
        )
        return _result(checks)

    add_check("config_exists", True, f"Found {config_path}")

    try:
        roots = load_authorized_roots(config_path)
    except Exception as exc:
        add_check(
            "config_loads",
            False,
            f"{exc.__class__.__name__}: {exc}",
        )
        return _result(checks)

    add_check("config_loads", True, f"Loaded {len(roots)} root(s)")

    enabled_roots = [root for root in roots if root.enabled]
    add_check(
        "enabled_roots_present",
        bool(enabled_roots),
        f"{len(enabled_roots)} enabled root(s)",
    )

    registry = build_default_registry()

    for root in enabled_roots:
        verification = verify_authorized_root(root)
        add_check(
            f"root_verify:{root.id}",
            verification.get("status") == "ok",
            str(verification),
            root_id=root.id,
            provider_type=root.provider_type,
        )

        try:
            adapter_status = registry.status(root)
            adapter_ok = (
                adapter_status.get("exists") is True
                and adapter_status.get("is_dir") is True
                and adapter_status.get("readable") is True
            )
            add_check(
                f"adapter_status:{root.id}",
                adapter_ok,
                str(adapter_status),
                root_id=root.id,
                provider_type=root.provider_type,
            )
        except Exception as exc:
            add_check(
                f"adapter_status:{root.id}",
                False,
                f"{exc.__class__.__name__}: {exc}",
                root_id=root.id,
                provider_type=root.provider_type,
            )

    try:
        roots_result = context_tool_call(operation="roots", config_path=config_path)
        add_check(
            "runtime_tool_roots",
            bool(roots_result.get("ok")),
            f"{len(roots_result.get('roots', []))} root(s) returned",
        )
    except Exception as exc:
        add_check(
            "runtime_tool_roots",
            False,
            f"{exc.__class__.__name__}: {exc}",
        )

    if smoke_query:
        try:
            smoke = context_tool_call(
                operation="search_read",
                query=smoke_query,
                config_path=config_path,
                limits=RuntimeContextToolLimits(
                    max_search_results=5,
                    max_read_files=2,
                    max_chars_per_file=1200,
                    max_total_chars=2400,
                ),
            )
            add_check(
                "runtime_tool_search_read_smoke",
                bool(smoke.get("ok")),
                (
                    f"query={smoke_query!r}; "
                    f"matches={smoke.get('match_count')}; "
                    f"documents={smoke.get('document_count')}"
                ),
            )
        except Exception as exc:
            add_check(
                "runtime_tool_search_read_smoke",
                False,
                f"{exc.__class__.__name__}: {exc}",
            )

    result = _result(checks)

    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_human(result)

    return result


def _result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    ok = all(check["ok"] for check in checks)
    return {
        "ok": ok,
        "check_count": len(checks),
        "failed_count": len([check for check in checks if not check["ok"]]),
        "checks": checks,
    }


def _print_human(result: dict[str, Any]) -> None:
    print("Atlas/context merge readiness")
    print("=" * 38)
    print(f"ok: {result['ok']}")
    print(f"checks: {result['check_count']}")
    print(f"failed: {result['failed_count']}")
    print("")

    for check in result["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"[{mark}] {check['name']}")
        print(f"       {check['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Atlas/context merge readiness.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to local authorized roots config.",
    )
    parser.add_argument(
        "--smoke-query",
        default=None,
        help="Optional read-only search_read smoke query.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    args = parser.parse_args()

    result = run_merge_readiness(
        config_path=Path(args.config),
        smoke_query=args.smoke_query,
        json_output=args.json,
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
