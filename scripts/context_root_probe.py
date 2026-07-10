"""Probe configured context roots with safe read-only operations.

Usage:
    python scripts/context_root_probe.py
    python scripts/context_root_probe.py --root odysseus_repo --list .
    python scripts/context_root_probe.py --root odysseus_repo --read README.md
    python scripts/context_root_probe.py --root ideaverse_vault --search Atlas
    python scripts/context_root_probe.py --json --root odysseus_repo --search context

This is an inspection utility only. It does not start the server, mutate config,
or connect anything to LangGraph/tool routing.
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

from src.context_providers import build_default_registry, load_authorized_roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe configured context roots.")
    parser.add_argument(
        "--config",
        default="config/local_roots.json",
        help="Path to local roots config. Defaults to config/local_roots.json.",
    )
    parser.add_argument(
        "--root",
        help="Root id to probe. Defaults to the first configured enabled root.",
    )
    parser.add_argument(
        "--list",
        dest="list_selector",
        default=None,
        help="List a directory selector inside the root.",
    )
    parser.add_argument(
        "--read",
        dest="read_selector",
        default=None,
        help="Read a text file selector inside the root.",
    )
    parser.add_argument(
        "--search",
        dest="search_query",
        default=None,
        help="Search text files inside the root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )

    args = parser.parse_args()

    roots = load_authorized_roots(Path(args.config))
    registry = build_default_registry()

    root = _select_root(roots, args.root)
    if root is None:
        payload = {
            "ok": False,
            "error": "No matching enabled root found.",
            "requested_root": args.root,
            "available_roots": [r.id for r in roots],
        }
        _emit(payload, args.json)
        return 2

    operations = []
    if args.list_selector is not None:
        operations.append(("list", args.list_selector))
    if args.read_selector is not None:
        operations.append(("read", args.read_selector))
    if args.search_query is not None:
        operations.append(("search", args.search_query))

    if not operations:
        operations.append(("status", None))
        operations.append(("list", "."))

    payload: dict[str, Any] = {
        "ok": True,
        "root": root.public_summary(),
        "operations": [],
    }

    for op, value in operations:
        try:
            if op == "status":
                result = registry.status(root)
            elif op == "list":
                result = registry.list(root, selector=value)
            elif op == "read":
                text = registry.read(root, value)
                result = {
                    "selector": value,
                    "characters": len(text),
                    "preview": text[:2000],
                }
            elif op == "search":
                result = registry.search(root, value)
            else:
                raise ValueError(f"Unknown operation: {op}")

            payload["operations"].append(
                {
                    "operation": op,
                    "input": value,
                    "ok": True,
                    "result": result,
                }
            )
        except Exception as exc:
            payload["operations"].append(
                {
                    "operation": op,
                    "input": value,
                    "ok": False,
                    "error": exc.__class__.__name__,
                    "detail": str(exc),
                }
            )
            payload["ok"] = False

    _emit(payload, args.json)
    return 0 if payload["ok"] else 1


def _select_root(roots, root_id: str | None):
    enabled = [root for root in roots if root.enabled]
    if root_id:
        for root in enabled:
            if root.id == root_id:
                return root
        return None
    return enabled[0] if enabled else None


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not payload.get("ok"):
        print("Context Root Probe: FAILED")
        print(f"Error: {payload.get('error', 'unknown')}")
        if payload.get("requested_root"):
            print(f"Requested root: {payload['requested_root']}")
        if payload.get("available_roots") is not None:
            print(f"Available roots: {', '.join(payload['available_roots'])}")
        return

    root = payload["root"]
    print("Context Root Probe")
    print("==================")
    print(f"Root: {root['id']} ({root['display_name']})")
    print(f"Provider: {root['provider_type']}")
    print()

    for op in payload["operations"]:
        print(f"Operation: {op['operation']}")
        print(f"Input: {op['input']}")
        print(f"OK: {op['ok']}")

        if not op["ok"]:
            print(f"Error: {op['error']}")
            print(f"Detail: {op['detail']}")
            print()
            continue

        result = op["result"]

        if op["operation"] == "status":
            for key, value in result.items():
                print(f"  {key}: {value}")

        elif op["operation"] == "list":
            for item in result[:50]:
                marker = "[D]" if item["type"] == "directory" else "[F]"
                print(f"  {marker} {item['path']}")
            if len(result) > 50:
                print(f"  ... {len(result) - 50} more")

        elif op["operation"] == "read":
            print(f"  selector: {result['selector']}")
            print(f"  characters: {result['characters']}")
            print("  preview:")
            for line in result["preview"].splitlines()[:40]:
                print(f"    {line}")

        elif op["operation"] == "search":
            for item in result[:20]:
                print(f"  [F] {item['path']}")
                snippet = item["snippet"].replace("\\n", " ")
                print(f"      {snippet[:240]}")
            if len(result) > 20:
                print(f"  ... {len(result) - 20} more")

        print()


if __name__ == "__main__":
    raise SystemExit(main())
