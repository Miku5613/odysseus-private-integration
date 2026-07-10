"""Print local context root status.

Usage:
    python scripts/context_roots_status.py
    python scripts/context_roots_status.py --config config/local_roots.example.json

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

from src.context_providers import (
    CapabilityNotSupported,
    ProviderNotRegistered,
    build_default_registry,
    load_authorized_roots,
    verify_authorized_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect configured context provider roots.")
    parser.add_argument(
        "--config",
        default="config/local_roots.json",
        help="Path to local roots config. Defaults to config/local_roots.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    roots = load_authorized_roots(config_path)
    registry = build_default_registry()

    payload: dict[str, Any] = {
        "config": str(config_path),
        "root_count": len(roots),
        "registry": registry.public_status(),
        "roots": [],
    }

    for root in roots:
        verified = verify_authorized_root(root)
        root_payload = {
            **root.public_summary(),
            "verification": verified,
            "adapter_status": _adapter_status(registry, root),
        }
        payload["roots"].append(root_payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)

    return 0


def _adapter_status(registry, root) -> dict[str, Any]:
    try:
        adapter = registry.adapter_for(root)
    except ProviderNotRegistered as exc:
        return {
            "status": "not_registered",
            "detail": str(exc),
        }

    unsupported = [
        capability
        for capability in root.allowed_capabilities
        if not adapter.supports(capability)
    ]

    if unsupported:
        return {
            "status": "unsupported_capabilities",
            "adapter": adapter.__class__.__name__,
            "unsupported": unsupported,
        }

    try:
        status_capability = (
            "context.status"
            if root.has_capability("context.status")
            else "provider.status"
        )
        if root.has_capability(status_capability):
            registry.require_capability(root, status_capability)
    except CapabilityNotSupported as exc:
        return {
            "status": "capability_blocked",
            "adapter": adapter.__class__.__name__,
            "detail": str(exc),
        }

    return {
        "status": "ok",
        "adapter": adapter.__class__.__name__,
    }


def _print_human(payload: dict[str, Any]) -> None:
    print("Context Roots Status")
    print("====================")
    print(f"Config: {payload['config']}")
    print(f"Root count: {payload['root_count']}")
    print()

    providers = payload["registry"]["providers"]
    print("Registered providers:")
    if providers:
        for provider_type, info in providers.items():
            caps = ", ".join(info["supported_capabilities"])
            print(f"  - {provider_type}: {info['adapter']} [{caps}]")
    else:
        print("  none")
    print()

    if not payload["roots"]:
        print("No local roots configured.")
        print("Copy config/local_roots.example.json to config/local_roots.json and customize it locally.")
        return

    print("Roots:")
    for root in payload["roots"]:
        caps = ", ".join(root["allowed_capabilities"])
        verification = root["verification"]["status"]
        adapter_status = root["adapter_status"]["status"]
        print(f"  - {root['id']} ({root['display_name']})")
        print(f"    provider: {root['provider_type']}")
        print(f"    enabled: {root['enabled']}")
        print(f"    scope: {root['safety_scope']}")
        print(f"    capabilities: {caps}")
        print(f"    verification: {verification}")
        print(f"    adapter: {adapter_status}")
        if root["adapter_status"].get("unsupported"):
            print(f"    unsupported: {', '.join(root['adapter_status']['unsupported'])}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
