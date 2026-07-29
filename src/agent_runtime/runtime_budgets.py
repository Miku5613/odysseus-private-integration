from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_VALUE_LIMIT = 64_000
DEFAULT_CHECKPOINT_LIMIT = 1_500_000
DEFAULT_PREVIEW_LIMIT = 4_000


class RuntimeArtifactStore:
    """Content-addressed storage for payloads too large for runtime state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def offload(self, value: Any, *, kind: str) -> dict[str, Any]:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{digest}.json"
        if not target.exists():
            temporary = self.root / f".{digest}.{os.getpid()}.tmp"
            temporary.write_bytes(encoded)
            try:
                temporary.replace(target)
            except FileExistsError:
                temporary.unlink(missing_ok=True)
        preview = encoded[:DEFAULT_PREVIEW_LIMIT].decode(
            "utf-8",
            errors="replace",
        )
        return {
            "offloaded": True,
            "kind": kind,
            "sha256": digest,
            "bytes": len(encoded),
            "preview": preview,
            "artifact_path": str(target),
            "content_type": "application/json",
        }


def apply_checkpoint_budgets(
    state: dict[str, Any],
    *,
    artifact_store: RuntimeArtifactStore,
    value_limit: int = DEFAULT_VALUE_LIMIT,
    checkpoint_limit: int = DEFAULT_CHECKPOINT_LIMIT,
) -> dict[str, Any]:
    bounded = dict(state)
    references = list(bounded.get("artifact_references") or [])

    dependencies: dict[str, Any] = {}
    for key, value in (bounded.get("dependency_results") or {}).items():
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded.encode("utf-8")) > value_limit:
            reference = artifact_store.offload(
                value,
                kind=f"dependency:{key}",
            )
            dependencies[str(key)] = reference
            references.append(reference)
        else:
            dependencies[str(key)] = value
    bounded["dependency_results"] = dependencies

    events: list[dict[str, Any]] = []
    for event in bounded.get("event_log") or []:
        encoded = json.dumps(event, ensure_ascii=False, default=str)
        if len(encoded.encode("utf-8")) > value_limit:
            reference = artifact_store.offload(event, kind="runtime_event")
            cursor = int((event.get("data") or {}).get("event_cursor") or 0)
            compact_event = {
                "type": event.get("type") or "runtime_artifact",
                "data": {
                    "event_cursor": cursor,
                    "payload_offloaded": True,
                    "artifact": reference,
                },
            }
            events.append(compact_event)
            references.append(reference)
        else:
            events.append(event)
    bounded["event_log"] = events
    bounded["artifact_references"] = _deduplicate_references(references)

    encoded_state = json.dumps(bounded, ensure_ascii=False, default=str)
    if len(encoded_state.encode("utf-8")) > checkpoint_limit:
        reference = artifact_store.offload(
            {
                "dependency_results": bounded.get("dependency_results"),
                "artifact_references": bounded.get("artifact_references"),
            },
            kind="checkpoint_overflow",
        )
        bounded["dependency_results"] = {
            "__checkpoint_overflow__": reference,
        }
        bounded["artifact_references"] = _deduplicate_references(
            [*bounded["artifact_references"], reference]
        )
    return bounded


def _deduplicate_references(
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        key = (
            str(reference.get("sha256") or ""),
            str(reference.get("artifact_path") or reference.get("href") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(reference)
    return result


__all__ = [
    "RuntimeArtifactStore",
    "apply_checkpoint_budgets",
]
