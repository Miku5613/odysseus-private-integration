from __future__ import annotations

from pathlib import Path


TOOL_INDEX = Path("src/tool_index.py")
TOOL_SCHEMAS = Path("src/tool_schemas.py")
TOOL_SECURITY = Path("src/tool_security.py")


READ_ONLY_ATLAS_ACTIONS = {"list", "read", "search", "backlinks", "graph"}
MUTATING_ATLAS_ACTIONS = {"write", "append", "delete"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _manage_atlas_index_line() -> str:
    for line in _read(TOOL_INDEX).splitlines():
        if '"manage_atlas"' in line:
            return line.lower()
    raise AssertionError("manage_atlas description not found in tool_index.py")


def _manage_atlas_schema_block() -> str:
    text = _read(TOOL_SCHEMAS)
    marker = '"name": "manage_atlas"'
    start = text.index(marker)

    next_function = text.find('"type": "function"', start + len(marker))
    if next_function == -1:
        return text[start:].lower()

    return text[start:next_function].lower()


def test_manage_atlas_model_schema_is_read_only_for_current_runtime():
    block = _manage_atlas_schema_block()

    assert '"name": "manage_atlas"' in block
    assert '"enum": ["list", "read", "search", "backlinks", "graph"]' in block

    forbidden_schema_fragments = [
        '"enum": ["list", "read", "write"',
        '"write", "append", "delete"',
        '"delete", "search"',
        "action='write'",
        "'write' creates",
        "'append' adds",
        "'delete' removes",
        "action='delete'",
        "for read/write/append/delete",
        "markdown body (for write/append)",
    ]

    for fragment in forbidden_schema_fragments:
        assert fragment not in block


def test_manage_atlas_description_does_not_advertise_current_write_actions():
    line = _manage_atlas_index_line()
    block = _manage_atlas_schema_block()
    combined = f"{line}\n{block}"

    assert "manage_atlas" in combined
    assert "read-only" in combined
    assert "write profile" in combined or "write graph" in combined

    forbidden_advertising = [
        "read/write the user's atlas",
        "action='write'",
        "'write' creates",
        "'append' adds",
        "'delete' removes",
        "action='delete'",
        "for read/write/append/delete",
        "markdown body (for write/append)",
    ]

    for phrase in forbidden_advertising:
        assert phrase not in combined


def test_manage_atlas_remains_defensively_classified_as_mutator():
    security = _read(TOOL_SECURITY)

    assert '"manage_atlas"' in security
    assert "_PLAN_MODE_KNOWN_MUTATORS" in security
    assert "NON_ADMIN_BLOCKED_TOOLS" in security


def test_static_atlas_action_risk_buckets_are_disjoint():
    assert READ_ONLY_ATLAS_ACTIONS
    assert MUTATING_ATLAS_ACTIONS
    assert READ_ONLY_ATLAS_ACTIONS.isdisjoint(MUTATING_ATLAS_ACTIONS)
