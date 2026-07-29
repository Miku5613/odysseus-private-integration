from __future__ import annotations

import re


FILE_INSPECTION_TOOLS = {
    "get_workspace",
    "ls",
    "glob_files",
    "grep",
    "read_file",
    "read_file_around",
}

WORKSPACE_MUTATION_TOOLS = FILE_INSPECTION_TOOLS | {"write_file"}


def has_mutation_intent(user_text: str) -> bool:
    """Lock mutating tool scope without misreading explicit read-only clauses."""

    text = str(user_text or "").lower()
    text = re.sub(
        r"\b(?:do\s+not|don't|never|without)\s+"
        r"(?:\w+\s+){0,2}(?:write|edit|modify|change|delete|remove|create|run|execute)\w*\b",
        "",
        text,
    )
    return bool(re.search(
        r"\b(?:write|edit|modify|change|delete|remove|create|patch|rename|move|"
        r"install|execute|fix|update|send|schedule)\w*\b"
        r"|\brun\s+(?:a\s+|the\s+)?(?:command|script|program|tests?|build)\b",
        text,
    ))


def infer_tool_scope(user_text: str, workspace: str | None = None) -> str:
    """Infer a simple tool scope.

    Runtime v0.2 only supports file inspection. Later this can expand to:
    - web_research
    - atlas_read
    - atlas_write
    - memory
    - email
    """
    if has_mutation_intent(user_text):
        return "workspace_mutation"
    return "file_inspection"


def allowed_tools_for_scope(scope: str) -> set[str]:
    if scope == "file_inspection":
        return set(FILE_INSPECTION_TOOLS)
    if scope == "workspace_mutation":
        return set(WORKSPACE_MUTATION_TOOLS)

    return set()
