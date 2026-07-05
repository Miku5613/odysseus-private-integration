from __future__ import annotations


FILE_INSPECTION_TOOLS = {
    "get_workspace",
    "ls",
    "glob_files",
    "grep",
    "read_file",
    "read_file_around",
}


def infer_tool_scope(user_text: str, workspace: str | None = None) -> str:
    """Infer a simple tool scope.

    Runtime v0.2 only supports file inspection. Later this can expand to:
    - web_research
    - atlas_read
    - atlas_write
    - memory
    - email
    """
    return "file_inspection"


def allowed_tools_for_scope(scope: str) -> set[str]:
    if scope == "file_inspection":
        return set(FILE_INSPECTION_TOOLS)

    return set()
