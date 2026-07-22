"""Workspace API - browse server directories to pick a tool workspace folder."""
import os
from fastapi import APIRouter, Request, HTTPException, Query

from src.auth_helpers import get_current_user
from src.tool_security import owner_is_admin_or_single_user

# Cap entries returned per directory (mirrors filesystem_tools._CODENAV_MAX_HITS).
# A huge directory shouldn't dump thousands of rows into the picker; the user can
# type/paste a path to jump straight in instead.
_MAX_BROWSE_DIRS = 500
_MAX_TREE_ENTRIES = 500


def _resolve_workspace_tree_path(root: str, relative: str = "") -> tuple[str, str]:
    from src.tool_execution import vet_workspace

    canonical_root = vet_workspace(str(root or "").strip())
    if not canonical_root:
        raise ValueError("Workspace root is not allowed")

    root_path = os.path.realpath(canonical_root)
    clean_relative = str(relative or "").replace("\\", "/").strip("/")
    candidate = os.path.realpath(os.path.join(root_path, clean_relative))
    try:
        if os.path.commonpath([root_path, candidate]) != root_path:
            raise ValueError("Tree path escapes the workspace root")
    except ValueError as exc:
        raise ValueError("Tree path escapes the workspace root") from exc
    if not os.path.isdir(candidate):
        raise ValueError("Tree path is not a directory")
    return root_path, candidate


def _workspace_tree_entries(root: str, relative: str = "") -> dict:
    root_path, target = _resolve_workspace_tree_path(root, relative)
    entries = []
    try:
        with os.scandir(target) as iterator:
            for entry in iterator:
                try:
                    if entry.name.startswith(".") or entry.is_symlink():
                        continue
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                    if not (is_directory or is_file):
                        continue
                    absolute = os.path.join(target, entry.name)
                    entries.append({
                        "name": entry.name,
                        "path": os.path.relpath(absolute, root_path).replace("\\", "/"),
                        "kind": "directory" if is_directory else "file",
                        "size": None if is_directory else entry.stat(follow_symlinks=False).st_size,
                    })
                except OSError:
                    continue
    except (PermissionError, OSError):
        entries = []

    entries.sort(key=lambda item: (item["kind"] != "directory", item["name"].casefold()))
    clean_relative = os.path.relpath(target, root_path).replace("\\", "/")
    if clean_relative == ".":
        clean_relative = ""
    return {
        "root": root_path,
        "relative": clean_relative,
        "entries": entries[:_MAX_TREE_ENTRIES],
        "truncated": len(entries) > _MAX_TREE_ENTRIES,
    }


def setup_workspace_routes():
    router = APIRouter(prefix="/api/workspace", tags=["workspace"])

    @router.get("/browse")
    def browse(request: Request, path: str = Query(default="")):
        """List subdirectories of `path` (default: home) so the UI can navigate
        the server filesystem and pick a workspace folder. Directories only.

        ADMIN-ONLY: this enumerates the server filesystem, so it is gated the
        same way the file/shell tools are (read_file/write_file/bash are in
        NON_ADMIN_BLOCKED_TOOLS). A non-admin who can't use those tools must not
        be able to map the host's directory tree either.
        """
        owner = get_current_user(request)
        if not owner_is_admin_or_single_user(owner):
            raise HTTPException(status_code=403, detail="Workspace browsing is admin-only")

        # Resolve symlinks so the reported path is canonical and the UI navigates
        # real directories (defends against symlink games in displayed paths).
        target = os.path.realpath(os.path.expanduser(path.strip() or "~"))
        if not os.path.isdir(target):
            target = os.path.realpath(os.path.expanduser("~"))

        dirs = []
        try:
            with os.scandir(target) as it:
                for entry in it:
                    try:
                        # Don't follow symlinks when classifying - a symlinked
                        # dir is skipped rather than letting the browser wander
                        # off via a link. Hidden entries are omitted.
                        if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                            # Build the child path server-side with os.path.join
                            # so it's correct on Windows (backslashes) and Linux.
                            dirs.append({"name": entry.name, "path": os.path.join(target, entry.name)})
                    except OSError:
                        continue
        except (PermissionError, OSError):
            dirs = []

        dirs_sorted = sorted(dirs, key=lambda d: d["name"].lower())
        truncated = len(dirs_sorted) > _MAX_BROWSE_DIRS
        parent = os.path.dirname(target)
        from src.tool_execution import vet_workspace
        return {
            "path": target,
            "parent": parent if parent and parent != target else None,
            "dirs": dirs_sorted[:_MAX_BROWSE_DIRS],
            "truncated": truncated,
            # Whether this directory may be bound as a workspace (filesystem
            # roots and sensitive dirs may be browsed through but not chosen).
            "selectable": vet_workspace(target) is not None,
        }

    @router.get("/vet")
    def vet(request: Request, path: str = Query(default="")):
        """Validate a workspace path without binding it.

        The UI calls this before persisting a manually typed path (/workspace
        set) so a typo, file path, deleted folder, sensitive dir, or filesystem
        root is rejected up front with the canonical path returned on success,
        instead of being stored client-side and silently dropped at chat time.
        Admin-gated like /browse: it confirms path existence on the host.
        """
        owner = get_current_user(request)
        if not owner_is_admin_or_single_user(owner):
            raise HTTPException(status_code=403, detail="Workspace selection is admin-only")
        from src.tool_execution import vet_workspace
        resolved = vet_workspace(path)
        return {"ok": resolved is not None, "path": resolved}

    @router.get("/tree")
    def tree(
        request: Request,
        root: str = Query(default=""),
        relative: str = Query(default=""),
    ):
        """Return one read-only, workspace-confined level for Atlas."""
        owner = get_current_user(request)
        if not owner_is_admin_or_single_user(owner):
            raise HTTPException(status_code=403, detail="Workspace browsing is admin-only")
        try:
            return _workspace_tree_entries(root, relative)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
