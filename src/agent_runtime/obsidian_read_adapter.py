
from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


class ObsidianReadError(RuntimeError):
    pass


@dataclass
class ObsidianResponse:
    status: int
    path: str
    content_type: str
    body: str

    def json(self) -> Any:
        return json.loads(self.body)


class ObsidianReadAdapter:
    """Read-only adapter for Obsidian Local REST API.

    Safety rule:
    - This adapter only supports HTTP GET and HEAD.
    - It deliberately has no method for POST/PUT/PATCH/DELETE.
    - It deliberately does not execute Obsidian commands.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout_s: float = 5.0):
        self.base_url = (base_url or os.environ.get("OBSIDIAN_REST_BASE") or "https://127.0.0.1:27124").rstrip("/")
        self.api_key = (api_key or os.environ.get("OBSIDIAN_API_KEY") or "").strip()
        self.timeout_s = timeout_s
        self._ssl_context = ssl._create_unverified_context()

        if not self.api_key:
            raise ObsidianReadError("OBSIDIAN_API_KEY is not set.")

    def _request(self, method: str, path: str, *, accept: str = "application/json,text/markdown,text/plain,*/*") -> ObsidianResponse:
        method = method.upper().strip()
        if method not in {"GET", "HEAD"}:
            raise ObsidianReadError(f"Blocked non-read HTTP method: {method}")

        if not path.startswith("/"):
            path = "/" + path

        url = self.base_url + path
        req = urllib.request.Request(
            url,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": accept,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s, context=self._ssl_context) as r:
                raw = b"" if method == "HEAD" else r.read()
                return ObsidianResponse(
                    status=int(r.status),
                    path=path,
                    content_type=str(r.headers.get("Content-Type") or ""),
                    body=raw.decode("utf-8", errors="replace"),
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ObsidianReadError(f"HTTP {exc.code} for {method} {path}: {body[:500]}") from exc
        except Exception as exc:
            raise ObsidianReadError(f"{type(exc).__name__} for {method} {path}: {exc}") from exc

    @staticmethod
    def _clean_vault_path(path: str) -> str:
        raw = str(path or "").replace("\\", "/").strip()
        raw = raw.lstrip("/")
        parts = []
        for part in raw.split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                raise ObsidianReadError("Blocked path traversal in vault path.")
            parts.append(part)
        return "/".join(parts)

    def status(self) -> dict[str, Any]:
        res = self._request("GET", "/")
        return res.json()

    def active_note(self) -> str:
        res = self._request("GET", "/active/", accept="text/markdown,text/plain,*/*")
        return res.body

    def list_dir(self, vault_path: str = "") -> list[str]:
        clean = self._clean_vault_path(vault_path)
        path = "/vault/" + clean
        if clean and not path.endswith("/"):
            # The plugin accepts both, but trailing slash makes intent clearer for folders.
            path += "/"
        res = self._request("GET", path)
        data = res.json()
        files = data.get("files")
        if not isinstance(files, list):
            raise ObsidianReadError(f"Expected list response at {path}, got: {data!r}")
        return [str(x) for x in files]

    def read_note(self, vault_path: str) -> str:
        clean = self._clean_vault_path(vault_path)
        if not clean:
            raise ObsidianReadError("Cannot read empty vault path.")
        if clean.endswith("/"):
            raise ObsidianReadError("Cannot read a directory as a note.")
        res = self._request("GET", "/vault/" + urllib.parse.quote(clean, safe="/"), accept="text/markdown,text/plain,*/*")
        return res.body

    def head(self, vault_path: str) -> int:
        clean = self._clean_vault_path(vault_path)
        path = "/vault/" + urllib.parse.quote(clean, safe="/")
        res = self._request("HEAD", path)
        return res.status
