from __future__ import annotations

import base64
import json
import threading
from typing import Any, Iterator
import urllib.error
import urllib.parse
import urllib.request

from .config import OpenCodeRuntimeConfig


class OpenCodeTransportError(RuntimeError):
    """Raised when the external OpenCode server rejects or loses a request."""


class OpenCodeHttpTransport:
    """Small standard-library transport for the OpenCode headless server."""

    def __init__(self, config: OpenCodeRuntimeConfig):
        self.config = config

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self.config.username or self.config.password:
            raw = (
                f"{self.config.username}:{self.config.password}"
            ).encode("utf-8")
            headers["Authorization"] = (
                "Basic " + base64.b64encode(raw).decode("ascii")
            )
        return headers

    def request_json(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        *,
        timeout: float | None = None,
    ) -> tuple[int, Any]:
        url = self.config.base_url + path
        data = None
        headers = self._headers()
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout or self.config.request_timeout_seconds,
            ) as response:
                status = int(response.status)
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenCodeTransportError(
                f"OpenCode HTTP {exc.code} for {method} {path}: "
                f"{body[:500]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OpenCodeTransportError(
                f"OpenCode request failed for {method} {path}: {exc}"
            ) from exc

        if not body.strip():
            return status, None
        try:
            return status, json.loads(body)
        except json.JSONDecodeError:
            return status, body

    def health(self) -> dict[str, Any]:
        status, value = self.request_json("GET", "/global/health")
        if status != 200 or not isinstance(value, dict):
            raise OpenCodeTransportError(
                "OpenCode health endpoint returned an invalid response."
            )
        return value

    def create_session(self, title: str) -> str:
        status, value = self.request_json(
            "POST",
            "/session",
            {"title": title},
        )
        if status not in {200, 201}:
            raise OpenCodeTransportError(
                f"OpenCode session creation returned HTTP {status}."
            )
        if isinstance(value, dict) and isinstance(value.get("data"), dict):
            value = value["data"]
        session_id = value.get("id") if isinstance(value, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise OpenCodeTransportError(
                "OpenCode session creation returned no session id."
            )
        return session_id

    def submit_prompt(self, session_id: str, message: str) -> None:
        encoded = urllib.parse.quote(session_id, safe="")
        payload = {
            "agent": self.config.agent,
            "system": self.config.system_prompt,
            "model": {
                "providerID": self.config.provider_id,
                "modelID": self.config.model_id,
            },
            "parts": [{"type": "text", "text": message}],
        }
        status, _ = self.request_json(
            "POST",
            f"/session/{encoded}/prompt_async",
            payload,
        )
        if status not in {200, 202, 204}:
            raise OpenCodeTransportError(
                f"OpenCode prompt admission returned HTTP {status}."
            )

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        encoded = urllib.parse.quote(session_id, safe="")
        status, value = self.request_json(
            "GET",
            f"/session/{encoded}/message?limit=100",
        )
        if status != 200:
            raise OpenCodeTransportError(
                f"OpenCode message retrieval returned HTTP {status}."
            )
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            value = value["data"]
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def abort_session(self, session_id: str) -> bool:
        encoded = urllib.parse.quote(session_id, safe="")
        status, value = self.request_json(
            "POST",
            f"/session/{encoded}/abort",
        )
        return status in {200, 202, 204} and value is not False

    def delete_session(self, session_id: str) -> bool:
        encoded = urllib.parse.quote(session_id, safe="")
        status, value = self.request_json(
            "DELETE",
            f"/session/{encoded}",
        )
        return status == 200 and value is True

    def reply_permission(
        self,
        request_id: str,
        reply: str,
        *,
        directory: str = "",
        workspace: str = "",
        message: str = "",
    ) -> bool:
        normalized_reply = str(reply or "").strip().lower()
        if normalized_reply not in {"once", "always", "reject"}:
            raise OpenCodeTransportError(
                f"Unsupported OpenCode permission reply: {reply}"
            )

        encoded = urllib.parse.quote(
            str(request_id or "").strip(),
            safe="",
        )
        # OpenCode's server is instance-scoped by the `directory` query
        # parameter. `workspace` is an Odysseus-side name and is not a valid
        # OpenCode query parameter. Use workspace only as a compatibility
        # fallback when an older/global permission event omitted directory.
        effective_directory = (
            str(directory or "").strip()
            or str(workspace or "").strip()
        )
        query = {}
        if effective_directory:
            query["directory"] = effective_directory
        suffix = (
            "?" + urllib.parse.urlencode(query)
            if query
            else ""
        )
        payload: dict[str, Any] = {
            "reply": normalized_reply,
        }
        if str(message or "").strip():
            payload["message"] = str(message).strip()[:500]

        status, value = self.request_json(
            "POST",
            f"/permission/{encoded}/reply{suffix}",
            payload,
        )
        return status in {200, 202, 204} and value is not False

    def iter_global_events(
        self,
        stop_event: threading.Event,
    ) -> Iterator[dict[str, Any]]:
        request = urllib.request.Request(
            self.config.base_url + "/event",
            headers=self._headers(accept="text/event-stream"),
            method="GET",
        )
        try:
            response = urllib.request.urlopen(
                request,
                timeout=max(30.0, self.config.request_timeout_seconds),
            )
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise OpenCodeTransportError(
                f"Could not open OpenCode event stream: {exc}"
            ) from exc

        data_lines: list[str] = []
        try:
            while not stop_event.is_set():
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data_lines:
                        payload = "\n".join(data_lines)
                        data_lines.clear()
                        try:
                            value = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, dict):
                            yield value
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        finally:
            response.close()
