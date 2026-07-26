"""Odysseus model-registry adapter for the standalone OpenCode runtime."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

from core.database import ModelEndpoint, SessionLocal


class OpenCodeModelRegistryError(RuntimeError):
    """Raised when a model selection is not authorized by Odysseus."""


@dataclass(frozen=True, slots=True)
class OpenCodeModelSelection:
    endpoint_id: str
    provider_id: str
    provider_name: str
    base_url: str
    model_id: str
    models: tuple[str, ...]
    api_key: str = ""


def _parse_model_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            value = re.split(r"[\r\n,]+", text)

    if not isinstance(value, (list, tuple, set)):
        value = [value]

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        model_id = str(item or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        result.append(model_id)
    return result


def visible_model_ids(endpoint: Any) -> tuple[str, ...]:
    cached = _parse_model_ids(getattr(endpoint, "cached_models", None))
    pinned = _parse_model_ids(getattr(endpoint, "pinned_models", None))
    hidden = set(_parse_model_ids(getattr(endpoint, "hidden_models", None)))

    result: list[str] = []
    seen: set[str] = set()
    for model_id in (*cached, *pinned):
        if model_id in hidden or model_id in seen:
            continue
        seen.add(model_id)
        result.append(model_id)
    return tuple(result)


def normalize_provider_base_url(value: str) -> str:
    base_url = str(value or "").strip().rstrip("/")
    if not base_url:
        raise OpenCodeModelRegistryError(
            "The selected Odysseus model endpoint has no base URL."
        )

    lowered = base_url.lower()
    if lowered.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")].rstrip("/")
    elif lowered.endswith("/responses"):
        base_url = base_url[: -len("/responses")].rstrip("/")

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if (
        (parsed.port == 11434 or "ollama" in host)
        and not parsed.path.rstrip("/").endswith("/v1")
    ):
        parsed = parsed._replace(
            path=(parsed.path.rstrip("/") + "/v1") or "/v1"
        )
        base_url = urlunparse(parsed).rstrip("/")
    return base_url


def provider_id_for_endpoint(endpoint_id: str) -> str:
    normalized = str(endpoint_id or "").strip()
    if not normalized:
        raise OpenCodeModelRegistryError(
            "The OpenCode model endpoint ID must not be empty."
        )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return "odysseus_%s" % digest


def resolve_model_selection(
    *,
    endpoint_id: str,
    model_id: str,
    session_factory: Callable[[], Any] = SessionLocal,
) -> OpenCodeModelSelection:
    selected_endpoint = str(endpoint_id or "").strip()
    selected_model = str(model_id or "").strip()

    if not selected_endpoint:
        raise OpenCodeModelRegistryError(
            "Choose an Odysseus model endpoint for OpenCode."
        )
    if not selected_model:
        raise OpenCodeModelRegistryError(
            "Choose an Odysseus model for OpenCode."
        )

    db = session_factory()
    try:
        endpoint = (
            db.query(ModelEndpoint)
            .filter(
                ModelEndpoint.id == selected_endpoint,
                ModelEndpoint.is_enabled == True,
            )
            .first()
        )
        if endpoint is None:
            raise OpenCodeModelRegistryError(
                "The selected endpoint is unavailable or disabled."
            )
        if str(getattr(endpoint, "model_type", None) or "llm") != "llm":
            raise OpenCodeModelRegistryError(
                "OpenCode requires an enabled LLM endpoint."
            )

        models = visible_model_ids(endpoint)
        if selected_model not in models:
            raise OpenCodeModelRegistryError(
                "The selected model is not enabled on that endpoint."
            )

        endpoint_name = str(
            getattr(endpoint, "name", None) or "Model Endpoint"
        ).strip()

        return OpenCodeModelSelection(
            endpoint_id=selected_endpoint,
            provider_id=provider_id_for_endpoint(selected_endpoint),
            provider_name="Odysseus · %s" % endpoint_name,
            base_url=normalize_provider_base_url(
                str(getattr(endpoint, "base_url", "") or "")
            ),
            model_id=selected_model,
            models=models,
            api_key=str(getattr(endpoint, "api_key", "") or ""),
        )
    finally:
        db.close()
