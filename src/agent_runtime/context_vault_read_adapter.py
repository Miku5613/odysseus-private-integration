from __future__ import annotations

from pathlib import Path

from src.context_providers.runtime_tool import (
    RuntimeContextToolLimits,
    context_tool_call,
)

from .obsidian_read_adapter import ObsidianReadError


DEFAULT_VAULT_ROOT_ID = "ideaverse_vault"
DEFAULT_ROOT_CONFIG = Path("config/local_roots.json")

_PROVIDER_READ_LIMITS = RuntimeContextToolLimits(
    max_search_results=8,
    max_read_files=1,
    max_chars_per_file=512_000,
    max_total_chars=512_000,
)


class ContextVaultReadError(ObsidianReadError):
    """Read-only context-provider vault failure."""


class ContextVaultReadAdapter:
    """Compatibility adapter for an authorized context-provider vault root.

    Exposes only list_dir/read_note for existing knowledge indexes.
    It does not expose write, edit, create, delete, move, or rename operations.
    """

    def __init__(
        self,
        root_id: str = DEFAULT_VAULT_ROOT_ID,
        config_path: str | Path = DEFAULT_ROOT_CONFIG,
    ) -> None:
        self.root_id = str(root_id or "").strip()
        self.config_path = Path(config_path)

        if not self.root_id:
            raise ContextVaultReadError(
                "A configured authorized context root_id is required."
            )

    def list_dir(self, vault_path: str = "") -> list[str]:
        selector = str(vault_path or "").strip().replace("\\", "/")

        try:
            result = context_tool_call(
                operation="list",
                root_id=self.root_id,
                selector=selector or ".",
                config_path=self.config_path,
            )

            items = result.get("items")
            if not isinstance(items, list):
                raise ContextVaultReadError(
                    "Context-provider list returned an invalid items payload."
                )

            entries: list[str] = []

            for item in items:
                if not isinstance(item, dict):
                    raise ContextVaultReadError(
                        "Context-provider list returned a non-object item."
                    )

                name = str(item.get("name") or "").strip()
                item_type = str(item.get("type") or "").strip().lower()

                if not name:
                    continue

                if item_type == "directory":
                    entries.append(name.rstrip("/") + "/")
                elif item_type == "file":
                    entries.append(name)
                else:
                    raise ContextVaultReadError(
                        f"Context-provider list returned unsupported item type: {item_type!r}"
                    )

            return entries

        except ContextVaultReadError:
            raise
        except Exception as exc:
            raise ContextVaultReadError(
                f"Context-provider list failed for authorized root {self.root_id!r}: {exc}"
            ) from exc

    def read_note(self, vault_path: str) -> str:
        selector = str(vault_path or "").strip().replace("\\", "/")

        if not selector:
            raise ContextVaultReadError("Cannot read an empty vault selector.")

        try:
            result = context_tool_call(
                operation="read",
                root_id=self.root_id,
                selector=selector,
                config_path=self.config_path,
                limits=_PROVIDER_READ_LIMITS,
            )

            if result.get("truncated") is True:
                raise ContextVaultReadError(
                    f"Context-provider read was truncated for selector: {selector}"
                )

            text = result.get("text")
            if not isinstance(text, str):
                raise ContextVaultReadError(
                    "Context-provider read returned a non-text payload."
                )

            return text

        except ContextVaultReadError:
            raise
        except Exception as exc:
            raise ContextVaultReadError(
                f"Context-provider read failed for authorized root {self.root_id!r}: {exc}"
            ) from exc


__all__ = [
    "ContextVaultReadAdapter",
    "ContextVaultReadError",
    "DEFAULT_ROOT_CONFIG",
    "DEFAULT_VAULT_ROOT_ID",
]