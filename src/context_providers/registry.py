"""Context provider registry.

This registry is intentionally not wired into the live runtime yet.

It provides the middle layer between:
- AuthorizedRoot metadata
- provider adapters
- future LangGraph/capability routing

The registry resolves a root/provider/capability request to a safe adapter call.
It fails closed if the provider is unknown or the capability is unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BaseContextAdapter, CapabilityNotSupported
from .roots import AuthorizedRoot


class ProviderRegistryError(RuntimeError):
    """Base registry error."""


class ProviderNotRegistered(ProviderRegistryError):
    """Raised when no adapter is registered for a root provider type."""


@dataclass
class ContextProviderRegistry:
    """Capability-aware registry for context provider adapters."""

    _adapters: dict[str, BaseContextAdapter] = field(default_factory=dict)

    def register(self, adapter: BaseContextAdapter) -> None:
        provider_type = str(getattr(adapter, "provider_type", "")).strip()
        if not provider_type:
            raise ValueError("Adapter must define a non-empty provider_type.")
        self._adapters[provider_type] = adapter

    def unregister(self, provider_type: str) -> None:
        self._adapters.pop(provider_type, None)

    def registered_provider_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def adapter_for(self, root: AuthorizedRoot) -> BaseContextAdapter:
        adapter = self._adapters.get(root.provider_type)
        if adapter is None:
            raise ProviderNotRegistered(
                f"No context provider adapter registered for provider_type={root.provider_type!r}"
            )
        return adapter

    def require_capability(self, root: AuthorizedRoot, capability: str) -> BaseContextAdapter:
        if not root.enabled:
            raise CapabilityNotSupported(f"Authorized root is disabled: {root.id}")

        if not root.has_capability(capability):
            raise CapabilityNotSupported(
                f"Authorized root {root.id!r} does not allow capability: {capability}"
            )

        adapter = self.adapter_for(root)
        adapter.require(capability)
        return adapter

    def status(self, root: AuthorizedRoot) -> dict[str, Any]:
        capability = (
            "context.status"
            if root.has_capability("context.status")
            else "provider.status"
        )
        adapter = self.require_capability(root, capability)
        return adapter.status(root)

    def list(self, root: AuthorizedRoot, selector: str | None = None) -> list[dict[str, Any]]:
        adapter = self.require_capability(root, "context.list")
        return adapter.list(root, selector=selector)

    def read(self, root: AuthorizedRoot, selector: str) -> str:
        adapter = self.require_capability(root, "context.read")
        return adapter.read(root, selector)

    def search(self, root: AuthorizedRoot, query: str) -> list[dict[str, Any]]:
        adapter = self.require_capability(root, "context.search")
        return adapter.search(root, query)

    def public_status(self) -> dict[str, Any]:
        """Return UI/model-safe registry metadata without local filesystem paths."""

        providers: dict[str, dict[str, Any]] = {}
        for provider_type, adapter in sorted(self._adapters.items()):
            providers[provider_type] = {
                "provider_type": provider_type,
                "adapter": adapter.__class__.__name__,
                "supported_capabilities": list(adapter.normalized_capabilities()),
            }
        return {"providers": providers}


def build_default_registry() -> ContextProviderRegistry:
    """Build the current isolated default registry.

    This is safe to import in tests or future inspection tools.
    It is not yet connected to live LangGraph/tool routing.
    """

    from .adapters import LocalDirectoryAdapter

    registry = ContextProviderRegistry()
    local_adapter = LocalDirectoryAdapter()
    registry.register(local_adapter)

    # Obsidian vault roots can currently use the read-only local directory adapter
    # because Obsidian vaults are local markdown directories at this layer.
    registry.register(
        LocalDirectoryAdapter(
            provider_type="obsidian_vault",
        )
    )

    return registry


__all__ = [
    "ContextProviderRegistry",
    "ProviderNotRegistered",
    "ProviderRegistryError",
    "build_default_registry",
]
