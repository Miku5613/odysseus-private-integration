"""Base interfaces for future context provider adapters.

Provider adapters hide provider-specific details from the general runtime.
The runtime should ask for capabilities; adapters decide whether they support
those capabilities and how to execute them inside an authorized root.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Iterable


class CapabilityNotSupported(RuntimeError):
    """Raised when a provider adapter does not support a requested capability."""


class BaseContextAdapter(ABC):
    """Minimal capability-aware provider adapter base.

    Subclasses should declare `provider_type` and `supported_capabilities`.

    This base intentionally does not force every provider to implement every
    method. A provider can support read/list/search without supporting write.
    Unsupported operations fail closed through CapabilityNotSupported.
    """

    provider_type: str = ""
    supported_capabilities: frozenset[str] = frozenset()

    def supports(self, capability: str) -> bool:
        return capability in self.supported_capabilities

    def require(self, capability: str) -> None:
        if not self.supports(capability):
            raise CapabilityNotSupported(
                f"{self.__class__.__name__} does not support capability: {capability}"
            )

    def list(self, root: Any, selector: str | None = None) -> list[dict[str, Any]]:
        self.require("context.list")
        raise NotImplementedError

    def read(self, root: Any, selector: str) -> str:
        self.require("context.read")
        raise NotImplementedError

    def search(self, root: Any, query: str) -> list[dict[str, Any]]:
        self.require("context.search")
        raise NotImplementedError

    def status(self, root: Any) -> dict[str, Any]:
        if not self.supports("context.status") and not self.supports("provider.status"):
            raise CapabilityNotSupported(
                f"{self.__class__.__name__} does not support status capabilities"
            )
        raise NotImplementedError

    @classmethod
    def normalized_capabilities(cls) -> tuple[str, ...]:
        return tuple(sorted(str(c) for c in cls.supported_capabilities))


def normalize_capabilities(values: Iterable[str]) -> tuple[str, ...]:
    """Return stable, unique capability names."""

    return tuple(sorted({str(v).strip() for v in values if str(v).strip()}))
