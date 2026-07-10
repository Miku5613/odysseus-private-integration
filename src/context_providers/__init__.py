"""Context provider foundation.

This package is intentionally isolated from the active runtime for now.

It defines portable root metadata, provider capability contracts, and a small
registry skeleton without wiring them into LangGraph, chat routes, or tool
execution.
"""

from .base import BaseContextAdapter, CapabilityNotSupported
from .registry import (
    ContextProviderRegistry,
    ProviderNotRegistered,
    ProviderRegistryError,
    build_default_registry,
)
from .roots import (
    AuthorizedRoot,
    RootConfigError,
    load_authorized_roots,
    verify_authorized_root,
)

__all__ = [
    "AuthorizedRoot",
    "BaseContextAdapter",
    "CapabilityNotSupported",
    "ContextProviderRegistry",
    "ProviderNotRegistered",
    "ProviderRegistryError",
    "RootConfigError",
    "build_default_registry",
    "load_authorized_roots",
    "verify_authorized_root",
]
