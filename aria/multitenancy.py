"""
aria.multitenancy — Multi-tenant isolation for ARIA deployments.

Provides tenant-scoped access to storage and auditing, ensuring that
one tenant cannot read or write another tenant's inference records.

Usage::

    from aria.multitenancy import TenantRegistry, TenantContext

    registry = TenantRegistry()
    registry.register("acme-corp", api_key="sk-acme-...", storage=storage_acme)
    registry.register("globex",    api_key="sk-globex-.", storage=storage_globex)

    # Resolve tenant from API key
    ctx = registry.resolve("sk-acme-...")
    auditor = ctx.get_auditor()
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface
    from .auditor import InferenceAuditor

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TenantConfig:
    """Configuration for a single tenant."""
    tenant_id:   str
    api_key:     str              # Full key (kept in memory, hashed for lookup)
    storage:     Any | None = None
    metadata:    dict = field(default_factory=dict)
    active:      bool = True

    @property
    def api_key_prefix(self) -> str:
        return self.api_key[:8] if len(self.api_key) >= 8 else self.api_key

    @property
    def api_key_hash(self) -> str:
        return hashlib.sha256(self.api_key.encode()).hexdigest()


@dataclass
class TenantContext:
    """Runtime context for a resolved tenant request."""
    tenant_id:  str
    config:     TenantConfig
    _auditor:   Any | None = field(default=None, repr=False)

    def get_storage(self) -> Any | None:
        return self.config.storage

    def get_auditor(self) -> Any | None:
        return self._auditor

    def set_auditor(self, auditor: Any) -> None:
        self._auditor = auditor


# ---------------------------------------------------------------------------
# TenantRegistry
# ---------------------------------------------------------------------------

class TenantRegistry:
    """Registry of tenants with API-key-based resolution.

    Stores tenant configurations and resolves incoming API keys to tenants.
    All lookups are O(1) via a hash map.
    """

    def __init__(self) -> None:
        self._by_id:   dict[str, TenantConfig] = {}
        self._by_hash: dict[str, TenantConfig] = {}

    def register(
        self,
        tenant_id: str,
        api_key: str,
        storage: Any | None = None,
        metadata: dict | None = None,
        replace: bool = False,
    ) -> TenantConfig:
        """Register a new tenant.

        Args:
            tenant_id: Unique tenant identifier.
            api_key:   API key for this tenant.
            storage:   Storage backend scoped to this tenant.
            metadata:  Arbitrary metadata dict.
            replace:   If True, overwrite existing tenant registration.

        Returns:
            TenantConfig for the registered tenant.

        Raises:
            ValueError: If tenant_id already exists and replace=False.
        """
        if tenant_id in self._by_id and not replace:
            raise ValueError(f"Tenant '{tenant_id}' already registered")

        # Remove old hash entry if replacing
        if tenant_id in self._by_id:
            old_hash = self._by_id[tenant_id].api_key_hash
            self._by_hash.pop(old_hash, None)

        cfg = TenantConfig(
            tenant_id=tenant_id,
            api_key=api_key,
            storage=storage,
            metadata=metadata or {},
        )
        self._by_id[tenant_id]          = cfg
        self._by_hash[cfg.api_key_hash] = cfg
        return cfg

    def resolve(self, api_key: str) -> TenantContext | None:
        """Resolve an API key to a TenantContext.

        Args:
            api_key: The tenant's API key.

        Returns:
            TenantContext if found and active, else None.
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        cfg = self._by_hash.get(key_hash)
        if cfg is None or not cfg.active:
            return None
        return TenantContext(tenant_id=cfg.tenant_id, config=cfg)

    def get(self, tenant_id: str) -> TenantConfig | None:
        """Get tenant config by tenant_id."""
        return self._by_id.get(tenant_id)

    def deactivate(self, tenant_id: str) -> bool:
        """Disable a tenant (key still registered but not resolvable)."""
        cfg = self._by_id.get(tenant_id)
        if cfg:
            cfg.active = False
            return True
        return False

    def activate(self, tenant_id: str) -> bool:
        """Re-enable a deactivated tenant."""
        cfg = self._by_id.get(tenant_id)
        if cfg:
            cfg.active = True
            return True
        return False

    def list_tenants(self, active_only: bool = False) -> list[str]:
        """Return list of tenant IDs."""
        if active_only:
            return [tid for tid, cfg in self._by_id.items() if cfg.active]
        return list(self._by_id.keys())

    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate a cryptographically secure API key."""
        token = secrets.token_hex(24)
        return f"{prefix}-{token}"


# ---------------------------------------------------------------------------
# Convenience: TenantAwareAuditor proxy
# ---------------------------------------------------------------------------

class TenantAwareAuditor:
    """Wraps an InferenceAuditor to inject tenant_id into all record metadata.

    Args:
        auditor:   Underlying InferenceAuditor.
        tenant_id: Tenant identifier to inject.
    """

    def __init__(self, auditor: "InferenceAuditor", tenant_id: str) -> None:
        self._auditor  = auditor
        self._tenant   = tenant_id

    def record(
        self,
        model_id: str,
        input_data: dict,
        output_data: dict,
        confidence: float | None = None,
        latency_ms: int = 0,
        metadata: dict | None = None,
    ) -> str:
        meta = dict(metadata or {})
        meta["tenant_id"] = self._tenant
        return self._auditor.record(
            model_id, input_data, output_data,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=meta,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._auditor, name)
