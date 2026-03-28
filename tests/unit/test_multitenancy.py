"""Tests for aria.multitenancy — TenantRegistry and TenantAwareAuditor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aria.multitenancy import (
    TenantAwareAuditor,
    TenantConfig,
    TenantContext,
    TenantRegistry,
)


# ---------------------------------------------------------------------------
# TenantConfig
# ---------------------------------------------------------------------------

class TestTenantConfig:
    def test_basic(self):
        cfg = TenantConfig(tenant_id="acme", api_key="sk-12345678abcd")
        assert cfg.tenant_id == "acme"
        assert cfg.active is True

    def test_api_key_prefix(self):
        cfg = TenantConfig(tenant_id="t", api_key="sk-abc12345xyz")
        assert cfg.api_key_prefix == "sk-abc12"

    def test_api_key_hash_consistent(self):
        cfg = TenantConfig(tenant_id="t", api_key="my-key")
        assert cfg.api_key_hash == cfg.api_key_hash

    def test_api_key_hash_different_keys(self):
        c1 = TenantConfig(tenant_id="a", api_key="key-1")
        c2 = TenantConfig(tenant_id="b", api_key="key-2")
        assert c1.api_key_hash != c2.api_key_hash


# ---------------------------------------------------------------------------
# TenantRegistry.register
# ---------------------------------------------------------------------------

class TestTenantRegistryRegister:
    def test_register_basic(self):
        reg = TenantRegistry()
        cfg = reg.register("acme", "sk-acme")
        assert cfg.tenant_id == "acme"

    def test_register_duplicate_raises(self):
        reg = TenantRegistry()
        reg.register("acme", "sk-acme")
        with pytest.raises(ValueError, match="acme"):
            reg.register("acme", "sk-acme-2")

    def test_register_replace(self):
        reg = TenantRegistry()
        reg.register("acme", "sk-old")
        cfg = reg.register("acme", "sk-new", replace=True)
        assert reg.resolve("sk-new") is not None
        assert reg.resolve("sk-old") is None

    def test_register_with_storage(self):
        storage = MagicMock()
        reg = TenantRegistry()
        cfg = reg.register("acme", "sk-acme", storage=storage)
        assert cfg.storage is storage

    def test_register_with_metadata(self):
        reg = TenantRegistry()
        cfg = reg.register("acme", "sk-acme", metadata={"plan": "enterprise"})
        assert cfg.metadata["plan"] == "enterprise"


# ---------------------------------------------------------------------------
# TenantRegistry.resolve
# ---------------------------------------------------------------------------

class TestTenantRegistryResolve:
    def test_resolve_valid_key(self):
        reg = TenantRegistry()
        reg.register("acme", "sk-acme-key")
        ctx = reg.resolve("sk-acme-key")
        assert ctx is not None
        assert ctx.tenant_id == "acme"

    def test_resolve_unknown_key(self):
        reg = TenantRegistry()
        assert reg.resolve("unknown-key") is None

    def test_resolve_inactive(self):
        reg = TenantRegistry()
        reg.register("acme", "sk-acme")
        reg.deactivate("acme")
        assert reg.resolve("sk-acme") is None

    def test_resolve_returns_tenant_context(self):
        reg = TenantRegistry()
        reg.register("acme", "sk-key")
        ctx = reg.resolve("sk-key")
        assert isinstance(ctx, TenantContext)

    def test_context_has_storage(self):
        storage = MagicMock()
        reg = TenantRegistry()
        reg.register("acme", "sk-key", storage=storage)
        ctx = reg.resolve("sk-key")
        assert ctx.get_storage() is storage


# ---------------------------------------------------------------------------
# TenantRegistry.deactivate / activate
# ---------------------------------------------------------------------------

class TestTenantActivation:
    def test_deactivate(self):
        reg = TenantRegistry()
        reg.register("acme", "sk-acme")
        assert reg.deactivate("acme") is True
        assert reg.get("acme").active is False

    def test_activate(self):
        reg = TenantRegistry()
        reg.register("acme", "sk-acme")
        reg.deactivate("acme")
        assert reg.activate("acme") is True
        assert reg.get("acme").active is True

    def test_deactivate_unknown(self):
        reg = TenantRegistry()
        assert reg.deactivate("unknown") is False

    def test_activate_unknown(self):
        reg = TenantRegistry()
        assert reg.activate("unknown") is False


# ---------------------------------------------------------------------------
# TenantRegistry.list_tenants
# ---------------------------------------------------------------------------

class TestListTenants:
    def test_list_all(self):
        reg = TenantRegistry()
        reg.register("a", "sk-a")
        reg.register("b", "sk-b")
        assert set(reg.list_tenants()) == {"a", "b"}

    def test_list_active_only(self):
        reg = TenantRegistry()
        reg.register("a", "sk-a")
        reg.register("b", "sk-b")
        reg.deactivate("b")
        assert reg.list_tenants(active_only=True) == ["a"]

    def test_empty(self):
        reg = TenantRegistry()
        assert reg.list_tenants() == []


# ---------------------------------------------------------------------------
# TenantRegistry.generate_api_key
# ---------------------------------------------------------------------------

class TestGenerateApiKey:
    def test_returns_string(self):
        key = TenantRegistry.generate_api_key()
        assert isinstance(key, str)

    def test_starts_with_prefix(self):
        key = TenantRegistry.generate_api_key(prefix="sk")
        assert key.startswith("sk-")

    def test_unique(self):
        keys = {TenantRegistry.generate_api_key() for _ in range(100)}
        assert len(keys) == 100

    def test_custom_prefix(self):
        key = TenantRegistry.generate_api_key(prefix="aria")
        assert key.startswith("aria-")


# ---------------------------------------------------------------------------
# TenantContext
# ---------------------------------------------------------------------------

class TestTenantContext:
    def test_set_and_get_auditor(self):
        cfg = TenantConfig(tenant_id="t", api_key="k")
        ctx = TenantContext(tenant_id="t", config=cfg)
        auditor = MagicMock()
        ctx.set_auditor(auditor)
        assert ctx.get_auditor() is auditor

    def test_get_storage_from_config(self):
        storage = MagicMock()
        cfg = TenantConfig(tenant_id="t", api_key="k", storage=storage)
        ctx = TenantContext(tenant_id="t", config=cfg)
        assert ctx.get_storage() is storage


# ---------------------------------------------------------------------------
# TenantAwareAuditor
# ---------------------------------------------------------------------------

class TestTenantAwareAuditor:
    def test_injects_tenant_id(self):
        auditor = MagicMock()
        ta = TenantAwareAuditor(auditor, tenant_id="acme")
        ta.record("model", {}, {}, metadata={"custom": "data"})
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["tenant_id"] == "acme"

    def test_preserves_existing_metadata(self):
        auditor = MagicMock()
        ta = TenantAwareAuditor(auditor, tenant_id="acme")
        ta.record("model", {}, {}, metadata={"custom": "data"})
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["custom"] == "data"

    def test_none_metadata_handled(self):
        auditor = MagicMock()
        ta = TenantAwareAuditor(auditor, tenant_id="acme")
        ta.record("model", {}, {}, metadata=None)
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["tenant_id"] == "acme"

    def test_proxies_other_attrs(self):
        auditor = MagicMock()
        auditor.flush = MagicMock(return_value="flushed")
        ta = TenantAwareAuditor(auditor, tenant_id="t")
        assert ta.flush() == "flushed"

    def test_passes_confidence_latency(self):
        auditor = MagicMock()
        ta = TenantAwareAuditor(auditor, tenant_id="t")
        ta.record("model", {}, {}, confidence=0.9, latency_ms=100)
        kwargs = auditor.record.call_args[1]
        assert kwargs["confidence"] == pytest.approx(0.9)
        assert kwargs["latency_ms"] == 100
