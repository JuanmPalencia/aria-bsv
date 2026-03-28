"""Tests that aria.__init__ exports are importable and versions match."""

from __future__ import annotations

import importlib

import pytest


class TestAriaInit:
    def test_version_is_string(self):
        import aria
        assert isinstance(aria.__version__, str)
        assert aria.__version__ >= "0.3.0"

    def test_core_exports(self):
        from aria import (
            AuditConfig,
            AuditRecord,
            InferenceAuditor,
            SQLiteStorage,
            StorageInterface,
        )
        assert AuditConfig is not None
        assert InferenceAuditor is not None
        assert SQLiteStorage is not None
        assert StorageInterface is not None
        assert AuditRecord is not None

    def test_error_exports(self):
        from aria import (
            ARIAError,
            ARIAConfigError,
            ARIABroadcastError,
            ARIAStorageError,
        )
        assert issubclass(ARIAConfigError, ARIAError)
        assert issubclass(ARIABroadcastError, ARIAError)
        assert issubclass(ARIAStorageError, ARIAError)

    def test_crypto_exports(self):
        from aria import hash_object, hash_file, canonical_json, ARIAMerkleTree
        assert callable(hash_object)
        assert callable(hash_file)
        assert callable(canonical_json)
        assert ARIAMerkleTree is not None

    def test_quick_importable(self):
        from aria.quick import ARIAQuick, EpochSummary, DriftSummary, quick_audit
        assert ARIAQuick is not None
        assert EpochSummary is not None
        assert DriftSummary is not None
        assert callable(quick_audit)

    def test_compliance_importable(self):
        from aria.compliance import (
            ComplianceChecker,
            ComplianceReport,
            Regulation,
            CheckSeverity,
        )
        assert ComplianceChecker is not None
        assert Regulation.BRC121 is not None
        assert Regulation.EU_AI is not None
        assert Regulation.GDPR is not None

    def test_metrics_importable(self):
        from aria.metrics import ARIAMetrics
        m = ARIAMetrics(namespace="test_init", system_id="sys")
        assert m.system_id == "sys"
        assert isinstance(m.prometheus_available, bool)

    def test_drift_importable(self):
        from aria.drift import DriftDetector, ks_statistic, js_divergence, kl_divergence
        assert DriftDetector is not None
        assert callable(ks_statistic)
        assert callable(js_divergence)
        assert callable(kl_divergence)

    def test_events_importable(self):
        from aria.events import InMemoryEventBus, EventType, ARIAEvent
        bus = InMemoryEventBus()
        event = ARIAEvent(type=EventType.RECORD_CREATED, data={"x": 1})
        bus.publish(event)
        assert len(bus.history) == 1

    def test_alerts_importable(self):
        from aria.alerts import (
            LogAlertChannel,
            SlackAlertChannel,
            WebhookAlertChannel,
            PagerDutyChannel,
            MultiAlertChannel,
            EmailAlertChannel,
        )
        assert LogAlertChannel is not None
        assert SlackAlertChannel is not None

    def test_analytics_importable(self):
        from aria.analytics import CrossEpochAnalytics
        assert CrossEpochAnalytics is not None

    def test_watchdog_importable(self):
        from aria.watchdog import WatchdogDaemon
        assert WatchdogDaemon is not None

    def test_reporting_importable(self):
        from aria.reporting import ReportGenerator
        assert ReportGenerator is not None

    def test_anchoring_importable(self):
        from aria.anchoring import AnchorPayload, MultiChainAnchor
        assert AnchorPayload is not None
        assert MultiChainAnchor is not None

    def test_contracts_importable(self):
        from aria.contracts.bonding import OperatorBondingContract
        from aria.contracts.notarization import EpochNotarization
        from aria.contracts.registry import ARIARegistry
        assert OperatorBondingContract is not None
        assert EpochNotarization is not None
        assert ARIARegistry is not None

    def test_zk_importable(self):
        from aria.zk import MockProver, EpochStatement
        assert MockProver is not None
        assert EpochStatement is not None
