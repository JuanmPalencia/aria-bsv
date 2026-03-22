"""Tests for registry.storage.RegistryStorage."""

from __future__ import annotations

import pytest

from registry.schemas import EpochCreate, SystemCreate
from registry.storage import RegistryStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_KEY = "supersecret-key"

SYSTEM_DATA = SystemCreate(
    system_id="kairos-v2",
    system_name="KAIROS Triage System v2",
    operator_name="Palencia Research",
    eu_ai_act_risk_level="high",
    eu_ai_act_article="Annex III §5",
    deployment_context="Emergency medical dispatch — municipality of Valencia",
)


def _store() -> RegistryStorage:
    return RegistryStorage("sqlite://")


# ---------------------------------------------------------------------------
# System CRUD
# ---------------------------------------------------------------------------


class TestCreateSystem:
    def test_returns_system_read(self):
        r = _store().create_system(SYSTEM_DATA, API_KEY)
        assert r.system_id == "kairos-v2"
        assert r.operator_name == "Palencia Research"

    def test_registered_at_is_set(self):
        r = _store().create_system(SYSTEM_DATA, API_KEY)
        assert r.registered_at is not None

    def test_api_key_not_in_result(self):
        r = _store().create_system(SYSTEM_DATA, API_KEY)
        assert not hasattr(r, "api_key") and not hasattr(r, "api_key_hash")

    def test_stats_zero_on_create(self):
        r = _store().create_system(SYSTEM_DATA, API_KEY)
        assert r.total_epochs == 0
        assert r.total_records == 0
        assert r.last_epoch_at is None

    def test_eu_ai_act_fields_preserved(self):
        r = _store().create_system(SYSTEM_DATA, API_KEY)
        assert r.eu_ai_act_risk_level == "high"
        assert r.eu_ai_act_article == "Annex III §5"


class TestGetSystem:
    def test_returns_none_for_unknown(self):
        assert _store().get_system("nope") is None

    def test_returns_system_after_create(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        r = store.get_system("kairos-v2")
        assert r is not None
        assert r.system_id == "kairos-v2"


class TestListSystems:
    def test_empty_initially(self):
        assert _store().list_systems() == []

    def test_returns_all_registered(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        store.create_system(
            SystemCreate(system_id="urban-vs", system_name="Urban VS", operator_name="UGVSM"),
            API_KEY,
        )
        assert len(store.list_systems()) == 2


class TestVerifyApiKey:
    def test_correct_key_accepted(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        assert store.verify_api_key("kairos-v2", API_KEY) is True

    def test_wrong_key_rejected(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        assert store.verify_api_key("kairos-v2", "wrong-key") is False

    def test_unknown_system_rejected(self):
        assert _store().verify_api_key("unknown", API_KEY) is False


# ---------------------------------------------------------------------------
# Epoch recording
# ---------------------------------------------------------------------------

EPOCH_DATA = EpochCreate(
    epoch_id="ep_1742848200000_0001",
    open_txid="a" * 64,
    close_txid="b" * 64,
    records_count=7,
    merkle_root="sha256:" + "f" * 64,
    opened_at=1_742_848_200,
    closed_at=1_742_848_210,
)


class TestRecordEpoch:
    def test_returns_epoch_record(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        r = store.record_epoch("kairos-v2", EPOCH_DATA)
        assert r.epoch_id == EPOCH_DATA.epoch_id
        assert r.records_count == 7

    def test_epoch_appears_in_history(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        store.record_epoch("kairos-v2", EPOCH_DATA)
        history = store.list_epochs("kairos-v2")
        assert len(history) == 1
        assert history[0].epoch_id == EPOCH_DATA.epoch_id

    def test_stats_updated_after_epoch(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        store.record_epoch("kairos-v2", EPOCH_DATA)
        r = store.get_system("kairos-v2")
        assert r is not None
        assert r.total_epochs == 1
        assert r.total_records == 7
        assert r.last_epoch_at is not None

    def test_history_ordered_newest_first(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        store.record_epoch(
            "kairos-v2",
            EpochCreate(
                epoch_id="ep_old",
                open_txid="a" * 64,
                close_txid="b" * 64,
                opened_at=1_000_000,
                closed_at=1_000_001,
            ),
        )
        store.record_epoch(
            "kairos-v2",
            EpochCreate(
                epoch_id="ep_new",
                open_txid="c" * 64,
                close_txid="d" * 64,
                opened_at=2_000_000,
                closed_at=2_000_001,
            ),
        )
        history = store.list_epochs("kairos-v2")
        assert history[0].epoch_id == "ep_new"

    def test_list_epochs_empty_for_no_epochs(self):
        store = _store()
        store.create_system(SYSTEM_DATA, API_KEY)
        assert store.list_epochs("kairos-v2") == []
