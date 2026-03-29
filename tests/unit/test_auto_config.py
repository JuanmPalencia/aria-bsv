"""Tests for aria.auto_config — Zero-config BSV setup."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aria.auto_config import (
    get_or_create_wif,
    load_keystore,
    save_keystore,
    get_address,
    _derive_key,
    _xor_encrypt,
    _xor_decrypt,
)


class TestXorEncrypt:
    def test_roundtrip(self):
        data = b"hello world"
        key = b"secret_key!"
        encrypted = _xor_encrypt(data, key)
        decrypted = _xor_decrypt(encrypted, key)
        assert decrypted == data

    def test_different_key_different_output(self):
        data = b"test"
        c1 = _xor_encrypt(data, b"key1_pad_enough")
        c2 = _xor_encrypt(data, b"key2_pad_enough")
        assert c1 != c2


class TestDeriveKey:
    def test_deterministic(self):
        k1 = _derive_key("test-pass")
        k2 = _derive_key("test-pass")
        assert k1 == k2

    def test_different_passphrase(self):
        k1 = _derive_key("pass1")
        k2 = _derive_key("pass2")
        assert k1 != k2

    def test_returns_32_bytes(self):
        k = _derive_key("test")
        assert len(k) == 32


class TestKeystore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            ks_path = Path(td) / "ks.json"
            data = {"testnet": {"wif": "test_wif", "address": "1xyz"}}
            with patch("aria.auto_config.KEYSTORE_FILE", ks_path), \
                 patch("aria.auto_config.ARIA_DIR", Path(td)):
                save_keystore(data, passphrase="testpass")
                loaded = load_keystore(passphrase="testpass")
                assert loaded == data

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as td:
            ks_path = Path(td) / "nonexistent" / "ks.json"
            with patch("aria.auto_config.KEYSTORE_FILE", ks_path):
                result = load_keystore()
                assert result is None


class TestGetOrCreateWif:
    def test_creates_new_key(self):
        with tempfile.TemporaryDirectory() as td:
            ks_path = Path(td) / "ks.json"
            with patch("aria.auto_config.KEYSTORE_FILE", ks_path), \
                 patch("aria.auto_config.ARIA_DIR", Path(td)), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ARIA_BSV_KEY", None)
                os.environ.pop("ARIA_PASSPHRASE", None)
                wif, network, created = get_or_create_wif(
                    network="testnet", passphrase="test123"
                )
                assert isinstance(wif, str)
                assert len(wif) > 40
                assert network == "testnet"
                assert created is True

    def test_reuses_existing_key(self):
        with tempfile.TemporaryDirectory() as td:
            ks_path = Path(td) / "ks.json"
            with patch("aria.auto_config.KEYSTORE_FILE", ks_path), \
                 patch("aria.auto_config.ARIA_DIR", Path(td)), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ARIA_BSV_KEY", None)
                os.environ.pop("ARIA_PASSPHRASE", None)
                wif1, _, created1 = get_or_create_wif(
                    network="testnet", passphrase="test123"
                )
                wif2, _, created2 = get_or_create_wif(
                    network="testnet", passphrase="test123"
                )
                assert wif1 == wif2
                assert created1 is True
                assert created2 is False

    def test_env_var_override(self):
        with patch.dict(os.environ, {"ARIA_BSV_KEY": "custom_wif_value"}):
            wif, network, created = get_or_create_wif()
            assert wif == "custom_wif_value"
            assert created is False

    def test_invalid_network(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARIA_BSV_KEY", None)
            with pytest.raises(Exception):
                get_or_create_wif(network="regtest")


class TestGetAddress:
    def test_returns_address_after_creation(self):
        with tempfile.TemporaryDirectory() as td:
            ks_path = Path(td) / "ks.json"
            with patch("aria.auto_config.KEYSTORE_FILE", ks_path), \
                 patch("aria.auto_config.ARIA_DIR", Path(td)), \
                 patch.dict(os.environ, {"ARIA_PASSPHRASE": "test"}, clear=False):
                os.environ.pop("ARIA_BSV_KEY", None)
                get_or_create_wif(network="testnet", passphrase="test")
                # get_address -> load_keystore reads ARIA_PASSPHRASE env
                addr = get_address(network="testnet")
                assert addr is not None
                assert isinstance(addr, str)
