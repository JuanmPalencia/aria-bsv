"""Tests for aria.wallet.keygen — BSV keypair generation."""

from __future__ import annotations

import os
import tempfile
import pytest

from aria.wallet.keygen import (
    KeyPair,
    generate_keypair,
    write_env_file,
    _base58_encode,
    _pubkey_to_address,
)


class TestGenerateKeypair:
    """Tests for the generate_keypair function."""

    def test_generates_testnet_key_by_default(self):
        kp = generate_keypair()
        assert kp.network == "testnet"

    def test_generates_mainnet_key(self):
        kp = generate_keypair(network="mainnet")
        assert kp.network == "mainnet"

    def test_rejects_invalid_network(self):
        with pytest.raises(Exception, match="network"):
            generate_keypair(network="regtest")

    def test_wif_is_non_empty_string(self):
        kp = generate_keypair()
        assert isinstance(kp.wif, str)
        assert len(kp.wif) > 40

    def test_address_is_non_empty_string(self):
        kp = generate_keypair()
        assert isinstance(kp.address, str)
        assert len(kp.address) > 20

    def test_public_key_hex_is_valid(self):
        kp = generate_keypair()
        assert isinstance(kp.public_key_hex, str)
        assert len(kp.public_key_hex) >= 64
        # Should be hex
        int(kp.public_key_hex, 16)

    def test_created_at_is_set(self):
        kp = generate_keypair()
        assert "UTC" in kp.created_at

    def test_two_keys_are_different(self):
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        assert kp1.wif != kp2.wif
        assert kp1.address != kp2.address

    def test_testnet_wif_not_starts_with_K_or_L(self):
        """Testnet compressed WIFs typically start with 'c'."""
        kp = generate_keypair(network="testnet")
        # Testnet WIF prefix: 'c' (compressed) or '9' (uncompressed)
        assert kp.wif[0] in ("c", "9")

    def test_mainnet_wif_starts_with_K_or_L(self):
        """Mainnet compressed WIFs start with 'K' or 'L'."""
        kp = generate_keypair(network="mainnet")
        assert kp.wif[0] in ("K", "L")


class TestKeyPair:
    """Tests for the KeyPair dataclass."""

    def _make(self, **kwargs) -> KeyPair:
        defaults = {
            "wif": "cTestWIF123456",
            "address": "mTestAddress123",
            "public_key_hex": "02" + "aa" * 32,
            "network": "testnet",
            "created_at": "2026-03-29 12:00:00 UTC",
        }
        defaults.update(kwargs)
        return KeyPair(**defaults)

    def test_str_contains_warning(self):
        kp = self._make()
        text = str(kp)
        assert "SAVE THIS KEY" in text
        assert "NOT be shown again" in text
        assert "does NOT store" in text

    def test_str_contains_wif(self):
        kp = self._make(wif="cMySecretKey123")
        text = str(kp)
        assert "cMySecretKey123" in text

    def test_to_dict_has_all_fields(self):
        kp = self._make()
        d = kp.to_dict()
        assert "wif" in d
        assert "address" in d
        assert "public_key_hex" in d
        assert "network" in d
        assert "created_at" in d

    def test_to_env_line_format(self):
        kp = self._make(wif="cMyKey123")
        assert kp.to_env_line() == "ARIA_BSV_KEY=cMyKey123"

    def test_immutable(self):
        kp = self._make()
        with pytest.raises(AttributeError):
            kp.wif = "something_else"  # type: ignore[misc]


class TestWriteEnvFile:
    """Tests for the write_env_file function."""

    def _make_kp(self) -> KeyPair:
        return KeyPair(
            wif="cTestWIF_write_env",
            address="mAddr",
            public_key_hex="02" + "bb" * 32,
            network="testnet",
            created_at="2026-03-29 12:00:00 UTC",
        )

    def test_creates_new_env_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".env")
            write_env_file(self._make_kp(), path)
            assert os.path.exists(path)
            content = open(path).read()
            assert "ARIA_BSV_KEY=cTestWIF_write_env" in content

    def test_appends_to_existing_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".env")
            with open(path, "w") as f:
                f.write("OTHER_VAR=hello\n")
            write_env_file(self._make_kp(), path)
            content = open(path).read()
            assert "OTHER_VAR=hello" in content
            assert "ARIA_BSV_KEY=cTestWIF_write_env" in content

    def test_contains_comment_with_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".env")
            write_env_file(self._make_kp(), path)
            content = open(path).read()
            assert "# ARIA BSV key" in content
            assert "2026-03-29" in content

    def test_never_writes_without_explicit_call(self):
        """Generating a key does NOT auto-write anything to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".env")
            generate_keypair()
            assert not os.path.exists(path)


class TestBase58Encode:
    """Tests for the Base58 encoder used in fallback key generation."""

    def test_encodes_empty_bytes(self):
        assert _base58_encode(b"") == ""

    def test_encodes_single_byte(self):
        result = _base58_encode(b"\x01")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_leading_zeros_preserved(self):
        result = _base58_encode(b"\x00\x00\x01")
        assert result.startswith("11")

    def test_encodes_known_value(self):
        # b"\x00" should encode to "1" in Base58 Bitcoin
        result = _base58_encode(b"\x00")
        assert result == "1"


class TestPubkeyToAddress:
    """Tests for the address derivation function."""

    def test_mainnet_address_starts_with_1(self):
        # 33-byte compressed pubkey
        fake_pubkey = bytes.fromhex("02" + "aa" * 32)
        addr = _pubkey_to_address(fake_pubkey, "mainnet")
        assert addr[0] == "1"

    def test_testnet_address_starts_with_m_or_n(self):
        fake_pubkey = bytes.fromhex("02" + "bb" * 32)
        addr = _pubkey_to_address(fake_pubkey, "testnet")
        assert addr[0] in ("m", "n")

    def test_deterministic(self):
        pubkey = bytes.fromhex("03" + "cc" * 32)
        a1 = _pubkey_to_address(pubkey, "mainnet")
        a2 = _pubkey_to_address(pubkey, "mainnet")
        assert a1 == a2
