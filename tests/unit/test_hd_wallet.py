"""Tests for aria.wallet.hd — HDWallet (BIP32/BIP44) and WatchOnlyHDWallet."""

from __future__ import annotations

import pytest

from aria.core.errors import ARIAWalletError
from aria.wallet.hd import (
    HDWallet,
    WatchOnlyHDWallet,
    _BIP32Node,
    _base58check_decode,
    _base58check_encode,
    _hash160,
    _hmac_sha512,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeBroadcaster:
    async def broadcast(self, raw):
        class R:
            propagated = True
            txid = "fakefakefakefakefakefakefakefakefakefakefakefakefakefakefakefake00"
            message = ""
        return R()


_SEED_16 = bytes(range(16))          # 16-byte seed
_SEED_64 = bytes(range(64))          # 64-byte seed
_SEED_ALT = bytes(range(1, 65))      # different 64-byte seed


def _wallet(seed: bytes = _SEED_64) -> HDWallet:
    return HDWallet.from_seed(seed, _FakeBroadcaster())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestHmacSha512:
    def test_returns_64_bytes(self):
        result = _hmac_sha512(b"key", b"data")
        assert len(result) == 64

    def test_deterministic(self):
        assert _hmac_sha512(b"k", b"d") == _hmac_sha512(b"k", b"d")

    def test_different_key_different_result(self):
        assert _hmac_sha512(b"k1", b"d") != _hmac_sha512(b"k2", b"d")


class TestBase58Check:
    def test_encode_decode_roundtrip(self):
        payload = b"\x00" + bytes(20)
        encoded = _base58check_encode(payload)
        decoded = _base58check_decode(encoded)
        assert decoded == payload

    def test_invalid_checksum_raises(self):
        payload = b"\x00" + bytes(20)
        encoded = _base58check_encode(payload)
        # Corrupt the last character
        corrupted = encoded[:-1] + ("z" if encoded[-1] != "z" else "a")
        with pytest.raises(ARIAWalletError):
            _base58check_decode(corrupted)


# ---------------------------------------------------------------------------
# _BIP32Node — key derivation
# ---------------------------------------------------------------------------

class TestBIP32Node:
    def _master_node(self) -> _BIP32Node:
        """Create a deterministic master node from seed."""
        I = _hmac_sha512(b"Bitcoin seed", _SEED_64)
        return _BIP32Node(
            private_key_bytes=I[:32],
            chain_code=I[32:],
            depth=0,
            index=0,
        )

    def test_derive_child_returns_new_node(self):
        node = self._master_node()
        child = node.derive_child(0)
        assert child is not node

    def test_derive_child_increments_depth(self):
        node = self._master_node()
        child = node.derive_child(0)
        assert child.depth == 1

    def test_derive_child_stores_index(self):
        node = self._master_node()
        child = node.derive_child(42)
        assert child.index == 42

    def test_hardened_child_different_from_normal(self):
        node = self._master_node()
        normal = node.derive_child(0)
        hardened = node.derive_child(0x80000000)
        assert normal.private_key_bytes != hardened.private_key_bytes

    def test_different_indices_different_keys(self):
        node = self._master_node()
        c0 = node.derive_child(0)
        c1 = node.derive_child(1)
        assert c0.private_key_bytes != c1.private_key_bytes

    def test_wif_returns_valid_string(self):
        node = self._master_node()
        wif = node.wif()
        assert len(wif) > 50
        assert wif[0] in ("L", "K", "5", "c", "9")  # WIF prefixes

    def test_address_returns_p2pkh(self):
        node = self._master_node()
        addr = node.address()
        assert addr[0] in ("1", "m", "n")

    def test_xprv_starts_with_xprv(self):
        node = self._master_node()
        xprv = node.xprv()
        assert xprv.startswith("xprv")

    def test_xpub_starts_with_xpub(self):
        node = self._master_node()
        xpub = node.xpub()
        assert xpub.startswith("xpub")

    def test_xprv_length_standard(self):
        # BIP32 xprv strings are always 111 characters
        node = self._master_node()
        assert len(node.xprv()) == 111

    def test_xpub_length_standard(self):
        node = self._master_node()
        assert len(node.xpub()) == 111


# ---------------------------------------------------------------------------
# HDWallet — construction
# ---------------------------------------------------------------------------

class TestHDWalletConstruction:
    def test_from_seed_creates_wallet(self):
        wallet = _wallet()
        assert wallet is not None

    def test_short_seed_raises(self):
        with pytest.raises(ARIAWalletError):
            HDWallet.from_seed(b"\x00" * 8, _FakeBroadcaster())

    def test_16_byte_seed_accepted(self):
        wallet = HDWallet.from_seed(_SEED_16, _FakeBroadcaster())
        assert wallet is not None

    def test_master_at_base_path_depth(self):
        wallet = _wallet()
        # BASE_PATH = m/44'/236'/0' → depth 3
        assert wallet._master.depth == 3

    def test_from_xprv_roundtrip_same_keys(self):
        wallet = _wallet()
        xprv = wallet.xprv()
        wallet2 = HDWallet.from_xprv(xprv, _FakeBroadcaster())
        assert wallet.derive_system_key(0).wif() == wallet2.derive_system_key(0).wif()

    def test_from_xprv_invalid_raises(self):
        with pytest.raises(ARIAWalletError):
            HDWallet.from_xprv("notanxprvstring", _FakeBroadcaster())

    def test_different_seeds_different_keys(self):
        w1 = _wallet(_SEED_64)
        w2 = _wallet(_SEED_ALT)
        assert w1.derive_system_key(0).wif() != w2.derive_system_key(0).wif()


# ---------------------------------------------------------------------------
# HDWallet — key derivation
# ---------------------------------------------------------------------------

class TestHDWalletDerivation:
    def test_derive_system_key_returns_bip32_node(self):
        wallet = _wallet()
        node = wallet.derive_system_key(0)
        assert isinstance(node, _BIP32Node)

    def test_different_systems_different_keys(self):
        wallet = _wallet()
        assert wallet.derive_system_key(0).wif() != wallet.derive_system_key(1).wif()
        assert wallet.derive_system_key(0).wif() != wallet.derive_system_key(2).wif()

    def test_same_system_same_key_deterministic(self):
        wallet = _wallet()
        k1 = wallet.derive_system_key(0).wif()
        k2 = wallet.derive_system_key(0).wif()
        assert k1 == k2

    def test_current_wif_matches_derive_system_key(self):
        wallet = _wallet()
        assert wallet.current_wif(0) == wallet.derive_system_key(0).wif()

    def test_explicit_key_index_0_same_as_default(self):
        wallet = _wallet()
        k_default = wallet.derive_system_key(0).wif()
        k_explicit = wallet.derive_system_key(0, key_index=0).wif()
        assert k_default == k_explicit

    def test_different_key_indices_different_keys(self):
        wallet = _wallet()
        k0 = wallet.derive_system_key(0, key_index=0).wif()
        k1 = wallet.derive_system_key(0, key_index=1).wif()
        assert k0 != k1


# ---------------------------------------------------------------------------
# HDWallet — key rotation
# ---------------------------------------------------------------------------

class TestHDWalletRotation:
    def test_key_index_starts_at_zero(self):
        wallet = _wallet()
        assert wallet.key_index(0) == 0

    def test_rotate_returns_new_wif(self):
        wallet = _wallet()
        original = wallet.current_wif(0)
        rotated = wallet.rotate(0)
        assert rotated != original

    def test_rotate_increments_key_index(self):
        wallet = _wallet()
        wallet.rotate(0)
        assert wallet.key_index(0) == 1

    def test_rotate_twice_increments_to_2(self):
        wallet = _wallet()
        wallet.rotate(0)
        wallet.rotate(0)
        assert wallet.key_index(0) == 2

    def test_rotate_does_not_affect_other_systems(self):
        wallet = _wallet()
        key_sys1_before = wallet.current_wif(1)
        wallet.rotate(0)
        assert wallet.current_wif(1) == key_sys1_before

    def test_old_key_still_derivable_after_rotation(self):
        wallet = _wallet()
        old_key = wallet.derive_system_key(0, key_index=0).wif()
        wallet.rotate(0)
        assert wallet.derive_system_key(0, key_index=0).wif() == old_key

    def test_rotated_key_matches_explicit_key_index(self):
        wallet = _wallet()
        rotated = wallet.rotate(0)
        explicit = wallet.derive_system_key(0, key_index=1).wif()
        assert rotated == explicit


# ---------------------------------------------------------------------------
# HDWallet — serialization
# ---------------------------------------------------------------------------

class TestHDWalletSerialization:
    def test_xprv_starts_with_xprv(self):
        wallet = _wallet()
        assert wallet.xprv().startswith("xprv")

    def test_xpub_starts_with_xpub(self):
        wallet = _wallet()
        assert wallet.xpub().startswith("xpub")

    def test_xprv_deterministic(self):
        w1 = _wallet(_SEED_64)
        w2 = _wallet(_SEED_64)
        assert w1.xprv() == w2.xprv()

    def test_xpub_deterministic(self):
        w1 = _wallet(_SEED_64)
        w2 = _wallet(_SEED_64)
        assert w1.xpub() == w2.xpub()


# ---------------------------------------------------------------------------
# WatchOnlyHDWallet
# ---------------------------------------------------------------------------

class TestWatchOnlyHDWallet:
    def test_from_xpub_creates_instance(self):
        wallet = _wallet()
        watch = WatchOnlyHDWallet.from_xpub(wallet.xpub())
        assert watch is not None

    def test_invalid_xpub_raises(self):
        with pytest.raises(ARIAWalletError):
            WatchOnlyHDWallet.from_xpub("notanxpub")

    def test_derive_system_address_matches_hd_wallet(self):
        wallet = _wallet()
        watch = WatchOnlyHDWallet.from_xpub(wallet.xpub())
        addr_watch = watch.derive_system_address(0)
        addr_hd = wallet.derive_system_key(0, key_index=0).address()
        assert addr_watch == addr_hd

    def test_different_systems_different_addresses(self):
        wallet = _wallet()
        watch = WatchOnlyHDWallet.from_xpub(wallet.xpub())
        addr0 = watch.derive_system_address(0)
        addr1 = watch.derive_system_address(1)
        assert addr0 != addr1

    def test_address_starts_with_1_mainnet(self):
        wallet = _wallet()
        watch = WatchOnlyHDWallet.from_xpub(wallet.xpub())
        addr = watch.derive_system_address(0)
        assert addr.startswith("1")

    def test_key_index_1_different_address(self):
        wallet = _wallet()
        watch = WatchOnlyHDWallet.from_xpub(wallet.xpub())
        addr0 = watch.derive_system_address(0, key_index=0)
        addr1 = watch.derive_system_address(0, key_index=1)
        assert addr0 != addr1

    def test_key_index_1_matches_rotated_wallet(self):
        wallet = _wallet()
        wallet.rotate(0)  # moves to key_index=1
        watch = WatchOnlyHDWallet.from_xpub(wallet.xpub())
        addr_watch = watch.derive_system_address(0, key_index=1)
        addr_hd = wallet.derive_system_key(0, key_index=1).address()
        assert addr_watch == addr_hd
