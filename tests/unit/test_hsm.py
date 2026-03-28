"""Tests for aria.hsm — HSM key management abstraction."""

from __future__ import annotations

import hashlib

import pytest

from aria.hsm import (
    AWSKMSError,
    AWSKMSHSM,
    HSMAlgorithm,
    HSMAlgorithmMismatchError,
    HSMError,
    HSMInterface,
    HSMKeyDisabledError,
    HSMKeyInfo,
    HSMKeyNotFoundError,
    HSMKeySpec,
    HSMKeyState,
    HSMSignResult,
    HSMSigningProxy,
    LocalHSM,
    MockHSM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(algo=HSMAlgorithm.ECDSA_P256, label="test-key"):
    return HSMKeySpec(algorithm=algo, label=label)


# ---------------------------------------------------------------------------
# HSMKeySpec
# ---------------------------------------------------------------------------

class TestHSMKeySpec:
    def test_defaults(self):
        s = HSMKeySpec(algorithm=HSMAlgorithm.HMAC_SHA256)
        assert s.label == ""
        assert s.extractable is False
        assert s.metadata == {}

    def test_custom(self):
        s = HSMKeySpec(
            algorithm=HSMAlgorithm.RSA_2048,
            label="epoch-key",
            extractable=True,
            metadata={"env": "prod"},
        )
        assert s.label == "epoch-key"
        assert s.metadata["env"] == "prod"


# ---------------------------------------------------------------------------
# HSMKeyInfo
# ---------------------------------------------------------------------------

class TestHSMKeyInfo:
    def test_auto_timestamp(self):
        info = HSMKeyInfo(
            key_id="k1", algorithm=HSMAlgorithm.ECDSA_P256,
            label="", state=HSMKeyState.ACTIVE, created_at="",
        )
        assert info.created_at != ""

    def test_to_dict_keys(self):
        info = HSMKeyInfo(
            key_id="k1", algorithm=HSMAlgorithm.ECDSA_P256,
            label="lab", state=HSMKeyState.ACTIVE, created_at="2025-01-01T00:00:00+00:00",
            public_key=b"\x01\x02",
        )
        d = info.to_dict()
        assert d["key_id"] == "k1"
        assert d["algorithm"] == "ecdsa-p256"
        assert d["state"] == "active"
        assert d["public_key"] == "0102"

    def test_to_dict_metadata(self):
        info = HSMKeyInfo(
            key_id="k2", algorithm=HSMAlgorithm.ED25519,
            label="", state=HSMKeyState.ACTIVE, created_at="ts",
            metadata={"team": "red"},
        )
        assert info.to_dict()["metadata"]["team"] == "red"


# ---------------------------------------------------------------------------
# HSMSignResult
# ---------------------------------------------------------------------------

class TestHSMSignResult:
    def test_signature_hex(self):
        r = HSMSignResult(
            key_id="k1", algorithm=HSMAlgorithm.HMAC_SHA256,
            signature=b"\xde\xad\xbe\xef",
        )
        assert r.signature_hex == "deadbeef"

    def test_auto_signed_at(self):
        r = HSMSignResult(key_id="k", algorithm=HSMAlgorithm.HMAC_SHA256, signature=b"")
        assert r.signed_at != ""


# ---------------------------------------------------------------------------
# MockHSM — key lifecycle
# ---------------------------------------------------------------------------

class TestMockHSMLifecycle:
    def test_generate_returns_key_id(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        assert kid.startswith("mock-")

    def test_list_keys_empty(self):
        hsm = MockHSM()
        assert hsm.list_keys() == []

    def test_list_keys_after_generate(self):
        hsm = MockHSM()
        hsm.generate_key(_spec())
        hsm.generate_key(_spec(algo=HSMAlgorithm.HMAC_SHA256))
        assert len(hsm.list_keys()) == 2

    def test_get_key_ok(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        info = hsm.get_key(kid)
        assert info.key_id == kid
        assert info.state == HSMKeyState.ACTIVE

    def test_get_key_not_found(self):
        hsm = MockHSM()
        with pytest.raises(HSMKeyNotFoundError):
            hsm.get_key("nonexistent")

    def test_disable_key(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        hsm.disable_key(kid)
        assert hsm.get_key(kid).state == HSMKeyState.DISABLED

    def test_delete_key(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        hsm.delete_key(kid)
        assert hsm.get_key(kid).state == HSMKeyState.DELETED

    def test_deleted_not_in_list(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        hsm.delete_key(kid)
        assert hsm.list_keys() == []

    def test_public_key_set(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        assert len(hsm.get_key(kid).public_key) > 0

    def test_label_stored(self):
        hsm = MockHSM()
        kid = hsm.generate_key(HSMKeySpec(algorithm=HSMAlgorithm.ED25519, label="epoch-key"))
        assert hsm.get_key(kid).label == "epoch-key"


# ---------------------------------------------------------------------------
# MockHSM — sign / verify
# ---------------------------------------------------------------------------

class TestMockHSMSignVerify:
    def test_sign_returns_result(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        r = hsm.sign(kid, b"hello")
        assert isinstance(r, HSMSignResult)
        assert r.key_id == kid
        assert len(r.signature) > 0

    def test_sign_deterministic(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        s1 = hsm.sign(kid, b"data").signature
        s2 = hsm.sign(kid, b"data").signature
        assert s1 == s2

    def test_sign_different_data_different_sig(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        s1 = hsm.sign(kid, b"data1").signature
        s2 = hsm.sign(kid, b"data2").signature
        assert s1 != s2

    def test_verify_correct(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        sig = hsm.sign(kid, b"payload").signature
        assert hsm.verify(kid, b"payload", sig) is True

    def test_verify_wrong_data(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        sig = hsm.sign(kid, b"correct").signature
        assert hsm.verify(kid, b"wrong", sig) is False

    def test_verify_tampered_sig(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        sig = bytearray(hsm.sign(kid, b"data").signature)
        sig[0] ^= 0xFF
        assert hsm.verify(kid, b"data", bytes(sig)) is False

    def test_sign_disabled_key_raises(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        hsm.disable_key(kid)
        with pytest.raises(HSMKeyDisabledError):
            hsm.sign(kid, b"data")

    def test_sign_deleted_key_raises(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        hsm.delete_key(kid)
        with pytest.raises(HSMKeyDisabledError):
            hsm.sign(kid, b"data")

    def test_verify_deleted_returns_false(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        sig = hsm.sign(kid, b"data").signature
        hsm.delete_key(kid)
        assert hsm.verify(kid, b"data", sig) is False

    def test_sign_not_found_raises(self):
        hsm = MockHSM()
        with pytest.raises(HSMKeyNotFoundError):
            hsm.sign("ghost", b"data")

    def test_algorithm_stored_in_result(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec(algo=HSMAlgorithm.HMAC_SHA256))
        r = hsm.sign(kid, b"x")
        assert r.algorithm == HSMAlgorithm.HMAC_SHA256


# ---------------------------------------------------------------------------
# LocalHSM — key lifecycle
# ---------------------------------------------------------------------------

class TestLocalHSMLifecycle:
    def test_generate_unique_ids(self):
        hsm = LocalHSM()
        ids = {hsm.generate_key(_spec()) for _ in range(5)}
        assert len(ids) == 5

    def test_list_after_generate(self):
        hsm = LocalHSM()
        hsm.generate_key(_spec())
        hsm.generate_key(_spec())
        assert len(hsm.list_keys()) == 2

    def test_get_key_ok(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec(label="lab"))
        info = hsm.get_key(kid)
        assert info.label == "lab"

    def test_disable_key(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        hsm.disable_key(kid)
        assert hsm.get_key(kid).state == HSMKeyState.DISABLED

    def test_delete_key_removes_material(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        hsm.delete_key(kid)
        assert kid not in hsm._private_material

    def test_deleted_key_not_in_list(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        hsm.delete_key(kid)
        assert all(k.key_id != kid for k in hsm.list_keys())

    def test_public_key_derived(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        info = hsm.get_key(kid)
        assert len(info.public_key) == 32

    def test_metadata_in_spec_stored(self):
        hsm = LocalHSM()
        spec = HSMKeySpec(
            algorithm=HSMAlgorithm.ECDSA_P256,
            label="m",
            metadata={"owner": "alice"},
        )
        kid = hsm.generate_key(spec)
        assert hsm.get_key(kid).metadata["owner"] == "alice"


# ---------------------------------------------------------------------------
# LocalHSM — sign / verify
# ---------------------------------------------------------------------------

class TestLocalHSMSignVerify:
    def test_sign_returns_bytes(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        r = hsm.sign(kid, b"data")
        assert isinstance(r.signature, bytes)
        assert len(r.signature) > 0

    def test_verify_correct(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        sig = hsm.sign(kid, b"test-payload").signature
        assert hsm.verify(kid, b"test-payload", sig) is True

    def test_verify_wrong_data(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        sig = hsm.sign(kid, b"correct").signature
        assert hsm.verify(kid, b"wrong", sig) is False

    def test_verify_tampered(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        sig = bytearray(hsm.sign(kid, b"data").signature)
        sig[0] ^= 0x01
        assert hsm.verify(kid, b"data", bytes(sig)) is False

    def test_sign_disabled_raises(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        hsm.disable_key(kid)
        with pytest.raises(HSMKeyDisabledError):
            hsm.sign(kid, b"x")

    def test_sign_deleted_raises(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        hsm.delete_key(kid)
        with pytest.raises(HSMKeyDisabledError):
            hsm.sign(kid, b"x")

    def test_sign_different_keys_different_sigs(self):
        hsm = LocalHSM()
        k1 = hsm.generate_key(_spec())
        k2 = hsm.generate_key(_spec())
        s1 = hsm.sign(k1, b"same").signature
        s2 = hsm.sign(k2, b"same").signature
        assert s1 != s2

    def test_verify_deleted_no_material(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        sig = hsm.sign(kid, b"data").signature
        hsm.delete_key(kid)
        # After delete, key is in DELETED state → verify raises HSMKeyDisabledError
        with pytest.raises(HSMKeyDisabledError):
            hsm.verify(kid, b"data", sig)

    def test_sign_not_found_raises(self):
        hsm = LocalHSM()
        with pytest.raises(HSMKeyNotFoundError):
            hsm.sign("ghost", b"x")


# ---------------------------------------------------------------------------
# HSMSigningProxy
# ---------------------------------------------------------------------------

class TestHSMSigningProxy:
    def test_sign_bytes(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        proxy = HSMSigningProxy(hsm=hsm, key_id=kid)
        sig = proxy.sign_bytes(b"commitment-hash")
        assert isinstance(sig, bytes)
        assert len(sig) > 0

    def test_verify_bytes_correct(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        proxy = HSMSigningProxy(hsm=hsm, key_id=kid)
        sig = proxy.sign_bytes(b"epoch-data")
        assert proxy.verify_bytes(b"epoch-data", sig) is True

    def test_verify_bytes_wrong(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        proxy = HSMSigningProxy(hsm=hsm, key_id=kid)
        sig = proxy.sign_bytes(b"correct")
        assert proxy.verify_bytes(b"wrong", sig) is False

    def test_public_key_hex(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        proxy = HSMSigningProxy(hsm=hsm, key_id=kid)
        pub = proxy.public_key_hex()
        assert len(pub) > 0
        int(pub, 16)  # should be valid hex

    def test_key_id_property(self):
        hsm = MockHSM()
        kid = hsm.generate_key(_spec())
        proxy = HSMSigningProxy(hsm=hsm, key_id=kid)
        assert proxy.key_id == kid

    def test_roundtrip_with_local_hsm(self):
        hsm = LocalHSM()
        kid = hsm.generate_key(_spec())
        proxy = HSMSigningProxy(hsm=hsm, key_id=kid)
        data = b"brc-121-epoch-commitment"
        sig = proxy.sign_bytes(data)
        assert proxy.verify_bytes(data, sig) is True
        assert proxy.verify_bytes(b"other", sig) is False


# ---------------------------------------------------------------------------
# HSMInterface protocol compliance
# ---------------------------------------------------------------------------

class TestHSMInterfaceCompliance:
    def test_mock_is_hsm_interface(self):
        hsm = MockHSM()
        assert isinstance(hsm, HSMInterface)

    def test_local_is_hsm_interface(self):
        hsm = LocalHSM()
        assert isinstance(hsm, HSMInterface)


# ---------------------------------------------------------------------------
# AWSKMSHSM — missing boto3
# ---------------------------------------------------------------------------

class TestAWSKMSHSMNoBoto3:
    def test_raises_if_no_boto3(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("boto3 not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(AWSKMSError, match="boto3"):
            AWSKMSHSM()


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class TestErrorHierarchy:
    def test_key_not_found_is_hsm_error(self):
        assert issubclass(HSMKeyNotFoundError, HSMError)

    def test_key_disabled_is_hsm_error(self):
        assert issubclass(HSMKeyDisabledError, HSMError)

    def test_algorithm_mismatch_is_hsm_error(self):
        assert issubclass(HSMAlgorithmMismatchError, HSMError)

    def test_aws_kms_error_is_hsm_error(self):
        assert issubclass(AWSKMSError, HSMError)

    def test_hsm_error_is_exception(self):
        assert issubclass(HSMError, Exception)


# ---------------------------------------------------------------------------
# Multiple key types
# ---------------------------------------------------------------------------

class TestMultipleKeyTypes:
    @pytest.mark.parametrize("algo", list(HSMAlgorithm))
    def test_generate_all_algorithms(self, algo):
        hsm = MockHSM()
        kid = hsm.generate_key(HSMKeySpec(algorithm=algo))
        assert hsm.get_key(kid).algorithm == algo

    @pytest.mark.parametrize("algo", list(HSMAlgorithm))
    def test_sign_all_algorithms(self, algo):
        hsm = MockHSM()
        kid = hsm.generate_key(HSMKeySpec(algorithm=algo))
        r = hsm.sign(kid, b"test")
        assert r.algorithm == algo


# ---------------------------------------------------------------------------
# AWSKMSHSM — with mocked boto3
# ---------------------------------------------------------------------------

class TestAWSKMSHSMWithMockBoto3:
    """Tests AWSKMSHSM with a fully mocked boto3.client."""

    def _make_mock_client(self, mocker_or_monkeypatch=None):
        """Return a mock KMS client and the AWSKMSHSM instance."""
        import sys, types
        from unittest.mock import MagicMock, patch
        from datetime import datetime, timezone

        mock_key_id = "arn:aws:kms:us-east-1:123456789:key/test-key-id"
        mock_pub = b"\x04" + b"\xab" * 64  # fake uncompressed pubkey bytes

        mock_client = MagicMock()
        mock_client.create_key.return_value = {
            "KeyMetadata": {"KeyId": mock_key_id, "KeyState": "Enabled"}
        }
        mock_client.get_public_key.return_value = {"PublicKey": mock_pub}
        mock_client.create_alias.return_value = {}
        mock_client.sign.return_value = {"Signature": b"\xde\xad\xbe\xef" * 16}
        mock_client.verify.return_value = {"SignatureValid": True}
        mock_client.disable_key.return_value = {}
        mock_client.schedule_key_deletion.return_value = {}
        mock_client.describe_key.return_value = {
            "KeyMetadata": {
                "KeyId": mock_key_id,
                "KeyState": "Enabled",
                "CustomerMasterKeySpec": "ECC_SECG_P256K1",
                "Description": "test-key",
                "CreationDate": datetime.now(timezone.utc),
            }
        }

        # Paginator for list_aliases
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Aliases": [{"AliasName": "alias/aria/test-key-id", "TargetKeyId": mock_key_id}]}
        ]
        mock_client.get_paginator.return_value = mock_paginator

        # Patch boto3
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            import importlib
            import aria.hsm as hsm_mod
            # Re-import to pick up mock
            original_boto3 = sys.modules.get("boto3")
            sys.modules["boto3"] = mock_boto3
            hsm = AWSKMSHSM(region="us-east-1", key_alias_prefix="aria/")
            hsm._client = mock_client
            if original_boto3:
                sys.modules["boto3"] = original_boto3
            else:
                del sys.modules["boto3"]

        return hsm, mock_client, mock_key_id

    def test_generate_key_returns_key_id(self):
        hsm, mock_client, expected_key_id = self._make_mock_client()
        key_id = hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1, label="test"))
        assert key_id == expected_key_id
        mock_client.create_key.assert_called_once()

    def test_generate_key_stores_in_cache(self):
        hsm, mock_client, expected_key_id = self._make_mock_client()
        key_id = hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        info = hsm.get_key(key_id)
        assert info.key_id == expected_key_id
        assert info.state == HSMKeyState.ACTIVE

    def test_sign_returns_result(self):
        hsm, mock_client, key_id = self._make_mock_client()
        hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        result = hsm.sign(key_id, b"test-data")
        assert isinstance(result.signature, bytes)
        assert len(result.signature) > 0

    def test_verify_returns_true_on_valid(self):
        hsm, mock_client, key_id = self._make_mock_client()
        hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        valid = hsm.verify(key_id, b"test-data", b"\xde\xad" * 8)
        assert valid is True

    def test_verify_returns_false_on_exception(self):
        hsm, mock_client, key_id = self._make_mock_client()
        hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        mock_client.verify.side_effect = Exception("KMS error")
        valid = hsm.verify(key_id, b"test-data", b"\x00" * 8)
        assert valid is False

    def test_disable_key_changes_state(self):
        hsm, mock_client, key_id = self._make_mock_client()
        hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        hsm.disable_key(key_id)
        assert hsm.get_key(key_id).state == HSMKeyState.DISABLED

    def test_delete_key_changes_state(self):
        hsm, mock_client, key_id = self._make_mock_client()
        hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        hsm.delete_key(key_id)
        assert hsm.get_key(key_id).state == HSMKeyState.DELETED

    def test_list_keys_uses_paginator(self):
        hsm, mock_client, key_id = self._make_mock_client()
        hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        keys = hsm.list_keys()
        assert len(keys) >= 1
        assert any(k.key_id == key_id for k in keys)
        mock_client.get_paginator.assert_called_with("list_aliases")

    def test_list_keys_falls_back_to_cache_on_error(self):
        hsm, mock_client, key_id = self._make_mock_client()
        hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        mock_client.get_paginator.side_effect = Exception("AWS error")
        keys = hsm.list_keys()
        assert len(keys) == 1
        assert keys[0].key_id == key_id

    def test_get_key_fetches_from_kms_if_not_cached(self):
        hsm, mock_client, key_id = self._make_mock_client()
        # Don't generate first — get_key should fetch from KMS
        info = hsm.get_key(key_id)
        assert info.key_id == key_id
        mock_client.describe_key.assert_called_with(KeyId=key_id)
