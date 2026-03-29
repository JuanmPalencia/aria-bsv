"""Tests for aria.integrations.vault — VaultKeyManager with mocked httpx calls.

All tests use unittest.mock to intercept httpx calls; no real Vault instance
is required. The test suite validates:
  - AppRole login flow
  - Token-based authentication
  - get_key success and error paths
  - rotate_key success and error paths
  - Namespace header propagation
  - Async get_key_async
  - Security invariant: WIF never appears in error messages
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from aria.integrations.vault import ARIAVaultError, VaultKeyManager


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_SAMPLE_WIF = "L1aW4aubDFB7yfras2S1mN3bqg9nwySY8nkoLmJebSLD5BWv3ENZ"
_VAULT_ADDR = "https://vault.example.com"


def _mock_resp(status_code: int, json_data: dict | None = None) -> MagicMock:
    """Return a mock httpx.Response with the given status code and JSON body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


def _kv_resp(wif: str = _SAMPLE_WIF, field: str = "wif") -> dict:
    """Return a standard Vault KV v2 GET response containing a WIF."""
    return {"data": {"data": {field: wif}}}


def _approle_resp(token: str = "hvs.TESTTOKEN") -> dict:
    """Return a standard Vault AppRole login response."""
    return {"auth": {"client_token": token}}


# ---------------------------------------------------------------------------
# AppRole login tests
# ---------------------------------------------------------------------------


class TestAppRoleLogin:
    """VaultKeyManager._login_approle() behaviour."""

    def test_login_success_returns_client_token(self) -> None:
        km = VaultKeyManager(
            vault_addr=_VAULT_ADDR,
            role_id="my-role",
            secret_id="my-secret",
        )
        with patch("httpx.post", return_value=_mock_resp(200, _approle_resp("hvs.ABC"))):
            token = km._login_approle()
        assert token == "hvs.ABC"

    def test_login_sends_role_id_and_secret_id(self) -> None:
        km = VaultKeyManager(
            vault_addr=_VAULT_ADDR,
            role_id="role-123",
            secret_id="sec-456",
        )
        with patch("httpx.post", return_value=_mock_resp(200, _approle_resp())) as mock_post:
            km._login_approle()
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert body["role_id"] == "role-123"
        assert body["secret_id"] == "sec-456"

    def test_login_posts_to_correct_approle_url(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, role_id="r", secret_id="s")
        with patch("httpx.post", return_value=_mock_resp(200, _approle_resp())) as mock_post:
            km._login_approle()
        url_called = mock_post.call_args[0][0]
        assert url_called == f"{_VAULT_ADDR}/v1/auth/approle/login"

    def test_login_non_200_raises_vault_error(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, role_id="r", secret_id="s")
        with patch("httpx.post", return_value=_mock_resp(403)):
            with pytest.raises(ARIAVaultError) as exc_info:
                km._login_approle()
        assert "403" in str(exc_info.value)

    def test_login_connection_error_raises_vault_error(self) -> None:
        import httpx as _httpx

        km = VaultKeyManager(vault_addr=_VAULT_ADDR, role_id="r", secret_id="s")
        with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
            with pytest.raises(ARIAVaultError) as exc_info:
                km._login_approle()
        assert "connection error" in str(exc_info.value).lower()

    def test_login_malformed_response_raises_vault_error(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, role_id="r", secret_id="s")
        # Response is 200 but does not contain auth.client_token
        with patch("httpx.post", return_value=_mock_resp(200, {"auth": {}})):
            with pytest.raises(ARIAVaultError):
                km._login_approle()


# ---------------------------------------------------------------------------
# _make_headers tests
# ---------------------------------------------------------------------------


class TestMakeHeaders:
    """VaultKeyManager._make_headers() behaviour."""

    def test_token_auth_returns_token_header(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="hvs.MYTOKEN")
        headers = km._make_headers()
        assert headers["X-Vault-Token"] == "hvs.MYTOKEN"

    def test_namespace_header_present_when_configured(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok", namespace="prod-ns")
        headers = km._make_headers()
        assert headers.get("X-Vault-Namespace") == "prod-ns"

    def test_namespace_header_absent_when_not_configured(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        headers = km._make_headers()
        assert "X-Vault-Namespace" not in headers

    def test_approle_auto_login_when_no_token(self) -> None:
        km = VaultKeyManager(
            vault_addr=_VAULT_ADDR,
            role_id="role",
            secret_id="secret",
        )
        with patch("httpx.post", return_value=_mock_resp(200, _approle_resp("hvs.NEW"))):
            headers = km._make_headers()
        assert headers["X-Vault-Token"] == "hvs.NEW"

    def test_no_credentials_raises_vault_error(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR)
        with pytest.raises(ARIAVaultError):
            km._make_headers()


# ---------------------------------------------------------------------------
# get_key tests
# ---------------------------------------------------------------------------


class TestGetKey:
    """VaultKeyManager.get_key() behaviour."""

    def test_get_key_success_returns_wif(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        with patch("httpx.get", return_value=_mock_resp(200, _kv_resp(_SAMPLE_WIF))):
            result = km.get_key("aria/bsv/signing-key")
        assert result == _SAMPLE_WIF

    def test_get_key_uses_correct_kv_url(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok", mount_path="kv")
        with patch("httpx.get", return_value=_mock_resp(200, _kv_resp())) as mock_get:
            km.get_key("my/path")
        url_called = mock_get.call_args[0][0]
        assert url_called == f"{_VAULT_ADDR}/v1/kv/data/my/path"

    def test_get_key_custom_field(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        resp_body = {"data": {"data": {"signing_key": "MYWIF"}}}
        with patch("httpx.get", return_value=_mock_resp(200, resp_body)):
            result = km.get_key("path", field="signing_key")
        assert result == "MYWIF"

    def test_get_key_404_raises_vault_error(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        with patch("httpx.get", return_value=_mock_resp(404)):
            with pytest.raises(ARIAVaultError) as exc_info:
                km.get_key("nonexistent/path")
        assert "404" in str(exc_info.value)

    def test_get_key_403_raises_vault_error(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="bad-token")
        with patch("httpx.get", return_value=_mock_resp(403)):
            with pytest.raises(ARIAVaultError) as exc_info:
                km.get_key("aria/bsv/key")
        assert "403" in str(exc_info.value)

    def test_get_key_missing_field_raises_vault_error(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        body = {"data": {"data": {"other_field": "value"}}}
        with patch("httpx.get", return_value=_mock_resp(200, body)):
            with pytest.raises(ARIAVaultError) as exc_info:
                km.get_key("path", field="wif")
        assert "wif" in str(exc_info.value)

    def test_get_key_wif_never_in_error_message(self) -> None:
        """WIF string must never appear in any ARIAVaultError message."""
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        with patch("httpx.get", return_value=_mock_resp(500)):
            with pytest.raises(ARIAVaultError) as exc_info:
                km.get_key("path")
        assert _SAMPLE_WIF not in str(exc_info.value)

    def test_get_key_namespace_sent_in_request(self) -> None:
        km = VaultKeyManager(
            vault_addr=_VAULT_ADDR, token="tok", namespace="my-ns"
        )
        with patch("httpx.get", return_value=_mock_resp(200, _kv_resp())) as mock_get:
            km.get_key("path")
        _, kwargs = mock_get.call_args
        assert kwargs.get("headers", {}).get("X-Vault-Namespace") == "my-ns"


# ---------------------------------------------------------------------------
# rotate_key tests
# ---------------------------------------------------------------------------


class TestRotateKey:
    """VaultKeyManager.rotate_key() behaviour."""

    def test_rotate_key_success_posts_new_key(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        with patch("httpx.post", return_value=_mock_resp(200, {"data": {"version": 2}})) as mock_post:
            km.rotate_key("aria/bsv/signing-key", new_wif="NEWWIF")
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert body["data"]["wif"] == "NEWWIF"

    def test_rotate_key_uses_correct_kv_url(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok", mount_path="secret")
        with patch("httpx.post", return_value=_mock_resp(200, {})) as mock_post:
            km.rotate_key("my/key/path", new_wif="W")
        url_called = mock_post.call_args[0][0]
        assert url_called == f"{_VAULT_ADDR}/v1/secret/data/my/key/path"

    def test_rotate_key_custom_field(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        with patch("httpx.post", return_value=_mock_resp(200, {})) as mock_post:
            km.rotate_key("path", new_wif="MYWIF", field="bsv_key")
        body = mock_post.call_args[1]["json"]
        assert "bsv_key" in body["data"]
        assert body["data"]["bsv_key"] == "MYWIF"

    def test_rotate_key_403_raises_vault_error(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        with patch("httpx.post", return_value=_mock_resp(403)):
            with pytest.raises(ARIAVaultError) as exc_info:
                km.rotate_key("path", new_wif="NEWWIF")
        assert "403" in str(exc_info.value)

    def test_rotate_key_new_wif_never_in_error_message(self) -> None:
        """The new WIF must never appear in any ARIAVaultError message."""
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        sensitive_wif = "L1aW4aubDFB7yfras2S1mN3bqg9nwySY8nkoLmJebSLD5BWv3ENZ"
        with patch("httpx.post", return_value=_mock_resp(500)):
            with pytest.raises(ARIAVaultError) as exc_info:
                km.rotate_key("path", new_wif=sensitive_wif)
        assert sensitive_wif not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Async get_key_async tests
# ---------------------------------------------------------------------------


class TestGetKeyAsync:
    """VaultKeyManager.get_key_async() behaviour."""

    def test_get_key_async_success(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        mock_resp = _mock_resp(200, _kv_resp(_SAMPLE_WIF))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(km.get_key_async("aria/bsv/signing-key"))

        assert result == _SAMPLE_WIF

    def test_get_key_async_vault_error_raises(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        mock_resp = _mock_resp(500)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ARIAVaultError) as exc_info:
                asyncio.run(km.get_key_async("path"))

        assert "500" in str(exc_info.value)

    def test_get_key_async_missing_field_raises(self) -> None:
        km = VaultKeyManager(vault_addr=_VAULT_ADDR, token="tok")
        mock_resp = _mock_resp(200, {"data": {"data": {}}})

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ARIAVaultError):
                asyncio.run(km.get_key_async("path", field="wif"))
