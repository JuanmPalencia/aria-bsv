"""aria.integrations.vault — HashiCorp Vault key management for ARIA.

Fetches BSV signing keys from Vault's KV secrets engine instead of
environment variables. Supports both Vault Token and AppRole authentication.

Usage::

    from aria.integrations.vault import VaultKeyManager
    km = VaultKeyManager(vault_addr="https://vault.example.com", token="hvs.xxx")
    wif = km.get_key("aria/bsv/signing-key")  # fetches from KV path
    auditor = ARIAClient(system_id="my-system", signing_key=wif)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aria.core.errors import ARIAError

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ARIAVaultError(ARIAError):
    """Raised when a HashiCorp Vault operation fails.

    The message never contains key material, WIF strings, or any
    cryptographic derivative of the signing key.
    """


# ---------------------------------------------------------------------------
# VaultKeyManager
# ---------------------------------------------------------------------------


class VaultKeyManager:
    """Fetch and manage BSV signing keys stored in HashiCorp Vault.

    Supports both static Token authentication and dynamic AppRole
    authentication. All keys are fetched from Vault's KV v2 secrets engine.

    Security contract: WIF strings are NEVER logged or included in exception
    messages. Any Vault error surfaces as :class:`ARIAVaultError` with HTTP
    status information only — never key material.

    Args:
        vault_addr:  Full URL of the Vault server, e.g.
                     ``"https://vault.example.com"``.
        token:       Vault token for direct token authentication.
        role_id:     AppRole role_id (used together with ``secret_id``).
        secret_id:   AppRole secret_id (used together with ``role_id``).
        mount_path:  KV v2 mount path (default: ``"secret"``).
        namespace:   Vault Enterprise namespace header value (optional).
    """

    def __init__(
        self,
        vault_addr: str,
        token: str | None = None,
        role_id: str | None = None,
        secret_id: str | None = None,
        mount_path: str = "secret",
        namespace: str | None = None,
    ) -> None:
        self._vault_addr = vault_addr.rstrip("/")
        self._token = token
        self._role_id = role_id
        self._secret_id = secret_id
        self._mount_path = mount_path
        self._namespace = namespace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_headers(self) -> dict[str, str]:
        """Build Vault request headers, performing AppRole login if needed.

        If a static token is configured, uses it directly. If AppRole
        credentials are configured and no token is present, performs an
        AppRole login to obtain a short-lived token first. The obtained
        token is cached on the instance for subsequent calls.

        Returns:
            Dict of HTTP headers including ``X-Vault-Token`` and, when
            ``namespace`` was set, ``X-Vault-Namespace``.

        Raises:
            ARIAVaultError: If AppRole login fails or no credentials are
                            configured at all.
        """
        if self._token is None:
            if self._role_id and self._secret_id:
                self._token = self._login_approle()
            else:
                raise ARIAVaultError(
                    "vault configuration error: provide either a token "
                    "or AppRole credentials (role_id + secret_id)"
                )

        headers: dict[str, str] = {"X-Vault-Token": self._token}
        if self._namespace:
            headers["X-Vault-Namespace"] = self._namespace
        return headers

    def _login_approle(self) -> str:
        """Authenticate with Vault using AppRole and return the client token.

        Issues a POST to ``/v1/auth/approle/login`` with the configured
        ``role_id`` and ``secret_id``.

        Returns:
            The ``client_token`` string from the Vault auth response.

        Raises:
            ARIAVaultError: If the login request fails or returns a non-200
                            status code.
        """
        url = f"{self._vault_addr}/v1/auth/approle/login"
        try:
            resp = httpx.post(
                url,
                json={"role_id": self._role_id, "secret_id": self._secret_id},
            )
        except httpx.HTTPError:
            raise ARIAVaultError("vault request failed: connection error")

        if resp.status_code != 200:
            raise ARIAVaultError(f"vault request failed: {resp.status_code}")

        try:
            token: str = resp.json()["auth"]["client_token"]
        except (KeyError, TypeError):
            raise ARIAVaultError(
                "vault request failed: unexpected AppRole login response format"
            )

        return token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_key(self, path: str, field: str = "wif") -> str:
        """Fetch a BSV signing key from Vault KV v2.

        Issues a GET to ``/v1/{mount_path}/data/{path}`` and returns the
        value of ``field`` from the secret's ``data`` object.

        Args:
            path:   KV v2 secret path relative to the mount, e.g.
                    ``"aria/bsv/signing-key"``.
            field:  The data field within the secret that holds the WIF
                    (default: ``"wif"``).

        Returns:
            The WIF string stored at the given path and field.

        Raises:
            ARIAVaultError: If Vault returns an error or the field is absent.
                            The error message never contains the key value.
        """
        url = f"{self._vault_addr}/v1/{self._mount_path}/data/{path}"
        try:
            resp = httpx.get(url, headers=self._make_headers())
        except httpx.HTTPError:
            raise ARIAVaultError("vault request failed: connection error")

        if resp.status_code != 200:
            raise ARIAVaultError(f"vault request failed: {resp.status_code}")

        try:
            value: Any = resp.json()["data"]["data"][field]
        except (KeyError, TypeError):
            raise ARIAVaultError(
                f"vault request failed: field '{field}' not found in secret at '{path}'"
            )

        if not isinstance(value, str) or not value:
            raise ARIAVaultError("vault request failed: invalid key material format")

        return value

    async def get_key_async(self, path: str, field: str = "wif") -> str:
        """Async version of :meth:`get_key`.

        Uses ``httpx.AsyncClient`` so the call is non-blocking in an
        ``asyncio`` event loop.

        Args:
            path:   KV v2 secret path, e.g. ``"aria/bsv/signing-key"``.
            field:  The data field within the secret that holds the WIF
                    (default: ``"wif"``).

        Returns:
            The WIF string stored at the given path and field.

        Raises:
            ARIAVaultError: If Vault returns an error or the field is absent.
                            The error message never contains the key value.
        """
        url = f"{self._vault_addr}/v1/{self._mount_path}/data/{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=self._make_headers())
        except httpx.HTTPError:
            raise ARIAVaultError("vault request failed: connection error")

        if resp.status_code != 200:
            raise ARIAVaultError(f"vault request failed: {resp.status_code}")

        try:
            value: Any = resp.json()["data"]["data"][field]
        except (KeyError, TypeError):
            raise ARIAVaultError(
                f"vault request failed: field '{field}' not found in secret at '{path}'"
            )

        if not isinstance(value, str) or not value:
            raise ARIAVaultError("vault request failed: invalid key material format")

        return value

    def rotate_key(self, path: str, new_wif: str, field: str = "wif") -> None:
        """Write a new signing key to Vault KV v2 (key rotation).

        Issues a POST to ``/v1/{mount_path}/data/{path}`` with the new key
        in the ``data`` object. Vault automatically creates a new secret
        version while preserving the previous one in its version history.

        Security: ``new_wif`` is never logged or included in exception
        messages.

        Args:
            path:     KV v2 secret path, e.g. ``"aria/bsv/signing-key"``.
            new_wif:  The new WIF key to store.
            field:    The data field name to write (default: ``"wif"``).

        Raises:
            ARIAVaultError: If the write operation fails. The error message
                            never exposes ``new_wif``.
        """
        url = f"{self._vault_addr}/v1/{self._mount_path}/data/{path}"
        payload = {"data": {field: new_wif}}
        try:
            resp = httpx.post(url, json=payload, headers=self._make_headers())
        except httpx.HTTPError:
            raise ARIAVaultError("vault request failed: connection error")

        if resp.status_code not in (200, 204):
            raise ARIAVaultError(f"vault request failed: {resp.status_code}")
