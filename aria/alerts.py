"""
aria.alerts — Alert channel integrations for the ARIA WatchdogDaemon.

Provides ready-to-use alert channels that integrate with WatchdogDaemon's
``add_alert_handler()`` mechanism.  Each channel implements a ``send(alert)``
method that can be registered directly as a handler.

Supported channels:

  SlackAlertChannel    — POST to a Slack incoming webhook URL.
  EmailAlertChannel    — Send via SMTP (uses stdlib smtplib — no extra deps).
  WebhookAlertChannel  — POST JSON payload to any HTTP endpoint.
  PagerDutyChannel     — POST to PagerDuty Events API v2.
  MultiAlertChannel    — Fan-out to multiple channels simultaneously.
  LogAlertChannel      — Write to Python logger (useful for development).

Usage::

    from aria.alerts import SlackAlertChannel, MultiAlertChannel
    from aria.watchdog import WatchdogDaemon

    channels = MultiAlertChannel([
        SlackAlertChannel("https://hooks.slack.com/services/T.../B.../xxx"),
        LogAlertChannel(level="WARNING"),
    ])

    watchdog = WatchdogDaemon(storage=storage)
    watchdog.add_alert_handler(channels.send)
    watchdog.start()
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .watchdog import Alert

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base channel
# ---------------------------------------------------------------------------

class AlertChannelBase:
    """Base class for alert channels.

    Subclasses implement ``_deliver(alert)``.  ``send()`` wraps it with
    exception handling so a broken channel never crashes the watchdog.
    """

    def send(self, alert: "Alert") -> None:
        """Deliver *alert* to this channel.  Never raises."""
        try:
            self._deliver(alert)
        except Exception as exc:
            _log.error("%s delivery failed: %s", type(self).__name__, exc)

    def _deliver(self, alert: "Alert") -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Log channel — zero deps, great for development
# ---------------------------------------------------------------------------

class LogAlertChannel(AlertChannelBase):
    """Write alerts to a Python logger.

    Args:
        logger_name: Logger name (default ``"aria.alerts"``).
        level:       Log level as string (``"WARNING"``, ``"ERROR"``, etc.).
    """

    def __init__(self, logger_name: str = "aria.alerts", level: str = "WARNING") -> None:
        self._log = logging.getLogger(logger_name)
        self._level = getattr(logging, level.upper(), logging.WARNING)

    def _deliver(self, alert: "Alert") -> None:
        self._log.log(self._level, "[ARIA ALERT] %s", alert)


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

class SlackAlertChannel(AlertChannelBase):
    """Send alerts to a Slack channel via an incoming webhook.

    Get your webhook URL from:
    https://api.slack.com/messaging/webhooks

    Args:
        webhook_url: Slack incoming webhook URL.
        username:    Bot display name (default ``"ARIA Watchdog"``).
        icon_emoji:  Emoji icon (default ``":robot_face:"``).
    """

    _SEVERITY_EMOJI = {
        "INFO": ":information_source:",
        "WARNING": ":warning:",
        "CRITICAL": ":rotating_light:",
    }

    def __init__(
        self,
        webhook_url: str,
        username: str = "ARIA Watchdog",
        icon_emoji: str = ":robot_face:",
    ) -> None:
        self._url = webhook_url
        self._username = username
        self._icon = icon_emoji

    def _deliver(self, alert: "Alert") -> None:
        emoji = self._SEVERITY_EMOJI.get(str(alert.severity), ":bell:")
        text = (
            f"{emoji} *{alert.kind}* [{alert.severity}]\n"
            f"{alert.message}"
        )
        if alert.epoch_id:
            text += f"\n> epoch: `{alert.epoch_id}`"

        payload = {
            "username": self._username,
            "icon_emoji": self._icon,
            "text": text,
        }
        resp = httpx.post(self._url, json=payload, timeout=10.0)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

class EmailAlertChannel(AlertChannelBase):
    """Send alerts via SMTP email.  Uses stdlib smtplib — no extra deps.

    Args:
        smtp_host:    SMTP server hostname.
        smtp_port:    SMTP port (default 587 for STARTTLS).
        username:     SMTP auth username.
        password:     SMTP auth password.
        from_addr:    Sender email address.
        to_addrs:     List of recipient email addresses.
        use_tls:      Use STARTTLS (default True).
        subject_prefix: Email subject prefix (default ``"[ARIA Alert]"``).
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_addr: str = "aria@localhost",
        to_addrs: list[str] | None = None,
        use_tls: bool = True,
        subject_prefix: str = "[ARIA Alert]",
    ) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._user = username
        self._pass = password
        self._from = from_addr
        self._to = to_addrs or []
        self._tls = use_tls
        self._subject_prefix = subject_prefix

    def _deliver(self, alert: "Alert") -> None:
        subject = f"{self._subject_prefix} {alert.kind} [{alert.severity}]"
        body = (
            f"ARIA Watchdog Alert\n"
            f"{'=' * 50}\n\n"
            f"Kind:     {alert.kind}\n"
            f"Severity: {alert.severity}\n"
            f"Message:  {alert.message}\n"
        )
        if alert.epoch_id:
            body += f"Epoch ID: {alert.epoch_id}\n"
        if alert.details:
            body += f"\nDetails:\n{json.dumps(alert.details, indent=2)}\n"

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self._from
        msg["To"] = ", ".join(self._to)

        with smtplib.SMTP(self._host, self._port) as server:
            if self._tls:
                server.starttls()
            if self._user:
                server.login(self._user, self._pass)
            server.sendmail(self._from, self._to, msg.as_string())


# ---------------------------------------------------------------------------
# Generic Webhook
# ---------------------------------------------------------------------------

class WebhookAlertChannel(AlertChannelBase):
    """POST a JSON alert payload to any HTTP endpoint.

    Useful for custom dashboards, n8n, Zapier, Make, etc.

    Args:
        url:     Target endpoint URL.
        headers: Extra HTTP headers (e.g. Authorization).
        method:  HTTP method (default ``"POST"``).
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        method: str = "POST",
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._method = method.upper()

    def _deliver(self, alert: "Alert") -> None:
        payload = {
            "kind": alert.kind,
            "severity": str(alert.severity),
            "message": alert.message,
            "epoch_id": alert.epoch_id,
            "details": alert.details,
            "timestamp": alert.timestamp,
        }
        resp = httpx.request(
            self._method,
            self._url,
            json=payload,
            headers=self._headers,
            timeout=10.0,
        )
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# PagerDuty
# ---------------------------------------------------------------------------

class PagerDutyChannel(AlertChannelBase):
    """Send alerts to PagerDuty via the Events API v2.

    Get your integration key from:
    https://support.pagerduty.com/docs/services-and-integrations

    Args:
        integration_key: PagerDuty Events API v2 integration key.
        source:          Event source identifier (default ``"aria-watchdog"``).
        severity_map:    Map ARIA severity → PagerDuty severity.
                         Default: WARNING→warning, CRITICAL→critical, INFO→info.
    """

    _PAGERDUTY_URL = "https://events.pagerduty.com/v2/enqueue"

    _DEFAULT_SEVERITY_MAP = {
        "INFO": "info",
        "WARNING": "warning",
        "CRITICAL": "critical",
    }

    def __init__(
        self,
        integration_key: str,
        source: str = "aria-watchdog",
        severity_map: dict[str, str] | None = None,
    ) -> None:
        self._key = integration_key
        self._source = source
        self._severity_map = severity_map or self._DEFAULT_SEVERITY_MAP

    def _deliver(self, alert: "Alert") -> None:
        pd_severity = self._severity_map.get(str(alert.severity), "warning")
        payload = {
            "routing_key": self._key,
            "event_action": "trigger",
            "payload": {
                "summary": f"[ARIA] {alert.kind}: {alert.message}",
                "severity": pd_severity,
                "source": self._source,
                "custom_details": {
                    "kind": alert.kind,
                    "epoch_id": alert.epoch_id or "",
                    **alert.details,
                },
            },
        }
        resp = httpx.post(self._PAGERDUTY_URL, json=payload, timeout=10.0)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Multi-channel fan-out
# ---------------------------------------------------------------------------

class MultiAlertChannel(AlertChannelBase):
    """Fan-out a single alert to multiple channels concurrently.

    Each channel's ``send()`` is called in a separate daemon thread so a slow
    channel (e.g. SMTP) doesn't block others.

    Args:
        channels: List of ``AlertChannelBase`` instances.
    """

    def __init__(self, channels: list[AlertChannelBase]) -> None:
        self._channels = channels

    def _deliver(self, alert: "Alert") -> None:
        threads = [
            threading.Thread(target=ch.send, args=(alert,), daemon=True)
            for ch in self._channels
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)  # Wait max 15s per channel fan-out
