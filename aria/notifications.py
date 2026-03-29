"""
aria.notifications — Event-driven notification system.

Builds on top of aria.events (EventBus) and aria.alerts (channels) to
provide a simple, configurable notification layer.

Usage::

    from aria.notifications import NotificationManager

    nm = NotificationManager()
    nm.on_low_confidence(threshold=0.7, channel="slack")
    nm.on_epoch_close(channel="webhook")
    nm.on_drift(channel="email")

    # Or configure from dict/env:
    nm = NotificationManager.from_config({
        "slack_url": "https://hooks.slack.com/...",
        "email_to": "audit@company.com",
        "rules": [
            {"event": "low_confidence", "threshold": 0.7},
            {"event": "epoch_close"},
        ]
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("aria.notifications")


@dataclass
class NotificationRule:
    """A rule that triggers a notification when conditions are met."""
    event: str
    channel: str = "log"
    threshold: float | None = None
    callback: Callable[..., Any] | None = None
    enabled: bool = True


@dataclass
class Notification:
    """A notification payload."""
    event: str
    message: str
    severity: str = "info"
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "message": self.message,
            "severity": self.severity,
            "data": self.data,
        }


class NotificationManager:
    """Configurable notification manager for ARIA events.

    Supports multiple channels (log, slack, webhook, email) and
    event-based rules with optional thresholds.
    """

    def __init__(self) -> None:
        self._rules: list[NotificationRule] = []
        self._channels: dict[str, Callable[[Notification], None]] = {
            "log": self._log_channel,
        }
        self._history: list[Notification] = []
        self._max_history: int = 1000

    # ------------------------------------------------------------------
    # Channel registration
    # ------------------------------------------------------------------

    def add_channel(self, name: str, handler: Callable[[Notification], None]) -> None:
        """Register a custom notification channel."""
        self._channels[name] = handler

    def add_slack(self, webhook_url: str) -> None:
        """Add a Slack webhook channel."""
        import httpx

        def send(n: Notification) -> None:
            try:
                httpx.post(
                    webhook_url,
                    json={"text": f"[ARIA] {n.severity.upper()}: {n.message}"},
                    timeout=10.0,
                )
            except Exception as exc:
                logger.warning("Slack notification failed: %s", exc)

        self._channels["slack"] = send

    def add_webhook(self, url: str, headers: dict[str, str] | None = None) -> None:
        """Add a generic webhook channel."""
        import httpx

        def send(n: Notification) -> None:
            try:
                httpx.post(
                    url,
                    json=n.to_dict(),
                    headers=headers or {},
                    timeout=10.0,
                )
            except Exception as exc:
                logger.warning("Webhook notification failed: %s", exc)

        self._channels["webhook"] = send

    # ------------------------------------------------------------------
    # Rule configuration
    # ------------------------------------------------------------------

    def on_low_confidence(
        self, threshold: float = 0.7, channel: str = "log"
    ) -> None:
        """Notify when a record confidence is below threshold."""
        self._rules.append(NotificationRule(
            event="low_confidence", channel=channel, threshold=threshold,
        ))

    def on_epoch_close(self, channel: str = "log") -> None:
        """Notify when an epoch is closed."""
        self._rules.append(NotificationRule(event="epoch_close", channel=channel))

    def on_drift(self, channel: str = "log") -> None:
        """Notify when drift is detected."""
        self._rules.append(NotificationRule(event="drift", channel=channel))

    def on_compliance_fail(self, channel: str = "log") -> None:
        """Notify on compliance failures."""
        self._rules.append(NotificationRule(
            event="compliance_fail", channel=channel,
        ))

    def on_event(
        self,
        event: str,
        channel: str = "log",
        threshold: float | None = None,
    ) -> None:
        """Add a custom event rule."""
        self._rules.append(NotificationRule(
            event=event, channel=channel, threshold=threshold,
        ))

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process_record(self, record: Any) -> list[Notification]:
        """Check a record against all rules and fire notifications."""
        fired: list[Notification] = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            if rule.event == "low_confidence":
                conf = getattr(record, "confidence", None)
                if conf is not None and rule.threshold is not None and conf < rule.threshold:
                    n = Notification(
                        event="low_confidence",
                        message=f"Low confidence {conf:.3f} < {rule.threshold} "
                                f"for record {getattr(record, 'record_id', '?')}",
                        severity="warning",
                        data={"confidence": conf, "threshold": rule.threshold,
                              "record_id": getattr(record, "record_id", "")},
                    )
                    self._fire(n, rule)
                    fired.append(n)

        return fired

    def process_event(
        self, event_type: str, data: dict[str, Any] | None = None
    ) -> list[Notification]:
        """Process a named event against all rules."""
        fired: list[Notification] = []
        data = data or {}

        for rule in self._rules:
            if not rule.enabled or rule.event != event_type:
                continue

            n = Notification(
                event=event_type,
                message=f"Event: {event_type}",
                severity="info",
                data=data,
            )
            self._fire(n, rule)
            fired.append(n)

        return fired

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[Notification]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NotificationManager":
        """Create a NotificationManager from a configuration dict.

        Config keys:
            slack_url: Slack webhook URL
            webhook_url: Generic webhook URL
            rules: List of {"event": ..., "threshold": ..., "channel": ...}
        """
        nm = cls()

        if "slack_url" in config:
            nm.add_slack(config["slack_url"])
        if "webhook_url" in config:
            nm.add_webhook(config["webhook_url"])

        for rule in config.get("rules", []):
            nm.on_event(
                event=rule["event"],
                channel=rule.get("channel", "log"),
                threshold=rule.get("threshold"),
            )

        return nm

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self, notification: Notification, rule: NotificationRule) -> None:
        """Dispatch a notification to its channel."""
        self._history.append(notification)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        channel_fn = self._channels.get(rule.channel, self._log_channel)
        try:
            channel_fn(notification)
        except Exception as exc:
            logger.error("Notification channel %s failed: %s", rule.channel, exc)

        if rule.callback:
            try:
                rule.callback(notification)
            except Exception as exc:
                logger.error("Notification callback failed: %s", exc)

    @staticmethod
    def _log_channel(notification: Notification) -> None:
        """Default channel: log the notification."""
        logger.info(
            "[%s] %s: %s",
            notification.severity.upper(),
            notification.event,
            notification.message,
        )
