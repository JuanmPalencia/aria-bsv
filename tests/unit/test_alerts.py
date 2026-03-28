"""Tests for aria.alerts — alert channel implementations."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from aria.alerts import (
    AlertChannelBase,
    EmailAlertChannel,
    LogAlertChannel,
    MultiAlertChannel,
    PagerDutyChannel,
    SlackAlertChannel,
    WebhookAlertChannel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAlert:
    def __init__(
        self,
        kind="LATENCY_SPIKE",
        severity="WARNING",
        message="Test alert",
        epoch_id="ep-123",
        details=None,
        timestamp=1700000000.0,
    ):
        self.kind = kind
        self.severity = severity
        self.message = message
        self.epoch_id = epoch_id
        self.details = details or {}
        self.timestamp = timestamp

    def __str__(self):
        return f"{self.kind} [{self.severity}] {self.message}"


# ---------------------------------------------------------------------------
# AlertChannelBase
# ---------------------------------------------------------------------------

class TestAlertChannelBase:
    def test_send_calls_deliver(self):
        delivered = []

        class MyChannel(AlertChannelBase):
            def _deliver(self, alert):
                delivered.append(alert)

        ch = MyChannel()
        alert = FakeAlert()
        ch.send(alert)
        assert len(delivered) == 1
        assert delivered[0] is alert

    def test_send_catches_deliver_exception(self):
        class BrokenChannel(AlertChannelBase):
            def _deliver(self, alert):
                raise RuntimeError("network down")

        ch = BrokenChannel()
        # Must not raise
        ch.send(FakeAlert())

    def test_base_deliver_raises_not_implemented(self):
        ch = AlertChannelBase()
        with pytest.raises(NotImplementedError):
            ch._deliver(FakeAlert())


# ---------------------------------------------------------------------------
# LogAlertChannel
# ---------------------------------------------------------------------------

class TestLogAlertChannel:
    def test_log_alert_warning_level(self, caplog):
        ch = LogAlertChannel(level="WARNING")
        with caplog.at_level(logging.WARNING, logger="aria.alerts"):
            ch.send(FakeAlert(message="slow response"))
        assert "slow response" in caplog.text

    def test_log_alert_critical_level(self, caplog):
        ch = LogAlertChannel(level="CRITICAL")
        with caplog.at_level(logging.CRITICAL, logger="aria.alerts"):
            ch.send(FakeAlert(message="system failure"))
        assert "system failure" in caplog.text

    def test_custom_logger_name(self):
        ch = LogAlertChannel(logger_name="myapp.alerts", level="INFO")
        assert ch._log.name == "myapp.alerts"

    def test_default_level_is_warning(self):
        ch = LogAlertChannel()
        assert ch._level == logging.WARNING


# ---------------------------------------------------------------------------
# SlackAlertChannel
# ---------------------------------------------------------------------------

class TestSlackAlertChannel:
    def test_posts_to_webhook(self):
        ch = SlackAlertChannel(webhook_url="https://hooks.slack.com/xxx")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            ch.send(FakeAlert(kind="CONFIDENCE_DROP", severity="CRITICAL"))

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://hooks.slack.com/xxx"
        payload = kwargs["json"]
        assert "CONFIDENCE_DROP" in payload["text"]
        assert "CRITICAL" in payload["text"]

    def test_includes_epoch_id_in_message(self):
        ch = SlackAlertChannel(webhook_url="https://x.com/hook")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            ch.send(FakeAlert(epoch_id="epoch-abc-123"))

        payload = mock_post.call_args[1]["json"]
        assert "epoch-abc-123" in payload["text"]

    def test_no_epoch_id_no_epoch_line(self):
        ch = SlackAlertChannel(webhook_url="https://x.com/hook")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            alert = FakeAlert()
            alert.epoch_id = None
            ch.send(alert)

        payload = mock_post.call_args[1]["json"]
        assert "epoch:" not in payload["text"]

    def test_http_error_does_not_propagate(self):
        ch = SlackAlertChannel(webhook_url="https://x.com/hook")
        with patch("httpx.post", side_effect=Exception("connection refused")):
            # Should not raise (caught by AlertChannelBase.send)
            ch.send(FakeAlert())

    def test_custom_username(self):
        ch = SlackAlertChannel(
            webhook_url="https://x.com",
            username="MyBot",
            icon_emoji=":fire:",
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            ch.send(FakeAlert())

        payload = mock_post.call_args[1]["json"]
        assert payload["username"] == "MyBot"
        assert payload["icon_emoji"] == ":fire:"


# ---------------------------------------------------------------------------
# WebhookAlertChannel
# ---------------------------------------------------------------------------

class TestWebhookAlertChannel:
    def test_posts_json_payload(self):
        ch = WebhookAlertChannel(url="https://my-server.com/alert")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            ch.send(FakeAlert(kind="STUCK_EPOCH", severity="INFO"))

        mock_req.assert_called_once()
        kwargs = mock_req.call_args[1]
        payload = kwargs["json"]
        assert payload["kind"] == "STUCK_EPOCH"
        assert payload["severity"] == "INFO"
        assert "message" in payload
        assert "epoch_id" in payload

    def test_custom_headers(self):
        ch = WebhookAlertChannel(
            url="https://x.com",
            headers={"Authorization": "Bearer token123"},
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            ch.send(FakeAlert())

        headers = mock_req.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer token123"

    def test_default_method_is_post(self):
        ch = WebhookAlertChannel(url="https://x.com")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            ch.send(FakeAlert())

        method = mock_req.call_args[0][0]
        assert method == "POST"


# ---------------------------------------------------------------------------
# EmailAlertChannel
# ---------------------------------------------------------------------------

class TestEmailAlertChannel:
    def test_sends_email_via_smtp(self):
        ch = EmailAlertChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user@example.com",
            password="secret",
            from_addr="aria@example.com",
            to_addrs=["admin@example.com"],
        )

        mock_server = MagicMock()
        mock_smtp = MagicMock(return_value=mock_server)
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", mock_smtp):
            ch.send(FakeAlert(kind="EPOCH_MISMATCH", severity="CRITICAL"))

        mock_server.sendmail.assert_called_once()
        from_arg, to_arg, msg_str = mock_server.sendmail.call_args[0]
        assert from_arg == "aria@example.com"
        assert "admin@example.com" in to_arg

    def test_subject_contains_kind_and_severity(self):
        ch = EmailAlertChannel(
            smtp_host="smtp.example.com",
            to_addrs=["x@y.com"],
        )

        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        captured = {}

        def fake_sendmail(from_, to_, msg_str):
            captured["msg"] = msg_str

        mock_server.sendmail = fake_sendmail

        with patch("smtplib.SMTP", return_value=mock_server):
            ch.send(FakeAlert(kind="DRIFT_ALERT", severity="WARNING"))

        assert "DRIFT_ALERT" in captured["msg"]
        assert "WARNING" in captured["msg"]

    def test_no_tls_skips_starttls(self):
        ch = EmailAlertChannel(
            smtp_host="smtp.example.com",
            to_addrs=["x@y.com"],
            use_tls=False,
        )
        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_server):
            ch.send(FakeAlert())

        mock_server.starttls.assert_not_called()


# ---------------------------------------------------------------------------
# PagerDutyChannel
# ---------------------------------------------------------------------------

class TestPagerDutyChannel:
    def test_posts_to_pagerduty_url(self):
        ch = PagerDutyChannel(integration_key="abc123")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            ch.send(FakeAlert(kind="CONFIDENCE_DROP", severity="CRITICAL"))

        url = mock_post.call_args[0][0]
        assert "pagerduty" in url
        payload = mock_post.call_args[1]["json"]
        assert payload["routing_key"] == "abc123"
        assert payload["event_action"] == "trigger"
        assert payload["payload"]["severity"] == "critical"

    def test_severity_mapping(self):
        ch = PagerDutyChannel(integration_key="key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        for aria_sev, pd_sev in [("INFO", "info"), ("WARNING", "warning"), ("CRITICAL", "critical")]:
            with patch("httpx.post", return_value=mock_resp) as mock_post:
                alert = FakeAlert(severity=aria_sev)
                ch.send(alert)
            pd_payload = mock_post.call_args[1]["json"]["payload"]
            assert pd_payload["severity"] == pd_sev

    def test_custom_severity_map(self):
        ch = PagerDutyChannel(
            integration_key="key",
            severity_map={"WARNING": "error", "CRITICAL": "critical"},
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            ch.send(FakeAlert(severity="WARNING"))

        pd_payload = mock_post.call_args[1]["json"]["payload"]
        assert pd_payload["severity"] == "error"


# ---------------------------------------------------------------------------
# MultiAlertChannel
# ---------------------------------------------------------------------------

class TestMultiAlertChannel:
    def test_delivers_to_all_channels(self):
        delivered = []

        class CountChannel(AlertChannelBase):
            def __init__(self, name):
                self.name = name
            def _deliver(self, alert):
                delivered.append(self.name)

        multi = MultiAlertChannel([CountChannel("a"), CountChannel("b"), CountChannel("c")])
        multi.send(FakeAlert())
        assert sorted(delivered) == ["a", "b", "c"]

    def test_broken_channel_does_not_block_others(self):
        delivered = []

        class GoodChannel(AlertChannelBase):
            def _deliver(self, alert):
                delivered.append("good")

        class BadChannel(AlertChannelBase):
            def _deliver(self, alert):
                raise RuntimeError("bad channel")

        multi = MultiAlertChannel([BadChannel(), GoodChannel()])
        multi.send(FakeAlert())
        assert "good" in delivered

    def test_channels_called_concurrently(self):
        order = []
        lock = threading.Lock()

        class SlowChannel(AlertChannelBase):
            def _deliver(self, alert):
                import time
                time.sleep(0.05)
                with lock:
                    order.append("slow")

        class FastChannel(AlertChannelBase):
            def _deliver(self, alert):
                with lock:
                    order.append("fast")

        multi = MultiAlertChannel([SlowChannel(), FastChannel()])
        start = threading.Event()

        def run():
            start.wait()
            multi.send(FakeAlert())

        t = threading.Thread(target=run)
        t.start()
        start.set()
        t.join(timeout=2.0)

        # Both should have been called
        assert "slow" in order
        assert "fast" in order

    def test_empty_channels_list_ok(self):
        multi = MultiAlertChannel([])
        multi.send(FakeAlert())  # Should not raise
