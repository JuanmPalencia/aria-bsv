"""Tests for aria.integrations.sagemaker — ARIASageMaker."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake boto3 injection helper
# ---------------------------------------------------------------------------

def _inject_boto3(invoke_response: dict | None = None) -> MagicMock:
    """Install a fake boto3 module and return the fake sagemaker-runtime client."""
    fake_boto3 = ModuleType("boto3")

    response_body = invoke_response or {"prediction": 1, "score": 0.99}
    body_bytes = json.dumps(response_body).encode()

    fake_client = MagicMock()
    fake_client.invoke_endpoint.return_value = {
        "Body": BytesIO(body_bytes),
        "ContentType": "application/json",
    }

    fake_boto3.client = MagicMock(return_value=fake_client)
    sys.modules["boto3"] = fake_boto3
    return fake_client


def _remove_boto3():
    sys.modules.pop("boto3", None)
    sys.modules.pop("aria.integrations.sagemaker", None)


# ---------------------------------------------------------------------------
# Import-level guard — no boto3
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_raises_import_error_when_boto3_missing(self):
        _remove_boto3()
        import importlib
        # Temporarily hide boto3
        real = sys.modules.pop("boto3", None)
        with pytest.raises(ImportError, match="boto3"):
            import importlib
            mod = importlib.import_module("aria.integrations.sagemaker")
            importlib.reload(mod)
            from aria.integrations.sagemaker import ARIASageMaker
            ARIASageMaker(endpoint_name="ep", auditor=MagicMock())
        if real:
            sys.modules["boto3"] = real


# ---------------------------------------------------------------------------
# ARIASageMaker — invoke
# ---------------------------------------------------------------------------

class TestARIASageMakerInvoke:
    def setup_method(self):
        self._fake_client = _inject_boto3({"prediction": 42, "score": 0.9})

    def teardown_method(self):
        _remove_boto3()

    def _make_client(self, auditor=None, aria=None):
        import importlib
        mod = importlib.import_module("aria.integrations.sagemaker")
        importlib.reload(mod)
        return mod.ARIASageMaker(
            endpoint_name="my-endpoint",
            auditor=auditor,
            aria=aria,
            model_id="fraud-v1",
            region_name="us-east-1",
        )

    def test_invoke_returns_parsed_response(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        result = client.invoke({"feature": 1.0})
        assert result["prediction"] == 42

    def test_invoke_calls_boto3_invoke_endpoint(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        client.invoke({"feature": 1.0})
        self._fake_client.invoke_endpoint.assert_called_once()

    def test_invoke_records_to_auditor(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        client.invoke({"feature": 1.0})
        auditor.record.assert_called_once()

    def test_invoke_payload_as_dict_serialised_to_json(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        client.invoke({"feature": 1.0})
        call_kwargs = self._fake_client.invoke_endpoint.call_args[1]
        # Body must be a JSON string
        body = call_kwargs.get("Body", "")
        parsed = json.loads(body)
        assert parsed["feature"] == 1.0

    def test_invoke_string_payload_sent_as_is(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        client.invoke('{"feature": 2.0}')
        call_kwargs = self._fake_client.invoke_endpoint.call_args[1]
        assert call_kwargs["Body"] == '{"feature": 2.0}'

    def test_metadata_includes_provider_sagemaker(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        client.invoke({"feature": 1.0})
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["provider"] == "sagemaker"

    def test_metadata_includes_endpoint(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        client.invoke({"feature": 1.0})
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["endpoint"] == "my-endpoint"

    def test_endpoint_name_used_as_model_id_when_none_supplied(self):
        import importlib
        mod = importlib.import_module("aria.integrations.sagemaker")
        importlib.reload(mod)
        auditor = MagicMock()
        client = mod.ARIASageMaker(
            endpoint_name="ep-default",
            auditor=auditor,
        )
        client.invoke({"x": 1})
        args = auditor.record.call_args[0]
        assert args[0] == "ep-default"

    def test_aria_backend_used(self):
        import importlib
        mod = importlib.import_module("aria.integrations.sagemaker")
        importlib.reload(mod)
        aria = MagicMock()
        client = mod.ARIASageMaker(
            endpoint_name="ep",
            aria=aria,
        )
        client.invoke({"x": 1})
        aria.record.assert_called_once()

    def test_auditor_error_does_not_raise(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("down")
        client = self._make_client(auditor=auditor)
        # Should not raise even when record() fails
        result = client.invoke({"feature": 1.0})
        assert result["prediction"] == 42
