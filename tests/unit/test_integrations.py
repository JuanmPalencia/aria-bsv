"""Tests for Django, Flask, and LangChain integrations."""

from __future__ import annotations

import json
import os
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.django_settings")

import django
try:
    django.setup()
except RuntimeError:
    pass

from aria.auditor import AuditConfig, InferenceAuditor
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _MockWallet(WalletInterface):
    async def sign_and_broadcast(self, payload: dict) -> str:
        return "a" * 64


class _MockBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid="b" * 64, propagated=True)


def _make_auditor(model_id: str = "test-model") -> InferenceAuditor:
    storage = SQLiteStorage("sqlite://")
    config = AuditConfig(
        system_id="test-system",
        bsv_key="placeholder",
        network="mainnet",
        batch_ms=30000,
        batch_size=1000,
    )
    return InferenceAuditor(
        config=config,
        model_hashes={model_id: "sha256:" + "a" * 64},
        _wallet=_MockWallet(),
        _broadcaster=_MockBroadcaster(),
        _storage=storage,
    )


# ---------------------------------------------------------------------------
# Django integration
# ---------------------------------------------------------------------------

class TestDjangoAuditView:
    def test_audit_view_records_inference(self):
        """audit_view decorator calls auditor.record() once per request."""
        from unittest.mock import MagicMock, patch
        from aria.integrations.django import audit_view

        auditor = _make_auditor()
        recorded = []
        original_record = auditor.record
        auditor.record = lambda *a, **kw: recorded.append(a) or original_record(*a, **kw)

        @audit_view("test-model", auditor)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"result": "ok"})

        from django.test import RequestFactory
        factory = RequestFactory()
        req = factory.post("/", data=json.dumps({"q": "test"}), content_type="application/json")
        response = my_view(req)

        assert response.status_code == 200
        assert len(recorded) == 1
        auditor.close()

    def test_audit_view_strips_pii_fields(self):
        """PII fields are stripped from input before recording."""
        from aria.integrations.django import audit_view

        auditor = _make_auditor()
        captured_inputs = []
        original_record = auditor.record

        def _capture(*args, **kwargs):
            captured_inputs.append(kwargs.get("input_data", args[1] if len(args) > 1 else None))
            return original_record(*args, **kwargs)

        auditor.record = _capture

        @audit_view("test-model", auditor, pii_fields=["ssn"])
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        from django.test import RequestFactory
        factory = RequestFactory()
        req = factory.post(
            "/",
            data=json.dumps({"ssn": "123-45-6789", "age": 35}),
            content_type="application/json",
        )
        my_view(req)

        # record(model_id, input_data, ...) — input_data is positional arg [1]
        assert len(captured_inputs) == 1
        inp = captured_inputs[0]  # captured_inputs holds the input dict directly
        assert "ssn" not in inp
        assert "age" in inp
        auditor.close()

    def test_audit_view_never_breaks_view_on_auditor_error(self):
        """If auditor.record raises, the view response is still returned."""
        from aria.integrations.django import audit_view

        auditor = _make_auditor()
        auditor.record = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore

        @audit_view("test-model", auditor)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        from django.test import RequestFactory
        req = RequestFactory().post("/", content_type="application/json", data=b"{}")
        response = my_view(req)
        assert response.status_code == 200

    def test_aria_middleware_adds_headers(self):
        """ARIAMiddleware adds X-ARIA-System-ID header."""
        from aria.integrations.django import ARIAMiddleware
        from django.http import HttpResponse

        auditor = _make_auditor()

        def get_response(request):
            return HttpResponse("ok")

        middleware = ARIAMiddleware(get_response=get_response, auditor=auditor)

        from django.test import RequestFactory
        req = RequestFactory().get("/")
        response = middleware(req)
        assert "X-ARIA-System-ID" in response
        assert response["X-ARIA-System-ID"] == "test-system"
        auditor.close()


# ---------------------------------------------------------------------------
# Flask integration
# ---------------------------------------------------------------------------

class TestFlaskAuditRoute:
    def _make_flask_app(self, auditor, model_id="test-model"):
        from flask import Flask, jsonify, request as flask_req
        from aria.integrations.flask import audit_route, ARIAFlask

        app = Flask(__name__)
        aria_ext = ARIAFlask(auditor=auditor)
        aria_ext.init_app(app)

        @app.route("/predict", methods=["POST"])
        @audit_route(model_id, auditor)
        def predict():
            data = flask_req.get_json()
            return jsonify({"result": "ok", "input": data})

        return app

    def test_audit_route_records_inference(self):
        auditor = _make_auditor()
        recorded = []
        original = auditor.record
        auditor.record = lambda *a, **kw: recorded.append(True) or original(*a, **kw)

        app = self._make_flask_app(auditor)
        with app.test_client() as client:
            resp = client.post("/predict", json={"q": "hello"})
            assert resp.status_code == 200

        assert len(recorded) == 1
        auditor.close()

    def test_aria_flask_adds_system_id_header(self):
        auditor = _make_auditor()
        app = self._make_flask_app(auditor)

        with app.test_client() as client:
            resp = client.post("/predict", json={"q": "hello"})
            assert "X-ARIA-System-ID" in resp.headers
            assert resp.headers["X-ARIA-System-ID"] == "test-system"

        auditor.close()

    def test_audit_route_strips_pii(self):
        from flask import Flask, jsonify, request as flask_req
        from aria.integrations.flask import audit_route

        auditor = _make_auditor()
        captured = []
        original = auditor.record

        def _capture(*args, **kw):
            # record(model_id, input_data, output_data, ...) — input_data is args[1]
            captured.append(args[1] if len(args) > 1 else kw.get("input_data", {}))
            return original(*args, **kw)

        auditor.record = _capture

        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        @audit_route("test-model", auditor, pii_fields=["patient_id"])
        def predict():
            return jsonify({"ok": True})

        with app.test_client() as client:
            client.post("/predict", json={"patient_id": "P123", "age": 42})

        assert len(captured) == 1
        assert "patient_id" not in captured[0]
        assert "age" in captured[0]
        auditor.close()

    def test_audit_route_never_breaks_on_auditor_error(self):
        from flask import Flask, jsonify
        from aria.integrations.flask import audit_route

        auditor = _make_auditor()
        auditor.record = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("crash"))  # type: ignore

        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        @audit_route("test-model", auditor)
        def predict():
            return jsonify({"ok": True})

        with app.test_client() as client:
            resp = client.post("/predict", json={})
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# LangChain integration
# ---------------------------------------------------------------------------

class TestLangChainCallback:
    def _make_llm_result(self, texts: list[str]):
        """Build a minimal LLMResult for testing."""
        from langchain_core.outputs import LLMResult, Generation
        return LLMResult(
            generations=[[Generation(text=t)] for t in texts],
            llm_output={"finish_reason": "stop"},
        )

    def test_callback_records_on_llm_end(self):
        from uuid import uuid4
        from aria.integrations.langchain import ARIACallbackHandler

        auditor = _make_auditor()
        recorded = []
        original = auditor.record

        def _capture(*a, **kw):
            recorded.append({"model_id": a[0], "input": a[1] if len(a) > 1 else None, **kw})
            return original(*a, **kw)

        auditor.record = _capture

        handler = ARIACallbackHandler(auditor=auditor, model_id="test-model")
        run_id = uuid4()
        handler.on_llm_start({"name": "test-model"}, ["What is ARIA?"], run_id=run_id)
        handler.on_llm_end(self._make_llm_result(["ARIA is a protocol."]), run_id=run_id)

        assert len(recorded) == 1
        assert recorded[0]["model_id"] == "test-model"
        assert recorded[0]["latency_ms"] >= 0
        auditor.close()

    def test_callback_cleanup_on_error(self):
        from uuid import uuid4
        from aria.integrations.langchain import ARIACallbackHandler

        auditor = _make_auditor()
        handler = ARIACallbackHandler(auditor=auditor, model_id="test-model")
        run_id = uuid4()
        handler.on_llm_start({"name": "test-model"}, ["hello"], run_id=run_id)
        handler.on_llm_error(RuntimeError("API error"), run_id=run_id)

        # State should be cleaned up
        assert str(run_id) not in handler._start_times
        assert str(run_id) not in handler._inputs
        auditor.close()

    def test_chat_model_start_captures_messages(self):
        from uuid import uuid4
        from langchain_core.messages import HumanMessage, SystemMessage
        from aria.integrations.langchain import ARIACallbackHandler

        auditor = _make_auditor()
        captured = []
        original = auditor.record

        def _capture(*a, **kw):
            # record(model_id, input_data, output_data, ...) — input_data is args[1]
            captured.append({"model_id": a[0], "input_data": a[1] if len(a) > 1 else {}, **kw})
            return original(*a, **kw)

        auditor.record = _capture

        handler = ARIACallbackHandler(auditor=auditor, model_id="test-model")
        run_id = uuid4()
        messages = [[SystemMessage(content="You are helpful."), HumanMessage(content="Hi")]]
        handler.on_chat_model_start({"name": "gpt-4"}, messages, run_id=run_id)
        handler.on_llm_end(self._make_llm_result(["Hello!"]), run_id=run_id)

        assert len(captured) == 1
        assert "messages" in captured[0]["input_data"]
        auditor.close()

    def test_audited_llm_wraps_invoke(self):
        """ARIAAuditedLLM.invoke passes callbacks config to underlying LLM."""
        from unittest.mock import MagicMock
        from aria.integrations.langchain import ARIAAuditedLLM

        auditor = _make_auditor()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "test response"

        wrapped = ARIAAuditedLLM(llm=mock_llm, auditor=auditor, model_id="test-model")
        result = wrapped.invoke("test input")

        assert result == "test response"
        mock_llm.invoke.assert_called_once()
        # Verify callbacks were injected
        call_kwargs = mock_llm.invoke.call_args
        config = call_kwargs[1].get("config") or call_kwargs[0][1]
        assert len(config.get("callbacks", [])) >= 1
        auditor.close()
