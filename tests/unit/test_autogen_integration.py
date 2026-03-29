"""Tests for aria.integrations.autogen — ARIAAutoGenHook."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aria.integrations.autogen import ARIAAutoGenHook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hook(auditor=None, aria=None, model_id="autogen-agent"):
    return ARIAAutoGenHook(auditor=auditor, aria=aria, model_id=model_id)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_model_id(self):
        hook = ARIAAutoGenHook()
        assert hook._model_id == "autogen-agent"

    def test_custom_model_id(self):
        hook = ARIAAutoGenHook(model_id="my-autogen")
        assert hook._model_id == "my-autogen"

    def test_no_auditor_or_aria_is_valid(self):
        hook = ARIAAutoGenHook()
        assert hook._auditor is None
        assert hook._aria is None


# ---------------------------------------------------------------------------
# on_agent_message_send
# ---------------------------------------------------------------------------

class TestOnAgentMessageSend:
    def test_calls_auditor_record(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_agent_message_send("UserProxy", "Coder", {"content": "hello"})
        auditor.record.assert_called_once()

    def test_model_id_includes_sender(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor, model_id="base")
        hook.on_agent_message_send("Alice", "Bob", "hi")
        args = auditor.record.call_args[0]
        assert args[0] == "base/Alice"

    def test_event_metadata_is_message_send(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_agent_message_send("A", "B", "msg")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["event"] == "message_send"

    def test_provider_metadata_is_autogen(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_agent_message_send("A", "B", "msg")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["provider"] == "autogen"

    def test_message_truncated_to_400(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        long_msg = "x" * 500
        hook.on_agent_message_send("A", "B", long_msg)
        args = auditor.record.call_args[0]
        input_data = args[1]
        assert len(input_data["content"]) == 400

    def test_dict_message_stringified(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_agent_message_send("A", "B", {"role": "user", "content": "test"})
        auditor.record.assert_called_once()

    def test_aria_backend_used(self):
        aria = MagicMock()
        hook = _hook(aria=aria)
        hook.on_agent_message_send("A", "B", "hello")
        aria.record.assert_called_once()

    def test_no_backend_is_noop_no_raise(self):
        hook = _hook()
        hook.on_agent_message_send("A", "B", "msg")  # must not raise

    def test_auditor_error_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("down")
        hook = _hook(auditor=auditor)
        hook.on_agent_message_send("A", "B", "msg")  # must not raise


# ---------------------------------------------------------------------------
# on_agent_message_receive
# ---------------------------------------------------------------------------

class TestOnAgentMessageReceive:
    def test_calls_auditor_record(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_agent_message_receive("Coder", "UserProxy", "reply")
        auditor.record.assert_called_once()

    def test_event_metadata_is_message_receive(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_agent_message_receive("B", "A", "msg")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["event"] == "message_receive"

    def test_sender_in_model_id(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor, model_id="base")
        hook.on_agent_message_receive("B", "A", "msg")
        args = auditor.record.call_args[0]
        # model_id = f"{model_id}/{sender_name}" — sender is "A"
        assert args[0] == "base/A"


# ---------------------------------------------------------------------------
# on_function_call
# ---------------------------------------------------------------------------

class TestOnFunctionCall:
    def test_calls_auditor_record(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_call("Coder", "execute_code", {"language": "python"})
        auditor.record.assert_called_once()

    def test_event_metadata_is_function_call(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_call("Coder", "exec", {})
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["event"] == "function_call"
        assert kwargs["metadata"]["function"] == "exec"

    def test_arguments_in_input_data(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_call("Coder", "fn", {"key": "val"})
        args = auditor.record.call_args[0]
        input_data = args[1]
        assert "fn" in input_data["function"]

    def test_arguments_truncated(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_call("Coder", "fn", {"k": "v" * 500})
        args = auditor.record.call_args[0]
        input_data = args[1]
        assert len(input_data["arguments"]) <= 400


# ---------------------------------------------------------------------------
# on_function_response
# ---------------------------------------------------------------------------

class TestOnFunctionResponse:
    def test_calls_auditor_record(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_response("Coder", "exec", "output text")
        auditor.record.assert_called_once()

    def test_event_metadata_is_function_response(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_response("Coder", "fn", "result")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["event"] == "function_response"

    def test_response_in_output_data(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_response("Coder", "fn", "the result")
        args = auditor.record.call_args[0]
        output_data = args[2]
        assert output_data["response"] == "the result"

    def test_response_truncated_to_400(self):
        auditor = MagicMock()
        hook = _hook(auditor=auditor)
        hook.on_function_response("Coder", "fn", "x" * 500)
        args = auditor.record.call_args[0]
        output_data = args[2]
        assert len(output_data["response"]) == 400

    def test_aria_backend_for_response(self):
        aria = MagicMock()
        hook = _hook(aria=aria)
        hook.on_function_response("A", "fn", "result")
        aria.record.assert_called_once()
