"""
aria.integrations.autogen — ARIA audit integration for Microsoft AutoGen.

Hooks into AutoGen v0.2 multi-agent conversation flows, recording every
agent message and function call as an AuditRecord.  No AutoGen import is
required at module load time — this is a pure Python adapter.

Usage::

    from aria.integrations.autogen import ARIAAutoGenHook
    from aria.quick import ARIAQuick

    aria = ARIAQuick(model_id="autogen-agent", db_path="aria.db")
    hook = ARIAAutoGenHook(aria=aria)

    # Instrument message passing manually (or via monkey-patching):
    hook.on_agent_message_send(
        sender_name="UserProxy",
        recipient_name="Assistant",
        message={"role": "user", "content": "What is BSV?"},
    )
    hook.on_agent_message_receive(
        recipient_name="Assistant",
        sender_name="UserProxy",
        message={"role": "user", "content": "What is BSV?"},
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Union

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


class ARIAAutoGenHook:
    """Hook class for Microsoft AutoGen v0.2 multi-agent conversations.

    Audits agent messages and function calls via ARIA.  Each event produces
    one AuditRecord with:

    - ``model_id``: ``f"{model_id}/{sender_name}"``
    - ``input_data``: ``{"from": sender_name, "to": recipient_name,
      "content": str(message)[:400]}``
    - ``output_data``: ``{"received_by": recipient_name,
      "content": str(message)[:400]}``
    - ``latency_ms``: ``0`` (messages are instantaneous events)
    - ``metadata``: ``{"provider": "autogen", "sender": sender_name,
      "recipient": recipient_name}``

    Args:
        auditor:  An initialised ``InferenceAuditor`` instance.
        aria:     An ``ARIAQuick`` instance (alternative to *auditor*).
        model_id: Base label prepended to each sender name.  Defaults to
                  ``"autogen-agent"``.

    Example::

        hook = ARIAAutoGenHook(aria=aria, model_id="my-autogen")
        hook.on_agent_message_send("UserProxy", "Coder",
                                   {"role": "user", "content": "Write a script"})
        hook.on_function_call("Coder", "execute_code",
                              {"language": "python", "code": "print('hi')"})
        hook.on_function_response("Coder", "execute_code", "hi\\n")
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str = "autogen-agent",
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Message hooks
    # ------------------------------------------------------------------

    def on_agent_message_send(
        self,
        sender_name: str,
        recipient_name: str,
        message: Union[dict, str],
    ) -> None:
        """Record an outgoing agent message as an AuditRecord.

        Args:
            sender_name:    Name of the agent sending the message.
            recipient_name: Name of the agent receiving the message.
            message:        Message payload — either a dict (with ``content``
                            key) or a plain string.
        """
        content = str(message)[:400]
        input_data: dict[str, Any] = {
            "from": sender_name,
            "to": recipient_name,
            "content": content,
        }
        output_data: dict[str, Any] = {
            "received_by": recipient_name,
            "content": content,
        }
        metadata: dict[str, Any] = {
            "provider": "autogen",
            "sender": sender_name,
            "recipient": recipient_name,
            "event": "message_send",
        }
        self._record(
            f"{self._model_id}/{sender_name}",
            input_data,
            output_data,
            None,
            0,
            metadata,
        )

    def on_agent_message_receive(
        self,
        recipient_name: str,
        sender_name: str,
        message: Union[dict, str],
    ) -> None:
        """Record an incoming agent message as an AuditRecord.

        Args:
            recipient_name: Name of the agent receiving the message.
            sender_name:    Name of the agent that sent the message.
            message:        Message payload — either a dict or a plain string.
        """
        content = str(message)[:400]
        input_data: dict[str, Any] = {
            "from": sender_name,
            "to": recipient_name,
            "content": content,
        }
        output_data: dict[str, Any] = {
            "received_by": recipient_name,
            "content": content,
        }
        metadata: dict[str, Any] = {
            "provider": "autogen",
            "sender": sender_name,
            "recipient": recipient_name,
            "event": "message_receive",
        }
        self._record(
            f"{self._model_id}/{sender_name}",
            input_data,
            output_data,
            None,
            0,
            metadata,
        )

    # ------------------------------------------------------------------
    # Function-call hooks
    # ------------------------------------------------------------------

    def on_function_call(
        self,
        agent_name: str,
        function_name: str,
        arguments: dict,
    ) -> None:
        """Record a function/tool call initiated by an agent.

        Args:
            agent_name:    Name of the agent making the function call.
            function_name: Name of the function being invoked.
            arguments:     Arguments dict passed to the function.
        """
        input_data: dict[str, Any] = {
            "agent": agent_name,
            "function": function_name,
            "arguments": repr(arguments)[:400],
        }
        output_data: dict[str, Any] = {
            "agent": agent_name,
            "function": function_name,
        }
        metadata: dict[str, Any] = {
            "provider": "autogen",
            "sender": agent_name,
            "recipient": agent_name,
            "event": "function_call",
            "function": function_name,
        }
        self._record(
            f"{self._model_id}/{agent_name}",
            input_data,
            output_data,
            None,
            0,
            metadata,
        )

    def on_function_response(
        self,
        agent_name: str,
        function_name: str,
        response: str,
    ) -> None:
        """Record the response returned from a function call.

        Args:
            agent_name:    Name of the agent that made the function call.
            function_name: Name of the function that returned the response.
            response:      Response string returned by the function.
        """
        input_data: dict[str, Any] = {
            "agent": agent_name,
            "function": function_name,
        }
        output_data: dict[str, Any] = {
            "agent": agent_name,
            "function": function_name,
            "response": response[:400],
        }
        metadata: dict[str, Any] = {
            "provider": "autogen",
            "sender": agent_name,
            "recipient": agent_name,
            "event": "function_response",
            "function": function_name,
        }
        self._record(
            f"{self._model_id}/{agent_name}",
            input_data,
            output_data,
            None,
            0,
            metadata,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _record(
        self,
        model_id: str,
        input_data: dict,
        output_data: dict,
        confidence: float | None,
        latency_ms: int,
        metadata: dict,
    ) -> None:
        try:
            if self._auditor is not None:
                self._auditor.record(
                    model_id,
                    input_data,
                    output_data,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
            elif self._aria is not None:
                self._aria.record(
                    model_id=model_id,
                    input_data=input_data,
                    output_data=output_data,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
        except Exception as exc:
            _log.warning("ARIAAutoGenHook: record error: %s", exc)
