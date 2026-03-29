"""
aria.integrations.instructor — ARIA audit integration for Instructor.

Instructor is a library for structured outputs from LLMs using Pydantic models.
It patches ``openai.OpenAI``, ``openai.AsyncOpenAI``, ``anthropic.Anthropic``,
and other clients to return validated Pydantic model instances instead of raw
API responses.

Usage::

    import openai
    import instructor
    from aria.integrations.instructor import ARIAInstructor, aria_patch

    # Option 1: Use ARIAInstructor factory method
    client = ARIAInstructor.from_openai(
        openai.OpenAI(),
        auditor=auditor,
        model_id="gpt-4o-structured",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=MyPydanticModel,
        messages=[{"role": "user", "content": "Extract the data."}],
    )
    # ↑ Records: model class name, prompt hash, structured output hash

    # Option 2: Convenience function
    patched = aria_patch(openai.OpenAI(), auditor=auditor)
    response = patched.chat.completions.create(
        model="gpt-4o",
        response_model=MyPydanticModel,
        messages=[...],
    )
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Type

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_input(messages: list[dict]) -> dict[str, Any]:
    """Convert messages list to a serialisable input dict."""
    return {
        "messages": [
            {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
            for m in (messages or [])
        ]
    }


def _structured_output_to_dict(response: Any) -> dict[str, Any]:
    """Convert a Pydantic model instance to a serialisable dict (truncated values)."""
    try:
        if hasattr(response, "model_dump"):
            return {k: str(v)[:500] for k, v in response.model_dump().items()}
        elif hasattr(response, "dict"):
            return {k: str(v)[:500] for k, v in response.dict().items()}
        return {"result": str(response)[:500]}
    except Exception:
        return {"result": str(response)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIAInstructorRecorder:
    """Shared recording logic for Instructor wrappers."""

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self.model_id = model_id

    def record(self, **kwargs: Any) -> None:
        try:
            if self._auditor is not None:
                self._auditor.record(
                    kwargs["model_id"],
                    kwargs["input_data"],
                    kwargs["output_data"],
                    confidence=kwargs.get("confidence"),
                    latency_ms=int(kwargs.get("latency_ms") or 0),
                    metadata=kwargs.get("metadata") or {},
                )
            elif self._aria is not None:
                self._aria.record(
                    model_id=kwargs["model_id"],
                    input_data=kwargs["input_data"],
                    output_data=kwargs["output_data"],
                    confidence=kwargs.get("confidence"),
                    latency_ms=kwargs.get("latency_ms"),
                    metadata=kwargs.get("metadata") or {},
                )
        except Exception as exc:
            _log.warning("ARIAInstructor: record error: %s", exc)


# ---------------------------------------------------------------------------
# Patched completions sub-resource
# ---------------------------------------------------------------------------

class _ARIAInstructorChatCompletions:
    """Wraps the ``chat.completions.create()`` method of an instructor-patched client."""

    def __init__(self, original: Any, recorder: "_ARIAInstructorRecorder") -> None:
        self._orig = original
        self._recorder = recorder

    def create(self, response_model: "Type[Any] | None" = None, **kwargs: Any) -> Any:
        """Create a structured completion and record to ARIA.

        Args:
            response_model: Pydantic model class for structured output.
            **kwargs:       Remaining kwargs forwarded to the underlying create().

        Returns:
            A validated Pydantic model instance (when response_model is set),
            or the raw API response.
        """
        base_model = kwargs.get("model", "instructor")
        if response_model is not None:
            schema_name = getattr(response_model, "__name__", str(response_model))
            model_id = self._recorder.model_id or f"{base_model}:{schema_name}"
        else:
            model_id = self._recorder.model_id or base_model

        messages = kwargs.get("messages", [])
        t0 = time.time()
        response = self._orig.create(response_model=response_model, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        output_data = (
            _structured_output_to_dict(response)
            if response_model is not None
            else {"result": str(response)[:500]}
        )

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=output_data,
            latency_ms=latency_ms,
            metadata={
                "provider": "instructor",
                "model": kwargs.get("model", ""),
                "response_model": (
                    getattr(response_model, "__name__", None)
                    if response_model is not None
                    else None
                ),
            },
        )
        return response


class _ARIAInstructorChat:
    """Wraps the ``chat`` attribute of an instructor-patched client."""

    def __init__(self, original: Any, recorder: "_ARIAInstructorRecorder") -> None:
        self.completions = _ARIAInstructorChatCompletions(original.completions, recorder)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class ARIAInstructor:
    """ARIA-audited wrapper around an instructor-patched OpenAI or Anthropic client.

    Use the factory class methods (``from_openai``, ``from_anthropic``) instead
    of the constructor directly.  The wrapper intercepts ``chat.completions.create``
    calls to record the Pydantic model class name, the prompt, and the structured
    output to ARIA.

    Args:
        client:   An instructor-patched client (result of ``instructor.from_openai()`` etc.).
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Override for the model_id label in ARIA records.

    Raises:
        ImportError: if the ``instructor`` package is not installed.
    """

    def __init__(
        self,
        client: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        try:
            import instructor  # type: ignore[import]  # noqa: F401
        except ImportError:
            raise ImportError(
                "instructor package not installed. pip install aria-bsv[instructor]"
            )
        self._client = client
        self._recorder = _ARIAInstructorRecorder(auditor=auditor, aria=aria, model_id=model_id)
        self.chat = _ARIAInstructorChat(client.chat, self._recorder)

    @classmethod
    def from_openai(
        cls,
        openai_client: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **instructor_kwargs: Any,
    ) -> "ARIAInstructor":
        """Create an ``ARIAInstructor`` from an ``openai.OpenAI()`` client.

        Args:
            openai_client:      An ``openai.OpenAI()`` instance.
            auditor:            ``InferenceAuditor`` instance.
            aria:               ``ARIAQuick`` instance.
            model_id:           Override for model_id in ARIA records.
            **instructor_kwargs: Extra kwargs forwarded to ``instructor.from_openai()``.

        Returns:
            An ``ARIAInstructor`` wrapping the instructor-patched client.
        """
        try:
            import instructor  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "instructor package not installed. pip install aria-bsv[instructor]"
            )
        patched = instructor.from_openai(openai_client, **instructor_kwargs)
        return cls(patched, auditor=auditor, aria=aria, model_id=model_id)

    @classmethod
    def from_anthropic(
        cls,
        anthropic_client: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **instructor_kwargs: Any,
    ) -> "ARIAInstructor":
        """Create an ``ARIAInstructor`` from an ``anthropic.Anthropic()`` client.

        Args:
            anthropic_client:   An ``anthropic.Anthropic()`` instance.
            auditor:            ``InferenceAuditor`` instance.
            aria:               ``ARIAQuick`` instance.
            model_id:           Override for model_id in ARIA records.
            **instructor_kwargs: Extra kwargs forwarded to ``instructor.from_anthropic()``.

        Returns:
            An ``ARIAInstructor`` wrapping the instructor-patched client.
        """
        try:
            import instructor  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "instructor package not installed. pip install aria-bsv[instructor]"
            )
        patched = instructor.from_anthropic(anthropic_client, **instructor_kwargs)
        return cls(patched, auditor=auditor, aria=aria, model_id=model_id)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying instructor-patched client."""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def aria_patch(
    client: Any,
    auditor: "InferenceAuditor | None" = None,
    aria: "ARIAQuick | None" = None,
    model_id: str | None = None,
    **instructor_kwargs: Any,
) -> ARIAInstructor:
    """Convenience function: patch a raw client with instructor and ARIA recording.

    Automatically detects whether ``client`` is an OpenAI or Anthropic client
    by inspecting its module name, then applies the appropriate instructor
    factory and wraps the result in ``ARIAInstructor``.

    Args:
        client:              An ``openai.OpenAI()``, ``openai.AsyncOpenAI()``,
                             or ``anthropic.Anthropic()`` instance.
        auditor:             ``InferenceAuditor`` instance.
        aria:                ``ARIAQuick`` instance.
        model_id:            Override for model_id in ARIA records.
        **instructor_kwargs: Extra kwargs forwarded to the instructor factory.

    Returns:
        An ``ARIAInstructor`` wrapping the instructor-patched client.

    Raises:
        ImportError: if the ``instructor`` package is not installed.
    """
    try:
        import instructor  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "instructor package not installed. pip install aria-bsv[instructor]"
        )

    client_module = type(client).__module__ or ""
    if "anthropic" in client_module:
        patched = instructor.from_anthropic(client, **instructor_kwargs)
    else:
        patched = instructor.from_openai(client, **instructor_kwargs)

    return ARIAInstructor(patched, auditor=auditor, aria=aria, model_id=model_id)
