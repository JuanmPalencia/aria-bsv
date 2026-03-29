"""Tests for aria.integrations.instructor — ARIAInstructor and aria_patch."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake instructor module factory
# ---------------------------------------------------------------------------

def _make_instructor_module() -> ModuleType:
    """Build a minimal fake ``instructor`` module."""
    mod = ModuleType("instructor")
    # from_openai / from_anthropic return a mock patched client
    mod.from_openai = MagicMock(side_effect=lambda client, **kw: _make_patched_client())
    mod.from_anthropic = MagicMock(side_effect=lambda client, **kw: _make_patched_client())
    return mod


def _make_patched_client(response=None):
    """Build a minimal mock instructor-patched client."""
    client = MagicMock()
    client.chat.completions.create = MagicMock(
        return_value=response or _make_pydantic_response()
    )
    return client


def _make_pydantic_response(name="Paris"):
    """A mock Pydantic model instance with model_dump()."""
    resp = MagicMock()
    resp.model_dump.return_value = {"city": name}
    return resp


# ---------------------------------------------------------------------------
# Helper: reload integration module with fake instructor
# ---------------------------------------------------------------------------

def _reload_instructor_integration(instructor_mod=None):
    import importlib
    import aria.integrations.instructor as m
    mod = instructor_mod or _make_instructor_module()
    with patch.dict(sys.modules, {"instructor": mod}):
        importlib.reload(m)
        return m, mod


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------

class TestMessagesToInput:
    def test_basic_conversion(self):
        m, _ = _reload_instructor_integration()
        msgs = [{"role": "user", "content": "Extract entity"}]
        result = m._messages_to_input(msgs)
        assert result["messages"][0]["role"] == "user"

    def test_truncation(self):
        m, _ = _reload_instructor_integration()
        msgs = [{"role": "user", "content": "a" * 1000}]
        result = m._messages_to_input(msgs)
        assert len(result["messages"][0]["content"]) == 500

    def test_empty_messages(self):
        m, _ = _reload_instructor_integration()
        assert m._messages_to_input([]) == {"messages": []}


class TestStructuredOutputToDict:
    def test_model_dump_used(self):
        m, _ = _reload_instructor_integration()
        resp = _make_pydantic_response("London")
        out = m._structured_output_to_dict(resp)
        assert out["city"] == "London"

    def test_fallback_to_dict_method(self):
        m, _ = _reload_instructor_integration()
        resp = MagicMock(spec=[])  # no model_dump
        resp.dict = MagicMock(return_value={"key": "val"})
        out = m._structured_output_to_dict(resp)
        assert out["key"] == "val"

    def test_fallback_to_str_on_exception(self):
        m, _ = _reload_instructor_integration()
        out = m._structured_output_to_dict(None)
        assert "result" in out


# ---------------------------------------------------------------------------
# Tests for ARIAInstructor
# ---------------------------------------------------------------------------

class TestARIAInstructorImportError:
    def test_import_error_when_instructor_missing(self):
        with patch.dict(sys.modules, {"instructor": None}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            patched = _make_patched_client()
            with pytest.raises(ImportError, match="instructor"):
                m.ARIAInstructor(client=patched, auditor=MagicMock())

    def test_from_openai_import_error(self):
        with patch.dict(sys.modules, {"instructor": None}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="instructor"):
                m.ARIAInstructor.from_openai(MagicMock(), auditor=MagicMock())

    def test_from_anthropic_import_error(self):
        with patch.dict(sys.modules, {"instructor": None}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="instructor"):
                m.ARIAInstructor.from_anthropic(MagicMock(), auditor=MagicMock())


class TestARIAInstructorCreate:
    def _build(self, auditor=None, aria=None, model_id=None, response=None):
        instr_mod = _make_instructor_module()
        patched_client = _make_patched_client(response=response or _make_pydantic_response())
        instr_mod.from_openai.side_effect = None
        instr_mod.from_openai.return_value = patched_client
        with patch.dict(sys.modules, {"instructor": instr_mod}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            wrapper = m.ARIAInstructor.from_openai(
                MagicMock(), auditor=auditor, aria=aria, model_id=model_id
            )
            return wrapper, patched_client

    def test_create_records_to_auditor(self):
        auditor = MagicMock()
        wrapper, _ = self._build(auditor=auditor)
        wrapper.chat.completions.create(
            model="gpt-4o",
            response_model=MagicMock(__name__="CityExtraction"),
            messages=[{"role": "user", "content": "Extract the city"}],
        )
        auditor.record.assert_called_once()

    def test_create_records_to_aria(self):
        aria = MagicMock()
        wrapper, _ = self._build(aria=aria)
        wrapper.chat.completions.create(
            model="gpt-4o",
            response_model=MagicMock(__name__="EntityModel"),
            messages=[],
        )
        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert "model_id" in kwargs

    def test_model_id_override(self):
        auditor = MagicMock()
        wrapper, _ = self._build(auditor=auditor, model_id="my-structured-model")
        wrapper.chat.completions.create(
            model="gpt-4o",
            response_model=MagicMock(__name__="MyModel"),
            messages=[],
        )
        args = auditor.record.call_args[0]
        assert args[0] == "my-structured-model"

    def test_schema_name_embedded_in_model_id(self):
        auditor = MagicMock()
        wrapper, _ = self._build(auditor=auditor)  # no model_id override

        class MySchema:
            pass

        wrapper.chat.completions.create(
            model="gpt-4o",
            response_model=MySchema,
            messages=[],
        )
        args = auditor.record.call_args[0]
        assert "MySchema" in args[0]

    def test_output_data_uses_model_dump(self):
        auditor = MagicMock()
        resp = _make_pydantic_response("Berlin")
        wrapper, _ = self._build(auditor=auditor, response=resp)
        wrapper.chat.completions.create(
            model="gpt-4o",
            response_model=MagicMock(__name__="CityModel"),
            messages=[],
        )
        args = auditor.record.call_args[0]
        output_data = args[2]
        assert output_data.get("city") == "Berlin"

    def test_record_error_is_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("storage fail")
        wrapper, _ = self._build(auditor=auditor)
        # Must not raise
        result = wrapper.chat.completions.create(
            model="gpt-4o",
            response_model=MagicMock(__name__="M"),
            messages=[],
        )
        assert result is not None

    def test_getattr_proxies_to_patched_client(self):
        instr_mod = _make_instructor_module()
        patched_client = _make_patched_client()
        patched_client.api_key = "sk-fake"
        with patch.dict(sys.modules, {"instructor": instr_mod}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            # Construct directly so self._client is exactly patched_client
            wrapper = m.ARIAInstructor(client=patched_client, auditor=MagicMock())
            assert wrapper.api_key == "sk-fake"

    def test_metadata_includes_response_model_name(self):
        auditor = MagicMock()
        wrapper, _ = self._build(auditor=auditor)
        wrapper.chat.completions.create(
            model="gpt-4o",
            response_model=MagicMock(__name__="AddressModel"),
            messages=[],
        )
        kwargs = auditor.record.call_args[1]
        meta = kwargs.get("metadata", {})
        assert meta.get("response_model") == "AddressModel"


class TestAriaQatch:
    def test_aria_patch_detects_openai(self):
        instr_mod = _make_instructor_module()
        openai_client = MagicMock()
        openai_client.__class__.__module__ = "openai"
        with patch.dict(sys.modules, {"instructor": instr_mod}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            wrapper = m.aria_patch(openai_client, auditor=MagicMock())
            assert isinstance(wrapper, m.ARIAInstructor)
            instr_mod.from_openai.assert_called_once()

    def test_aria_patch_detects_anthropic(self):
        instr_mod = _make_instructor_module()
        anthropic_client = MagicMock()
        anthropic_client.__class__.__module__ = "anthropic"
        with patch.dict(sys.modules, {"instructor": instr_mod}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            wrapper = m.aria_patch(anthropic_client, auditor=MagicMock())
            assert isinstance(wrapper, m.ARIAInstructor)
            instr_mod.from_anthropic.assert_called_once()

    def test_aria_patch_import_error(self):
        with patch.dict(sys.modules, {"instructor": None}):
            import importlib
            import aria.integrations.instructor as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="instructor"):
                m.aria_patch(MagicMock(), auditor=MagicMock())
