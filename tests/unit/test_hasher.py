"""
tests/unit/test_hasher.py

Unit tests for aria.core.hasher.

Coverage targets:
  - canonical_json: determinism, key sorting, None, NaN, Infinity, bad types
  - hash_object: format, distinctness
  - hash_file: happy path, missing file, unreadable file
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from aria.core.errors import ARIASerializationError
from aria.core.hasher import canonical_json, hash_file, hash_object


# ---------------------------------------------------------------------------
# canonical_json — determinism
# ---------------------------------------------------------------------------

class TestCanonicalJsonDeterminism:

    def test_same_dict_same_bytes(self) -> None:
        obj = {"z": 1, "a": 2, "m": 3}
        assert canonical_json(obj) == canonical_json(obj)

    def test_key_order_does_not_matter(self) -> None:
        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        assert canonical_json(a) == canonical_json(b)

    def test_nested_key_order_sorted(self) -> None:
        obj = {"outer_z": {"inner_z": 1, "inner_a": 2}, "outer_a": 0}
        result = json.loads(canonical_json(obj))
        assert list(result.keys()) == ["outer_a", "outer_z"]
        assert list(result["outer_z"].keys()) == ["inner_a", "inner_z"]

    def test_no_whitespace(self) -> None:
        data = canonical_json({"key": "value"})
        text = data.decode("utf-8")
        assert " " not in text
        assert "\n" not in text
        assert "\t" not in text

    def test_utf8_encoding(self) -> None:
        obj = {"name": "José", "city": "Málaga"}
        result = canonical_json(obj)
        assert isinstance(result, bytes)
        # Must decode cleanly as UTF-8
        result.decode("utf-8")

    def test_unicode_not_escaped(self) -> None:
        obj = {"emoji": "🚑"}
        text = canonical_json(obj).decode("utf-8")
        assert "🚑" in text


# ---------------------------------------------------------------------------
# canonical_json — value types
# ---------------------------------------------------------------------------

class TestCanonicalJsonValueTypes:

    def test_none_serialises_as_null(self) -> None:
        result = canonical_json({"x": None})
        assert result == b'{"x":null}'

    def test_integer(self) -> None:
        assert canonical_json(42) == b"42"

    def test_float_round_trip(self) -> None:
        value = 0.1 + 0.2
        serialised = canonical_json(value)
        reconstructed = json.loads(serialised)
        assert reconstructed == value

    def test_boolean_true(self) -> None:
        assert canonical_json(True) == b"true"

    def test_boolean_false(self) -> None:
        assert canonical_json(False) == b"false"

    def test_empty_dict(self) -> None:
        assert canonical_json({}) == b"{}"

    def test_empty_list(self) -> None:
        assert canonical_json([]) == b"[]"

    def test_nested_list_order_preserved(self) -> None:
        obj = {"items": [3, 1, 2]}
        result = json.loads(canonical_json(obj))
        assert result["items"] == [3, 1, 2]  # order must NOT change

    def test_list_of_dicts_keys_sorted(self) -> None:
        obj = [{"z": 1, "a": 2}, {"z": 3, "a": 4}]
        result = json.loads(canonical_json(obj))
        for item in result:
            assert list(item.keys()) == ["a", "z"]


# ---------------------------------------------------------------------------
# canonical_json — forbidden values
# ---------------------------------------------------------------------------

class TestCanonicalJsonForbiddenValues:

    def test_nan_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            canonical_json(float("nan"))

    def test_nan_in_dict_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            canonical_json({"value": float("nan")})

    def test_positive_infinity_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            canonical_json(float("inf"))

    def test_negative_infinity_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            canonical_json(float("-inf"))

    def test_set_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            canonical_json({1, 2, 3})

    def test_bytes_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            canonical_json(b"raw bytes")

    def test_custom_object_raises(self) -> None:
        class Foo:
            pass
        with pytest.raises(ARIASerializationError):
            canonical_json(Foo())


# ---------------------------------------------------------------------------
# hash_object
# ---------------------------------------------------------------------------

class TestHashObject:

    def test_returns_sha256_prefix(self) -> None:
        result = hash_object({"x": 1})
        assert result.startswith("sha256:")

    def test_hex_part_is_64_chars(self) -> None:
        result = hash_object({"x": 1})
        hex_part = result[len("sha256:"):]
        assert len(hex_part) == 64

    def test_hex_part_is_valid_hex(self) -> None:
        result = hash_object({"x": 1})
        hex_part = result[len("sha256:"):]
        bytes.fromhex(hex_part)  # raises if invalid

    def test_same_object_same_hash(self) -> None:
        obj = {"model": "clf", "score": 0.95}
        assert hash_object(obj) == hash_object(obj)

    def test_different_objects_different_hashes(self) -> None:
        assert hash_object({"a": 1}) != hash_object({"a": 2})

    def test_key_order_irrelevant(self) -> None:
        assert hash_object({"z": 1, "a": 2}) == hash_object({"a": 2, "z": 1})

    def test_nan_propagates_error(self) -> None:
        with pytest.raises(ARIASerializationError):
            hash_object({"bad": float("nan")})

    def test_known_hash(self) -> None:
        # Regression: empty dict canonical JSON is b"{}" → known SHA-256
        import hashlib
        expected = "sha256:" + hashlib.sha256(b"{}").hexdigest()
        assert hash_object({}) == expected


# ---------------------------------------------------------------------------
# hash_file
# ---------------------------------------------------------------------------

class TestHashFile:

    def test_returns_sha256_prefix(self, tmp_path: Path) -> None:
        f = tmp_path / "model.bin"
        f.write_bytes(b"fake model weights")
        assert hash_file(f).startswith("sha256:")

    def test_same_file_same_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "model.bin"
        f.write_bytes(b"deterministic content")
        assert hash_file(f) == hash_file(f)

    def test_different_contents_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"version one")
        f2.write_bytes(b"version two")
        assert hash_file(f1) != hash_file(f2)

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        f = tmp_path / "model.bin"
        f.write_bytes(b"content")
        result = hash_file(str(f))  # str, not Path
        assert result.startswith("sha256:")

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/path/model.bin")

    def test_known_hash(self, tmp_path: Path) -> None:
        import hashlib
        content = b"aria test content"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert hash_file(f) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        import hashlib
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = "sha256:" + hashlib.sha256(b"").hexdigest()
        assert hash_file(f) == expected
