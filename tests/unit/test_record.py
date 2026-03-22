"""
tests/unit/test_record.py

Unit tests for aria.core.record.AuditRecord.

Coverage targets:
  - Auto-computed fields (record_id, aria_version)
  - Field validation (__post_init__)
  - to_canonical_dict structure and ordering
  - hash() format, determinism, distinctness
"""

import pytest

from aria.core.errors import ARIASerializationError
from aria.core.record import ARIA_VERSION, AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_HASH = "sha256:" + "a" * 64
VALID_HASH_2 = "sha256:" + "b" * 64


def make_record(**overrides) -> AuditRecord:
    defaults = dict(
        epoch_id="ep_1742848200000_0001",
        model_id="severity_clf",
        input_hash=VALID_HASH,
        output_hash=VALID_HASH_2,
        sequence=0,
    )
    defaults.update(overrides)
    return AuditRecord(**defaults)


# ---------------------------------------------------------------------------
# Auto-computed fields
# ---------------------------------------------------------------------------

class TestAutoComputedFields:

    def test_aria_version_is_constant(self) -> None:
        r = make_record()
        assert r.aria_version == ARIA_VERSION
        assert r.aria_version == "1.0"

    def test_record_id_format(self) -> None:
        r = make_record(epoch_id="ep_1742848200000_0001", sequence=7)
        assert r.record_id == "rec_ep_1742848200000_0001_000007"

    def test_record_id_zero_padded_sequence(self) -> None:
        r = make_record(sequence=0)
        assert r.record_id.endswith("_000000")

    def test_record_id_large_sequence(self) -> None:
        r = make_record(sequence=999999)
        assert r.record_id.endswith("_999999")

    def test_aria_version_not_in_constructor(self) -> None:
        # aria_version is init=False; passing it should raise TypeError
        with pytest.raises(TypeError):
            AuditRecord(  # type: ignore[call-arg]
                epoch_id="ep_1",
                model_id="m",
                input_hash=VALID_HASH,
                output_hash=VALID_HASH_2,
                sequence=0,
                aria_version="2.0",
            )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_confidence_defaults_to_none(self) -> None:
        assert make_record().confidence is None

    def test_latency_ms_defaults_to_zero(self) -> None:
        assert make_record().latency_ms == 0

    def test_metadata_defaults_to_empty_dict(self) -> None:
        r = make_record()
        assert r.metadata == {}

    def test_metadata_instances_are_independent(self) -> None:
        # Default factory must create a new dict per instance
        r1 = make_record()
        r2 = make_record()
        r1.metadata["x"] = 1
        assert "x" not in r2.metadata


# ---------------------------------------------------------------------------
# Validation — required string fields
# ---------------------------------------------------------------------------

class TestValidationRequiredFields:

    def test_empty_epoch_id_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="epoch_id"):
            make_record(epoch_id="")

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="model_id"):
            make_record(model_id="")


# ---------------------------------------------------------------------------
# Validation — hash format
# ---------------------------------------------------------------------------

class TestValidationHashFormat:

    def test_input_hash_missing_prefix_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="input_hash"):
            make_record(input_hash="deadbeef" * 8)

    def test_output_hash_missing_prefix_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="output_hash"):
            make_record(output_hash="deadbeef" * 8)

    def test_input_hash_too_short_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="input_hash"):
            make_record(input_hash="sha256:abc")

    def test_output_hash_invalid_hex_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="output_hash"):
            make_record(output_hash="sha256:" + "z" * 64)

    def test_valid_lowercase_hash_accepted(self) -> None:
        r = make_record(input_hash="sha256:" + "f" * 64)
        assert r.input_hash == "sha256:" + "f" * 64


# ---------------------------------------------------------------------------
# Validation — numeric fields
# ---------------------------------------------------------------------------

class TestValidationNumericFields:

    def test_negative_sequence_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="sequence"):
            make_record(sequence=-1)

    def test_zero_sequence_accepted(self) -> None:
        assert make_record(sequence=0).sequence == 0

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="latency_ms"):
            make_record(latency_ms=-1)

    def test_zero_latency_accepted(self) -> None:
        assert make_record(latency_ms=0).latency_ms == 0

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="confidence"):
            make_record(confidence=-0.01)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="confidence"):
            make_record(confidence=1.01)

    def test_confidence_zero_accepted(self) -> None:
        assert make_record(confidence=0.0).confidence == 0.0

    def test_confidence_one_accepted(self) -> None:
        assert make_record(confidence=1.0).confidence == 1.0

    def test_confidence_nan_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="confidence"):
            make_record(confidence=float("nan"))

    def test_confidence_infinity_raises(self) -> None:
        with pytest.raises(ARIASerializationError, match="confidence"):
            make_record(confidence=float("inf"))


# ---------------------------------------------------------------------------
# to_canonical_dict
# ---------------------------------------------------------------------------

class TestToCanonicalDict:

    def test_all_expected_keys_present(self) -> None:
        d = make_record().to_canonical_dict()
        expected_keys = {
            "aria_version", "confidence", "epoch_id", "input_hash",
            "latency_ms", "metadata", "model_id", "output_hash",
            "record_id", "sequence",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match_record(self) -> None:
        r = make_record(
            epoch_id="ep_test",
            model_id="clf",
            sequence=5,
            confidence=0.9,
            latency_ms=42,
        )
        d = r.to_canonical_dict()
        assert d["epoch_id"] == "ep_test"
        assert d["model_id"] == "clf"
        assert d["sequence"] == 5
        assert d["confidence"] == 0.9
        assert d["latency_ms"] == 42
        assert d["aria_version"] == "1.0"
        assert d["record_id"] == r.record_id

    def test_confidence_none_included(self) -> None:
        d = make_record(confidence=None).to_canonical_dict()
        assert d["confidence"] is None


# ---------------------------------------------------------------------------
# hash()
# ---------------------------------------------------------------------------

class TestRecordHash:

    def test_returns_sha256_prefix(self) -> None:
        assert make_record().hash().startswith("sha256:")

    def test_deterministic(self) -> None:
        r = make_record()
        assert r.hash() == r.hash()

    def test_same_data_same_hash(self) -> None:
        r1 = make_record()
        r2 = make_record()
        assert r1.hash() == r2.hash()

    def test_different_model_different_hash(self) -> None:
        r1 = make_record(model_id="model_a")
        r2 = make_record(model_id="model_b")
        assert r1.hash() != r2.hash()

    def test_different_sequence_different_hash(self) -> None:
        r1 = make_record(sequence=0)
        r2 = make_record(sequence=1)
        assert r1.hash() != r2.hash()

    def test_different_output_hash_different_record_hash(self) -> None:
        r1 = make_record(output_hash="sha256:" + "a" * 64)
        r2 = make_record(output_hash="sha256:" + "b" * 64)
        assert r1.hash() != r2.hash()

    def test_hash_hex_part_is_64_chars(self) -> None:
        h = make_record().hash()
        assert len(h[len("sha256:"):]) == 64
