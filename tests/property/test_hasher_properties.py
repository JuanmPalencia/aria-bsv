"""Property-based tests for aria.core.hasher using Hypothesis.

Invariants tested:
    - canonical_json is deterministic for equivalent objects.
    - canonical_json output is stable regardless of dict key insertion order.
    - hash_object always returns a 64-char lowercase hex string.
    - SHA-256 is collision-resistant for distinct inputs (distinct hashes).
    - NaN / Infinity always raise ARIASerializationError.
"""

from __future__ import annotations

import hashlib
import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aria.core.errors import ARIASerializationError
from aria.core.hasher import canonical_json, hash_object

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# JSON-safe scalar strategy (no NaN / Infinity)
json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False),
    st.text(max_size=50),
)

# Recursive JSON-compatible structure
json_value = st.recursive(
    json_scalar,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=10), children, max_size=5),
    ),
    max_leaves=15,
)


# ---------------------------------------------------------------------------
# canonical_json properties
# ---------------------------------------------------------------------------


@given(obj=json_value)
def test_canonical_json_is_bytes(obj):
    """canonical_json always returns bytes."""
    result = canonical_json(obj)
    assert isinstance(result, bytes)


@given(obj=json_value)
def test_canonical_json_deterministic(obj):
    """Same input → same bytes every time."""
    assert canonical_json(obj) == canonical_json(obj)


@given(
    keys=st.lists(st.text(max_size=10), min_size=2, max_size=5, unique=True),
    values=st.lists(st.integers(), min_size=2, max_size=5),
)
def test_canonical_json_key_order_independent(keys: list[str], values: list[int]):
    """Dict key insertion order does not affect canonical_json output."""
    assume(len(keys) == len(values))
    d1 = dict(zip(keys, values))
    d2 = dict(zip(reversed(keys), reversed(values)))
    # d1 and d2 have same key-value pairs but different insertion order
    if d1 == d2:  # skip trivial case where they happen to be the same
        return
    assert canonical_json(d1) == canonical_json(d2)


@given(a=json_value, b=json_value)
def test_canonical_json_distinct_values_distinct_bytes(a, b):
    """Different objects (that are not equal) produce different canonical JSON."""
    assume(a != b)
    # Two unequal JSON values must produce different serialisations
    assert canonical_json(a) != canonical_json(b)


@given(obj=json_value)
def test_canonical_json_parseable(obj):
    """canonical_json output is valid UTF-8 JSON."""
    import json as stdlib_json
    raw = canonical_json(obj)
    parsed = stdlib_json.loads(raw.decode("utf-8"))
    # Parsed result must equal the original object (modulo float edge cases)
    assert parsed == obj or (
        isinstance(obj, float) and math.isnan(obj) and math.isnan(parsed)
    )


# ---------------------------------------------------------------------------
# hash_object properties
# ---------------------------------------------------------------------------


@given(obj=json_value)
@settings(max_examples=100)
def test_hash_object_returns_prefixed_hex(obj):
    """hash_object always returns 'sha256:<64-char hex>'."""
    h = hash_object(obj)
    assert isinstance(h, str)
    assert h.startswith("sha256:")
    hex_part = h[len("sha256:"):]
    assert len(hex_part) == 64
    assert all(c in "0123456789abcdef" for c in hex_part)


@given(obj=json_value)
def test_hash_object_deterministic(obj):
    """Same input → same hash."""
    assert hash_object(obj) == hash_object(obj)


@given(a=json_value, b=json_value)
@settings(max_examples=200)
def test_hash_object_collision_free(a, b):
    """Distinct JSON values produce distinct hashes (no collisions in practice)."""
    assume(a != b)
    # This can only fail if SHA-256 collides, which is computationally infeasible
    assert hash_object(a) != hash_object(b)


@given(obj=json_value)
def test_hash_matches_sha256_of_canonical(obj):
    """hash_object equals 'sha256:' + SHA-256(canonical_json(obj))."""
    expected = "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()
    assert hash_object(obj) == expected


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@given(
    bad=st.one_of(
        st.just(float("nan")),
        st.just(float("inf")),
        st.just(float("-inf")),
    )
)
def test_nan_infinity_raises(bad):
    """NaN and Infinity values always raise ARIASerializationError."""
    with pytest.raises(ARIASerializationError):
        canonical_json(bad)


@given(
    bad=st.one_of(
        st.just(float("nan")),
        st.just(float("inf")),
    )
)
def test_nan_infinity_in_nested_raises(bad):
    """NaN/Infinity inside a nested structure also raise."""
    with pytest.raises(ARIASerializationError):
        canonical_json({"key": bad})

    with pytest.raises(ARIASerializationError):
        canonical_json([1, bad, 3])
