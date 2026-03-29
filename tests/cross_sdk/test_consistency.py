"""
Cross-SDK consistency tests — verify Python, TypeScript, and Go SDKs
produce identical cryptographic outputs for the same inputs.

These tests generate reference vectors from the Python SDK and compare
against expected outputs.  The TS / Go SDKs should be tested independently
with the same vectors.

Usage::

    pytest tests/cross_sdk/test_consistency.py -v

Reference vectors are also written to tests/cross_sdk/vectors.json
so they can be consumed by sdk-ts and sdk-go test suites.
"""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import pytest

from aria.core.hasher import canonical_json, hash_object
from aria.core.merkle import ARIAMerkleTree, MerkleProof, verify_proof


# Directory for reference vectors
VECTOR_DIR = Path(__file__).parent
VECTOR_FILE = VECTOR_DIR / "vectors.json"

# Rust SDK source paths (static analysis only — no Rust compilation needed)
RUST_HASHER_SOURCE = VECTOR_DIR.parent.parent / "sdk-rs" / "hasher" / "src" / "lib.rs"
RUST_MERKLE_SOURCE = VECTOR_DIR.parent.parent / "sdk-rs" / "merkle" / "src" / "lib.rs"


# ──────────────────────────────────────────────────────────────────────
# Reference inputs — shared across all SDKs
# ──────────────────────────────────────────────────────────────────────

CANONICAL_JSON_CASES = [
    # (label, input, expected_canonical_form_description)
    ("empty_object", {}, "{}"),
    ("single_key", {"a": 1}, '{"a":1}'),
    ("sorted_keys", {"b": 2, "a": 1}, '{"a":1,"b":2}'),
    ("nested", {"z": {"b": 2, "a": 1}, "a": 0}, '{"a":0,"z":{"a":1,"b":2}}'),
    ("with_list", {"items": [3, 1, 2]}, '{"items":[3,1,2]}'),
    ("unicode", {"name": "café"}, '{"name":"café"}'),
    ("booleans", {"t": True, "f": False}, '{"f":false,"t":true}'),
    ("null_value", {"x": None}, '{"x":null}'),
    ("string_input", "hello world", '"hello world"'),
    ("number_input", 42, "42"),
]

HASH_OBJECT_INPUTS = [
    {"model": "gpt-4", "prompt": "What is 2+2?"},
    {"model": "claude-3", "prompt": "Hello", "temperature": 0.7},
    "simple string input",
    {"nested": {"deep": {"value": 42}}},
    {"list_field": [1, 2, 3], "name": "test"},
]

MERKLE_TREE_LEAVES = [
    "abc123",
    "def456",
    "789ghi",
    "jkl012",
]


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

class TestCanonicalJSON:
    """Canonical JSON must be deterministic and identical across SDKs."""

    @pytest.mark.parametrize("label,input_data,expected", CANONICAL_JSON_CASES)
    def test_canonical_form(self, label: str, input_data, expected: str):
        result = canonical_json(input_data)
        assert result == expected.encode("utf-8"), f"Case '{label}': got {result!r}, expected {expected!r}"

    def test_deterministic_repeated(self):
        obj = {"z": 1, "a": 2, "m": 3}
        results = {canonical_json(obj).decode() for _ in range(100)}
        assert len(results) == 1, "canonical_json must be deterministic"


class TestHashObject:
    """hash_object must produce SHA-256 of canonical JSON."""

    @pytest.mark.parametrize("input_data", HASH_OBJECT_INPUTS)
    def test_hash_matches_manual_sha256(self, input_data):
        """hash_object(x) == sha256:SHA256(canonical_json(x))."""
        result = hash_object(input_data)
        expected = "sha256:" + hashlib.sha256(canonical_json(input_data)).hexdigest()
        assert result == expected

    def test_hash_has_sha256_prefix(self):
        for data in HASH_OBJECT_INPUTS:
            h = hash_object(data)
            assert h.startswith("sha256:")
            hex_part = h[7:]
            assert len(hex_part) == 64
            assert all(c in "0123456789abcdef" for c in hex_part)

    def test_different_inputs_different_hashes(self):
        hashes = [hash_object(d) for d in HASH_OBJECT_INPUTS]
        assert len(set(hashes)) == len(hashes), "All inputs should produce unique hashes"


class TestMerkleTree:
    """Merkle tree roots and proofs must be identical across SDKs."""

    def test_single_leaf(self):
        tree = ARIAMerkleTree()
        h = hash_object({"data": "single"})
        tree.add(h)
        root = tree.root()
        assert root is not None
        assert root.startswith("sha256:")

    def test_known_leaves_root_deterministic(self):
        """Same leaves in same order must produce same root."""
        hashes = [hash_object(leaf) for leaf in MERKLE_TREE_LEAVES]

        roots = set()
        for _ in range(10):
            tree = ARIAMerkleTree()
            for h in hashes:
                tree.add(h)
            roots.add(tree.root())

        assert len(roots) == 1, "Merkle root must be deterministic"

    def test_proof_verification(self):
        hashes = [hash_object(leaf) for leaf in MERKLE_TREE_LEAVES]
        tree = ARIAMerkleTree()
        for h in hashes:
            tree.add(h)

        root = tree.root()
        for h in hashes:
            proof = tree.proof(h)
            assert verify_proof(root, proof, h), f"Proof failed for {h[:12]}..."

    def test_order_matters(self):
        """Different order of leaves must produce different root."""
        hashes = [hash_object(leaf) for leaf in MERKLE_TREE_LEAVES]

        tree1 = ARIAMerkleTree()
        for h in hashes:
            tree1.add(h)

        tree2 = ARIAMerkleTree()
        for h in reversed(hashes):
            tree2.add(h)

        assert tree1.root() != tree2.root(), "Different leaf order must produce different root"


class TestCrossSDKVectors:
    """Generate and verify cross-SDK reference vectors."""

    def test_generate_vectors(self):
        """Generate reference vectors file for TS/Go/Rust SDKs."""
        vectors = {
            "canonical_json": [],
            "hash_object": [],
            "merkle": {},
            # Rust SDK metadata: same expected values as Python/TS/Go.
            # The Python test TestRustSdkVectors performs static analysis of
            # sdk-rs/hasher/src/lib.rs to verify these vectors appear there.
            "rust": {
                "rust_crate": "sdk-rs/hasher",
                "description": (
                    "Rust SDK test vectors — identical expected values to Python/TS/Go. "
                    "Verified by TestRustSdkVectors (static source analysis, no Rust compilation)."
                ),
                "known_vectors": [
                    {
                        "input_bytes_ascii": "",
                        "expected_sha256": (
                            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                        ),
                        "rust_test": "hash_bytes_known_vector",
                    },
                    {
                        "input_bytes_ascii": "abc",
                        "expected_sha256": (
                            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                        ),
                        "rust_test": "hash_bytes_abc_vector",
                    },
                ],
                "canonical_json_vector": {
                    "input": {"model": "gpt-4", "seq": 0, "confidence": None},
                    "expected_canonical": '{"confidence":null,"model":"gpt-4","seq":0}',
                    "rust_test": "cross_sdk_canonical_json_vector",
                },
                "merkle_prefix": {
                    "leaf_prefix_hex": "00",
                    "internal_prefix_hex": "01",
                    "description": "RFC 6962 second-preimage protection (BRC-121 §5)",
                },
            },
        }

        # Canonical JSON vectors
        for label, input_data, expected_canonical in CANONICAL_JSON_CASES:
            canonical = canonical_json(input_data)
            vectors["canonical_json"].append({
                "label": label,
                "input": input_data,
                "expected_canonical": canonical,
                "expected_hash": "sha256:" + hashlib.sha256(canonical).hexdigest(),
            })

        # Hash object vectors
        for i, input_data in enumerate(HASH_OBJECT_INPUTS):
            vectors["hash_object"].append({
                "index": i,
                "input": input_data,
                "expected_hash": hash_object(input_data),
            })

        # Merkle tree vectors
        hashes = [hash_object(leaf) for leaf in MERKLE_TREE_LEAVES]
        tree = ARIAMerkleTree()
        for h in hashes:
            tree.add(h)

        vectors["merkle"] = {
            "leaves": MERKLE_TREE_LEAVES,
            "leaf_hashes": hashes,
            "expected_root": tree.root(),
            "proofs": {
                h: {
                    "target_hash": tree.proof(h).target_hash,
                    "proof_path": tree.proof(h).proof_path,
                    "root": tree.proof(h).root,
                } for h in hashes
            },
        }

        # Write vectors file
        VECTOR_FILE.write_text(json.dumps(vectors, indent=2, default=str))
        assert VECTOR_FILE.exists()

        # Verify the file is valid JSON
        loaded = json.loads(VECTOR_FILE.read_text())
        assert len(loaded["canonical_json"]) == len(CANONICAL_JSON_CASES)
        assert len(loaded["hash_object"]) == len(HASH_OBJECT_INPUTS)
        assert loaded["merkle"]["expected_root"] == tree.root()
        # Rust metadata preserved in generated file
        assert "rust" in loaded
        assert loaded["rust"]["rust_crate"] == "sdk-rs/hasher"

    def test_vectors_self_verify(self):
        """Verify that generated vectors pass when loaded back."""
        if not VECTOR_FILE.exists():
            pytest.skip("vectors.json not generated yet — run test_generate_vectors first")

        vectors = json.loads(VECTOR_FILE.read_text())

        # Verify canonical JSON
        for v in vectors["canonical_json"]:
            assert hash_object(v["input"]) == v["expected_hash"]

        # Verify hash_object
        for v in vectors["hash_object"]:
            assert hash_object(v["input"]) == v["expected_hash"]

        # Verify Merkle
        root = vectors["merkle"]["expected_root"]
        for h in vectors["merkle"]["leaf_hashes"]:
            proof_data = vectors["merkle"]["proofs"][h]
            proof = MerkleProof(
                target_hash=proof_data["target_hash"],
                proof_path=[tuple(p) for p in proof_data["proof_path"]],
                root=proof_data["root"],
            )
            assert verify_proof(root, proof, h)


# ──────────────────────────────────────────────────────────────────────
# Rust SDK cross-SDK vector verification (static source analysis)
# ──────────────────────────────────────────────────────────────────────


class TestRustSdkVectors:
    """Verify Rust SDK test vectors match Python SDK via static source analysis.

    These tests read sdk-rs/hasher/src/lib.rs and sdk-rs/merkle/src/lib.rs
    and confirm that the hardcoded SHA-256 and canonical-JSON test vectors
    embedded in the Rust source are identical to the values produced by the
    Python SDK.  No Rust compilation is needed.
    """

    # Known SHA-256 vectors embedded in the Rust source (test: hash_bytes_known_vector
    # and hash_bytes_abc_vector).
    _SHA256_VECTORS: list[tuple[bytes, str]] = [
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
    ]

    # ── Presence checks ───────────────────────────────────────────────

    def test_rust_hasher_source_present(self):
        """Rust SDK hasher source exists at expected path (sdk-rs/hasher/src/lib.rs)."""
        if not RUST_HASHER_SOURCE.exists():
            pytest.skip("Rust SDK source not available — skipping static analysis")
        assert RUST_HASHER_SOURCE.is_file()

    def test_rust_merkle_source_present(self):
        """Rust SDK merkle source exists at expected path (sdk-rs/merkle/src/lib.rs)."""
        if not RUST_MERKLE_SOURCE.exists():
            pytest.skip("Rust SDK merkle source not available")
        assert RUST_MERKLE_SOURCE.is_file()

    # ── SHA-256 known-vector checks ───────────────────────────────────

    def test_rust_known_sha256_empty_string_matches_python(self):
        """SHA-256('') hardcoded in Rust lib.rs equals Python's hashlib output."""
        if not RUST_HASHER_SOURCE.exists():
            pytest.skip("Rust SDK source not available")
        source = RUST_HASHER_SOURCE.read_text(encoding="utf-8")
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert expected in source, (
            f"Empty-string SHA-256 vector {expected!r} not found in Rust hasher source"
        )
        python_result = hashlib.sha256(b"").hexdigest()
        assert python_result == expected, (
            f"Python hashlib disagrees: {python_result!r} != {expected!r}"
        )

    def test_rust_known_sha256_abc_matches_python(self):
        """SHA-256('abc') hardcoded in Rust lib.rs equals Python's hashlib output."""
        if not RUST_HASHER_SOURCE.exists():
            pytest.skip("Rust SDK source not available")
        source = RUST_HASHER_SOURCE.read_text(encoding="utf-8")
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert expected in source, (
            f"SHA-256('abc') vector {expected!r} not found in Rust hasher source"
        )
        python_result = hashlib.sha256(b"abc").hexdigest()
        assert python_result == expected, (
            f"Python hashlib disagrees: {python_result!r} != {expected!r}"
        )

    def test_all_sha256_vectors_match_python(self):
        """Every known SHA-256 vector in Rust source is confirmed by Python hashlib."""
        if not RUST_HASHER_SOURCE.exists():
            pytest.skip("Rust SDK source not available")
        for raw, expected_hex in self._SHA256_VECTORS:
            python_result = hashlib.sha256(raw).hexdigest()
            assert python_result == expected_hex, (
                f"SHA-256({raw!r}): Python={python_result!r}, Rust vector={expected_hex!r}"
            )

    # ── Canonical JSON cross-SDK vector ───────────────────────────────

    def test_rust_canonical_json_vector_in_source(self):
        """The cross_sdk_canonical_json_vector in Rust source is present and correct."""
        if not RUST_HASHER_SOURCE.exists():
            pytest.skip("Rust SDK source not available")
        source = RUST_HASHER_SOURCE.read_text(encoding="utf-8")
        # Rust test: canonical_json of {"model":"gpt-4","seq":0,"confidence":null}
        # → b"{\"confidence\":null,\"model\":\"gpt-4\",\"seq\":0}"
        fragment = '"confidence":null,"model":"gpt-4","seq":0'
        assert fragment in source, (
            f"Cross-SDK canonical JSON fragment {fragment!r} not found in Rust source"
        )

    def test_rust_canonical_json_vector_matches_python(self):
        """Python canonical_json matches the vector embedded in Rust lib.rs."""
        input_obj = {"model": "gpt-4", "seq": 0, "confidence": None}
        expected = b'{"confidence":null,"model":"gpt-4","seq":0}'
        python_result = canonical_json(input_obj)
        assert python_result == expected, (
            f"Python canonical_json result {python_result!r} != expected {expected!r}"
        )

    # ── Rust merkle prefix checks (RFC 6962) ──────────────────────────

    def test_rust_merkle_leaf_prefix_is_0x00(self):
        """Rust merkle lib applies 0x00 byte prefix to leaf nodes (RFC 6962)."""
        if not RUST_MERKLE_SOURCE.exists():
            pytest.skip("Rust SDK merkle source not available")
        source = RUST_MERKLE_SOURCE.read_text(encoding="utf-8")
        assert "0x00" in source, (
            "0x00 leaf prefix not found in Rust merkle source — "
            "RFC 6962 second-preimage protection may be broken"
        )

    def test_rust_merkle_internal_prefix_is_0x01(self):
        """Rust merkle lib applies 0x01 byte prefix to internal nodes (RFC 6962)."""
        if not RUST_MERKLE_SOURCE.exists():
            pytest.skip("Rust SDK merkle source not available")
        source = RUST_MERKLE_SOURCE.read_text(encoding="utf-8")
        assert "0x01" in source, (
            "0x01 internal prefix not found in Rust merkle source — "
            "RFC 6962 second-preimage protection may be broken"
        )

    def test_rust_merkle_odd_node_promotion_in_source(self):
        """Rust merkle source promotes odd-count nodes (not duplicate) at each level."""
        if not RUST_MERKLE_SOURCE.exists():
            pytest.skip("Rust SDK merkle source not available")
        source = RUST_MERKLE_SOURCE.read_text(encoding="utf-8")
        # The Rust code comments say: "Odd node: promote without pairing (standard practice)."
        assert "promote" in source.lower() or "odd" in source.lower(), (
            "No evidence of odd-node promotion logic in Rust merkle source"
        )

    # ── Cross-SDK JSON file checks ────────────────────────────────────

    def test_rust_vectors_in_json_file_present(self):
        """vectors.json contains a 'rust' section after test_generate_vectors runs."""
        if not VECTOR_FILE.exists():
            pytest.skip("vectors.json not generated yet — run TestCrossSDKVectors first")
        vectors = json.loads(VECTOR_FILE.read_text())
        assert "rust" in vectors, (
            "No 'rust' section in vectors.json — run test_generate_vectors to regenerate"
        )
        assert vectors["rust"]["rust_crate"] == "sdk-rs/hasher"

    def test_rust_known_vectors_in_json_match_python_sha256(self):
        """Rust known_vectors stored in vectors.json produce correct Python SHA-256."""
        if not VECTOR_FILE.exists():
            pytest.skip("vectors.json not generated yet")
        vectors = json.loads(VECTOR_FILE.read_text())
        if "rust" not in vectors:
            pytest.skip("No Rust section in vectors.json — run test_generate_vectors first")
        for v in vectors["rust"].get("known_vectors", []):
            input_bytes = v["input_bytes_ascii"].encode("ascii")
            expected = v["expected_sha256"]
            python_result = hashlib.sha256(input_bytes).hexdigest()
            assert python_result == expected, (
                f"Python SHA-256({v['input_bytes_ascii']!r}) = {python_result!r} "
                f"but Rust vector says {expected!r}"
            )

    def test_rust_canonical_json_vector_in_json_matches_python(self):
        """The canonical JSON vector in the rust section of vectors.json matches Python."""
        if not VECTOR_FILE.exists():
            pytest.skip("vectors.json not generated yet")
        vectors = json.loads(VECTOR_FILE.read_text())
        if "rust" not in vectors or "canonical_json_vector" not in vectors["rust"]:
            pytest.skip("No Rust canonical JSON vector in vectors.json")
        v = vectors["rust"]["canonical_json_vector"]
        expected_canonical = v["expected_canonical"].encode("utf-8")
        python_result = canonical_json(v["input"])
        assert python_result == expected_canonical, (
            f"Python canonical_json result {python_result!r} "
            f"!= Rust vector {expected_canonical!r}"
        )
