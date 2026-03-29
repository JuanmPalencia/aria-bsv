"""Property-based tests for aria.zk.ezkl_prover.CommitmentProver using Hypothesis.

CommitmentProver generates an HMAC-SHA256 commitment over the canonical JSON of
(model_hash, input_hash, output_hash, record_id), keyed by the epoch nonce.
No EZKL or circuit compilation is involved — it is purely a keyed-hash commitment.

Invariants tested:
    - prove() always returns proof_bytes of exactly 32 bytes (HMAC-SHA256 output).
    - prove() is deterministic: same inputs always produce the same commitment.
    - prove() differs when input_data changes.
    - prove() differs when output_data changes.
    - prove() differs when record_id changes.
    - prove() differs when epoch_id changes (when no nonce is set).
    - prove() differs when the epoch nonce changes.
    - prove() always sets proving_system == "commitment".
    - prove() always sets tier == "commitment".
    - prove() public_inputs always has exactly 3 elements.
    - prove() public_inputs contains the input_hash and output_hash.
    - verify() returns True for any structurally valid commitment proof.
    - verify() returns False when proof.proving_system != "commitment".
    - verify() returns False when proof.model_hash != vk.model_hash.
    - verify() returns False when len(proof_bytes) != 32.
    - verify() returns False when len(public_inputs) != 3.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aria.core.hasher import canonical_json, hash_object
from aria.zk.base import ProvingKey, VerifyingKey, ZKProof
from aria.zk.ezkl_prover import CommitmentProver


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# JSON-safe scalars (no NaN / Infinity which would raise ARIASerializationError)
json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False),
    st.text(max_size=20),
)

# Flat dict with JSON-safe values — used as input_data / output_data
io_dict = st.dictionaries(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
    json_scalar,
    min_size=1,
    max_size=5,
)

# Arbitrary epoch identifiers
epoch_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=50,
)

# Arbitrary record identifiers
record_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=50,
)

# Arbitrary model_hash strings (CommitmentProver treats this as an opaque str)
model_hash = st.text(
    alphabet="0123456789abcdef",
    min_size=64,
    max_size=64,
).map(lambda h: f"sha256:{h}")

# Arbitrary nonces for CommitmentProver(epoch_nonce=...)
nonce_str = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=64),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pk(mh: str) -> ProvingKey:
    """Build a minimal ProvingKey for CommitmentProver (bypasses setup())."""
    pk_bytes = hashlib.sha256(f"test:pk:{mh}".encode()).digest()
    return ProvingKey(pk_bytes=pk_bytes, model_hash=mh, proving_system="commitment")


def _make_vk(mh: str) -> VerifyingKey:
    """Build a minimal VerifyingKey for CommitmentProver (bypasses setup())."""
    vk_bytes = hashlib.sha256(f"test:vk:{mh}".encode()).digest()
    return VerifyingKey(vk_bytes=vk_bytes, model_hash=mh, proving_system="commitment")


def _prove_sync(
    prover: CommitmentProver,
    input_data: dict,
    output_data: dict,
    pk: ProvingKey,
    rec_id: str,
    ep_id: str,
) -> ZKProof:
    """Run CommitmentProver.prove() synchronously (Hypothesis tests are sync)."""
    return asyncio.run(
        prover.prove(
            model_id="test-model",
            input_data=input_data,
            output_data=output_data,
            pk=pk,
            record_id=rec_id,
            epoch_id=ep_id,
        )
    )


def _make_valid_proof(model_hash_str: str, proof_bytes: bytes | None = None) -> ZKProof:
    """Construct a structurally valid commitment ZKProof."""
    pb = proof_bytes if proof_bytes is not None else b"\xab" * 32
    input_hash = hash_object({"x": 1})
    output_hash = hash_object({"y": 2})
    return ZKProof(
        proof_bytes=pb,
        public_inputs=[input_hash, output_hash, model_hash_str],
        proving_system="commitment",
        tier="commitment",
        model_hash=model_hash_str,
        prover_version="commitment-prover-1.0",
        epoch_id="ep-001",
        record_id="rec-001",
    )


# ---------------------------------------------------------------------------
# prove() — output shape invariants
# ---------------------------------------------------------------------------


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=80)
def test_prove_proof_bytes_always_32_bytes(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """CommitmentProver.prove() always returns proof_bytes of exactly 32 bytes."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)
    assert len(proof.proof_bytes) == 32
    assert isinstance(proof.proof_bytes, bytes)


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=80)
def test_prove_proving_system_is_commitment(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """CommitmentProver.prove() always sets proving_system to 'commitment'."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)
    assert proof.proving_system == "commitment"


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=80)
def test_prove_tier_is_commitment(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """CommitmentProver.prove() always sets tier to 'commitment'."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)
    assert proof.tier == "commitment"


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=80)
def test_prove_public_inputs_has_exactly_three_elements(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """CommitmentProver.prove() always produces exactly 3 public_inputs."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)
    assert len(proof.public_inputs) == 3


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=80)
def test_prove_public_inputs_contains_hashes_of_input_and_output(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """public_inputs[0] and [1] are the sha256 hashes of input_data and output_data."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)
    expected_input_hash = hash_object(inp)
    expected_output_hash = hash_object(out)
    assert proof.public_inputs[0] == expected_input_hash
    assert proof.public_inputs[1] == expected_output_hash


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=80)
def test_prove_public_inputs_third_element_is_model_hash(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """public_inputs[2] always equals pk.model_hash."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)
    assert proof.public_inputs[2] == mh


# ---------------------------------------------------------------------------
# prove() — determinism
# ---------------------------------------------------------------------------


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=80)
def test_prove_is_deterministic_same_inputs(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """Same inputs always produce the same commitment (HMAC is deterministic)."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof1 = _prove_sync(prover, inp, out, pk, rec, ep)
    proof2 = _prove_sync(prover, inp, out, pk, rec, ep)
    assert proof1.proof_bytes == proof2.proof_bytes


# ---------------------------------------------------------------------------
# prove() — sensitivity to each input
# ---------------------------------------------------------------------------


@given(
    inp1=io_dict,
    inp2=io_dict,
    out=io_dict,
    mh=model_hash,
    rec=record_id,
    ep=epoch_id,
)
@settings(max_examples=80)
def test_prove_differs_when_input_data_changes(
    inp1: dict, inp2: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """Different input_data always produce a different commitment."""
    assume(hash_object(inp1) != hash_object(inp2))
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof1 = _prove_sync(prover, inp1, out, pk, rec, ep)
    proof2 = _prove_sync(prover, inp2, out, pk, rec, ep)
    assert proof1.proof_bytes != proof2.proof_bytes


@given(
    inp=io_dict,
    out1=io_dict,
    out2=io_dict,
    mh=model_hash,
    rec=record_id,
    ep=epoch_id,
)
@settings(max_examples=80)
def test_prove_differs_when_output_data_changes(
    inp: dict, out1: dict, out2: dict, mh: str, rec: str, ep: str
) -> None:
    """Different output_data always produce a different commitment."""
    assume(hash_object(out1) != hash_object(out2))
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof1 = _prove_sync(prover, inp, out1, pk, rec, ep)
    proof2 = _prove_sync(prover, inp, out2, pk, rec, ep)
    assert proof1.proof_bytes != proof2.proof_bytes


@given(
    inp=io_dict,
    out=io_dict,
    mh=model_hash,
    rec1=record_id,
    rec2=record_id,
    ep=epoch_id,
)
@settings(max_examples=80)
def test_prove_differs_when_record_id_changes(
    inp: dict, out: dict, mh: str, rec1: str, rec2: str, ep: str
) -> None:
    """Different record_id values produce different commitments."""
    assume(rec1 != rec2)
    prover = CommitmentProver()
    pk = _make_pk(mh)
    proof1 = _prove_sync(prover, inp, out, pk, rec1, ep)
    proof2 = _prove_sync(prover, inp, out, pk, rec2, ep)
    assert proof1.proof_bytes != proof2.proof_bytes


@given(
    inp=io_dict,
    out=io_dict,
    mh=model_hash,
    rec=record_id,
    ep1=epoch_id,
    ep2=epoch_id,
)
@settings(max_examples=80)
def test_prove_differs_when_epoch_id_changes_and_no_nonce(
    inp: dict, out: dict, mh: str, rec: str, ep1: str, ep2: str
) -> None:
    """When no nonce is set, different epoch_id values produce different commitments.

    The HMAC key is epoch_id when self._nonce is None, so changing epoch_id
    changes the key and therefore the HMAC output.
    """
    assume(ep1 != ep2)
    prover = CommitmentProver(epoch_nonce=None)
    pk = _make_pk(mh)
    proof1 = _prove_sync(prover, inp, out, pk, rec, ep1)
    proof2 = _prove_sync(prover, inp, out, pk, rec, ep2)
    assert proof1.proof_bytes != proof2.proof_bytes


@given(
    inp=io_dict,
    out=io_dict,
    mh=model_hash,
    rec=record_id,
    ep=epoch_id,
    nonce1=st.text(min_size=1, max_size=50),
    nonce2=st.text(min_size=1, max_size=50),
)
@settings(max_examples=80)
def test_prove_differs_when_nonce_changes(
    inp: dict,
    out: dict,
    mh: str,
    rec: str,
    ep: str,
    nonce1: str,
    nonce2: str,
) -> None:
    """Different epoch nonces produce different commitments."""
    assume(nonce1 != nonce2)
    pk = _make_pk(mh)
    proof1 = _prove_sync(CommitmentProver(epoch_nonce=nonce1), inp, out, pk, rec, ep)
    proof2 = _prove_sync(CommitmentProver(epoch_nonce=nonce2), inp, out, pk, rec, ep)
    assert proof1.proof_bytes != proof2.proof_bytes


# ---------------------------------------------------------------------------
# verify() — structural validation
# ---------------------------------------------------------------------------


@given(mh=model_hash)
@settings(max_examples=80)
def test_verify_passes_for_structurally_valid_proof(mh: str) -> None:
    """verify() returns True for any structurally valid commitment proof."""
    prover = CommitmentProver()
    proof = _make_valid_proof(mh)
    vk = _make_vk(mh)
    assert prover.verify(proof, vk) is True


@given(inp=io_dict, out=io_dict, mh=model_hash, rec=record_id, ep=epoch_id)
@settings(max_examples=60)
def test_verify_passes_for_proof_produced_by_prove(
    inp: dict, out: dict, mh: str, rec: str, ep: str
) -> None:
    """A proof produced by prove() always passes verify() against the matching vk."""
    prover = CommitmentProver()
    pk = _make_pk(mh)
    vk = _make_vk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)
    assert prover.verify(proof, vk) is True


@given(mh1=model_hash, mh2=model_hash)
@settings(max_examples=60)
def test_verify_fails_when_model_hash_mismatches(mh1: str, mh2: str) -> None:
    """verify() returns False when proof.model_hash != vk.model_hash."""
    assume(mh1 != mh2)
    prover = CommitmentProver()
    proof = _make_valid_proof(mh1)
    vk = _make_vk(mh2)  # different model_hash
    assert prover.verify(proof, vk) is False


@given(mh=model_hash, bad_system=st.text(min_size=1, max_size=20))
@settings(max_examples=60)
def test_verify_fails_for_wrong_proving_system(mh: str, bad_system: str) -> None:
    """verify() returns False when proof.proving_system != 'commitment'."""
    assume(bad_system != "commitment")
    prover = CommitmentProver()
    vk = _make_vk(mh)
    proof = ZKProof(
        proof_bytes=b"\x00" * 32,
        public_inputs=[hash_object({"a": 1}), hash_object({"b": 2}), mh],
        proving_system=bad_system,  # wrong system
        tier="commitment",
        model_hash=mh,
        prover_version="commitment-prover-1.0",
        epoch_id="ep",
        record_id="rec",
    )
    assert prover.verify(proof, vk) is False


@given(
    mh=model_hash,
    bad_length=st.one_of(
        st.integers(min_value=0, max_value=31),
        st.integers(min_value=33, max_value=64),
    ),
)
@settings(max_examples=60)
def test_verify_fails_for_wrong_proof_bytes_length(mh: str, bad_length: int) -> None:
    """verify() returns False when len(proof_bytes) != 32."""
    prover = CommitmentProver()
    vk = _make_vk(mh)
    proof = ZKProof(
        proof_bytes=b"\x00" * bad_length,
        public_inputs=[hash_object({"a": 1}), hash_object({"b": 2}), mh],
        proving_system="commitment",
        tier="commitment",
        model_hash=mh,
        prover_version="commitment-prover-1.0",
        epoch_id="ep",
        record_id="rec",
    )
    assert prover.verify(proof, vk) is False


@given(
    mh=model_hash,
    extra=st.lists(st.text(max_size=10), min_size=0, max_size=5),
)
@settings(max_examples=60)
def test_verify_fails_for_wrong_public_inputs_count(
    mh: str, extra: list[str]
) -> None:
    """verify() returns False when public_inputs does not have exactly 3 elements."""
    # Build public_inputs with != 3 elements by removing one or adding extras
    bad_inputs_cases = [
        [],
        ["only_one"],
        ["one", "two"],
        ["one", "two", "three", "four"],  # 4 elements
        extra,
    ]
    prover = CommitmentProver()
    vk = _make_vk(mh)
    for bad_inputs in bad_inputs_cases:
        if len(bad_inputs) == 3:
            continue  # skip the accidental 3-element case
        proof = ZKProof(
            proof_bytes=b"\x00" * 32,
            public_inputs=bad_inputs,
            proving_system="commitment",
            tier="commitment",
            model_hash=mh,
            prover_version="commitment-prover-1.0",
            epoch_id="ep",
            record_id="rec",
        )
        assert prover.verify(proof, vk) is False, (
            f"Expected False for public_inputs={bad_inputs!r}"
        )


# ---------------------------------------------------------------------------
# CommitmentProver properties
# ---------------------------------------------------------------------------


def test_commitment_prover_proving_system_property() -> None:
    """CommitmentProver.proving_system property always returns 'commitment'."""
    assert CommitmentProver().proving_system == "commitment"


def test_commitment_prover_tier_property() -> None:
    """CommitmentProver.tier property always returns 'commitment'."""
    assert CommitmentProver().tier == "commitment"


def test_commitment_prover_max_model_params_is_none() -> None:
    """CommitmentProver.max_model_params is None (no model size limit)."""
    assert CommitmentProver().max_model_params is None


@given(
    inp=io_dict,
    out=io_dict,
    mh=model_hash,
    rec=record_id,
    ep=epoch_id,
    nonce=st.text(min_size=1, max_size=50),
)
@settings(max_examples=60)
def test_prove_commitment_matches_manual_hmac_computation(
    inp: dict, out: dict, mh: str, rec: str, ep: str, nonce: str
) -> None:
    """The commitment equals HMAC-SHA256(key=nonce, msg=canonical_json({...})).

    This test independently recomputes the HMAC to verify CommitmentProver
    follows the documented construction in BRC-121.
    """
    prover = CommitmentProver(epoch_nonce=nonce)
    pk = _make_pk(mh)
    proof = _prove_sync(prover, inp, out, pk, rec, ep)

    # Reproduce the commitment manually
    input_hash = hash_object(inp)
    output_hash = hash_object(out)
    key = nonce.encode("utf-8")
    msg = canonical_json({
        "model_hash": mh,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "record_id": rec,
    })
    expected = hmac.new(key, msg, hashlib.sha256).digest()

    assert proof.proof_bytes == expected, (
        f"Commitment mismatch — prover produced {proof.proof_bytes.hex()!r}, "
        f"expected {expected.hex()!r}"
    )
