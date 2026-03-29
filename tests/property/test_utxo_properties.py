"""Property-based tests for aria.wallet.utxo using Hypothesis.

Invariants:
    - select_utxos never over-spends (change ≥ 0).
    - select_utxos always covers the target + fee.
    - total_in = sum(inputs.satoshis).
    - Increasing the fee rate never decreases the fee.
    - If total available < target + min_fee, always raises.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aria.core.errors import ARIAWalletError
from aria.wallet.utxo import UTXO, CoinSelection, estimate_tx_bytes, select_utxos

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A single UTXO value (satoshis): 546 (dust limit) to 10 BSV
utxo_value = st.integers(min_value=546, max_value=1_000_000_000)

# List of UTXO values
utxo_list = st.lists(utxo_value, min_size=1, max_size=20)

# Fee rate: 0.1 to 10 sat/byte
fee_rate = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)

# Target amount: 1 sat to 1 BSV
target = st.integers(min_value=1, max_value=100_000_000)


def _make_utxos(values: list[int]) -> list[UTXO]:
    return [
        UTXO(txid="a" * 62 + f"{i:02d}", vout=0, satoshis=v, script_pubkey="76a914")
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _can_cover(values: list[int], target: int, fee_rate: float) -> bool:
    """Return True if the UTXO set can definitely cover target + worst-case fee."""
    # Worst-case fee: all UTXOs used as inputs
    max_fee = int(estimate_tx_bytes(len(values), 2) * fee_rate) + 1
    return sum(values) >= target + max_fee


@given(values=utxo_list, fee_rate_=fee_rate)
@settings(max_examples=200)
def test_selection_is_sufficient(values: list[int], fee_rate_: float):
    """When selection succeeds, total_in >= target + fee."""
    safe_target = 500
    assume(_can_cover(values, safe_target, fee_rate_))

    utxos = _make_utxos(values)
    sel = select_utxos(utxos, target_satoshis=safe_target, fee_rate=fee_rate_)
    assert sel.total_in >= safe_target + sel.fee_satoshis


@given(values=utxo_list, fee_rate_=fee_rate)
@settings(max_examples=200)
def test_change_non_negative(values: list[int], fee_rate_: float):
    """When selection succeeds, change_satoshis >= 0."""
    safe_target = 500
    assume(_can_cover(values, safe_target, fee_rate_))

    utxos = _make_utxos(values)
    sel = select_utxos(utxos, target_satoshis=safe_target, fee_rate=fee_rate_)
    assert sel.change_satoshis >= 0


@given(values=utxo_list, fee_rate_=fee_rate)
@settings(max_examples=200)
def test_total_in_equals_sum_of_inputs(values: list[int], fee_rate_: float):
    """total_in == sum(input.satoshis) for every selection."""
    safe_target = 500
    assume(_can_cover(values, safe_target, fee_rate_))

    utxos = _make_utxos(values)
    sel = select_utxos(utxos, target_satoshis=safe_target, fee_rate=fee_rate_)
    assert sel.total_in == sum(u.satoshis for u in sel.inputs)


@given(values=utxo_list)
@settings(max_examples=100)
def test_higher_fee_rate_same_or_higher_fee(values: list[int]):
    """For the same inputs, a higher fee rate produces a >= fee."""
    safe_target = 500
    assume(_can_cover(values, safe_target, 5.0))  # ensure high rate also works

    utxos = _make_utxos(values)
    sel_low = select_utxos(utxos, target_satoshis=safe_target, fee_rate=1.0)
    sel_high = select_utxos(utxos, target_satoshis=safe_target, fee_rate=5.0)
    assert sel_high.fee_satoshis >= sel_low.fee_satoshis


@given(target_=st.integers(min_value=1, max_value=100_000))
def test_insufficient_funds_raises(target_: int):
    """When available < target + fee, AriaWalletError is always raised."""
    # All UTXOs sum to 100 satoshis — target can be set above that
    utxos = _make_utxos([10, 20, 30, 40])
    assume(target_ > 90)  # definitely more than available after fees

    with pytest.raises(ARIAWalletError, match="Insufficient"):
        select_utxos(utxos, target_satoshis=target_, fee_rate=1.0)


@given(n_inputs=st.integers(min_value=0, max_value=50), n_outputs=st.integers(min_value=0, max_value=20))
def test_estimate_tx_bytes_monotone(n_inputs: int, n_outputs: int):
    """Adding one more input always increases the estimated size by 148 bytes."""
    size1 = estimate_tx_bytes(n_inputs, n_outputs)
    size2 = estimate_tx_bytes(n_inputs + 1, n_outputs)
    assert size2 - size1 == 148


@given(n_inputs=st.integers(min_value=0, max_value=50), n_outputs=st.integers(min_value=0, max_value=20))
def test_estimate_tx_bytes_output_monotone(n_inputs: int, n_outputs: int):
    """Adding one more output always increases the estimated size by 34 bytes."""
    size1 = estimate_tx_bytes(n_inputs, n_outputs)
    size2 = estimate_tx_bytes(n_inputs, n_outputs + 1)
    assert size2 - size1 == 34
