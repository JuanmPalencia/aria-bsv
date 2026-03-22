"""aria.contracts — BSV smart contract primitives for ARIA.

Provides three contract types:

OperatorBondingContract  (bonding.py)
    Operator stake management.  An operator bonds BSV to a known address
    (P2PKH or multisig).  Governance can slash the bond by spending the UTXO
    to a confiscation address.  Used to incentivise honest epoch reporting.

EpochNotarization        (notarization.py)
    2-of-3 multisig P2MS for EPOCH_CLOSE transactions.  Requires 2 of 3
    designated notary operators to co-sign before a close is accepted.
    Provides redundancy against a single operator going offline or misbehaving.

ARIARegistry             (registry.py)
    OP_RETURN-based on-chain registry for ARIA system and model registration.
    Protocol prefix: b"ARIA-REG".  Entries are immutable once mined.
"""

from .bonding import OperatorBondingContract, BondState, BondRecord
from .notarization import EpochNotarization, NotarizationPolicy
from .registry import ARIARegistry, RegistryEntry, RegistryEntryType

__all__ = [
    "OperatorBondingContract",
    "BondState",
    "BondRecord",
    "EpochNotarization",
    "NotarizationPolicy",
    "ARIARegistry",
    "RegistryEntry",
    "RegistryEntryType",
]
