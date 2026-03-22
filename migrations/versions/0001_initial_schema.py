"""Initial ARIA database schema.

Creates the four core tables:
  - aria_epochs       — epoch lifecycle (open/close txids, state hash, Merkle root)
  - aria_records      — individual inference audit records
  - aria_zk_proofs    — ZK proof storage for ZK-tier records
  - aria_vk_keys      — verifying keys indexed by model hash

Revision ID: 0001
Revises: (none)
Create Date: 2026-03-22
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "aria_epochs",
        sa.Column("epoch_id", sa.String(), nullable=False),
        sa.Column("system_id", sa.String(), nullable=False),
        sa.Column("open_txid", sa.String(), nullable=False, server_default=""),
        sa.Column("close_txid", sa.String(), nullable=False, server_default=""),
        sa.Column("state_hash", sa.String(), nullable=False, server_default=""),
        sa.Column("model_hashes_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("opened_at", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("closed_at", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("merkle_root", sa.String(), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("epoch_id"),
    )
    op.create_index("ix_aria_epochs_system_id", "aria_epochs", ["system_id"])
    op.create_index("ix_aria_epochs_opened_at", "aria_epochs", ["opened_at"])

    op.create_table(
        "aria_records",
        sa.Column("record_id", sa.String(), nullable=False),
        sa.Column("epoch_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("output_hash", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("aria_version", sa.String(), nullable=False),
        sa.Column("record_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("record_id"),
    )
    op.create_index("ix_aria_records_epoch_id", "aria_records", ["epoch_id"])
    op.create_index("ix_aria_records_model_id", "aria_records", ["model_id"])
    op.create_index(
        "ix_aria_records_epoch_seq", "aria_records", ["epoch_id", "sequence"]
    )

    op.create_table(
        "aria_zk_proofs",
        sa.Column("record_id", sa.String(), nullable=False),
        sa.Column("epoch_id", sa.String(), nullable=False),
        sa.Column("proof_hex", sa.Text(), nullable=False),
        sa.Column("public_inputs_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("proving_system", sa.String(), nullable=False),
        sa.Column("tier", sa.String(), nullable=False),
        sa.Column("model_hash", sa.String(), nullable=False),
        sa.Column("prover_version", sa.String(), nullable=False),
        sa.Column("proof_digest", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("record_id"),
    )
    op.create_index("ix_aria_zk_proofs_epoch_id", "aria_zk_proofs", ["epoch_id"])

    op.create_table(
        "aria_vk_keys",
        sa.Column("model_hash", sa.String(), nullable=False),
        sa.Column("vk_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("proving_system", sa.String(), nullable=False),
        sa.Column("vk_digest", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("model_hash"),
    )


def downgrade() -> None:
    op.drop_table("aria_vk_keys")
    op.drop_index("ix_aria_zk_proofs_epoch_id", table_name="aria_zk_proofs")
    op.drop_table("aria_zk_proofs")
    op.drop_index("ix_aria_records_epoch_seq", table_name="aria_records")
    op.drop_index("ix_aria_records_model_id", table_name="aria_records")
    op.drop_index("ix_aria_records_epoch_id", table_name="aria_records")
    op.drop_table("aria_records")
    op.drop_index("ix_aria_epochs_opened_at", table_name="aria_epochs")
    op.drop_index("ix_aria_epochs_system_id", table_name="aria_epochs")
    op.drop_table("aria_epochs")
