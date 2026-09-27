"""add browser_session_revocations table

Revision ID: mm1a2b3c4d5e
Revises: ll1a2b3c4d5e
Create Date: 2026-09-27 00:00:00.000000

Adds ``browser_session_revocations``: one row per browser session (``sid``)
ended by logout, kept until the session's absolute expiry so no replica, and
no restart, can accept or renew a cookie from that session again.

The table is brand-new, so it carries the tenant-partition ``workspace_id``
column as the leading primary-key member like every other table, and no
foreign keys (schema Rule R032).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "mm1a2b3c4d5e"
down_revision: str | None = "ll1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``browser_session_revocations`` table."""
    op.create_table(
        "browser_session_revocations",
        sa.Column("workspace_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("sid", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("revoked_at", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "sid"),
    )
    op.create_index(
        "ix_browser_session_revocations_expires_at",
        "browser_session_revocations",
        ["workspace_id", "expires_at", "sid"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the ``browser_session_revocations`` table."""
    op.drop_index(
        "ix_browser_session_revocations_expires_at",
        table_name="browser_session_revocations",
    )
    op.drop_table("browser_session_revocations")
