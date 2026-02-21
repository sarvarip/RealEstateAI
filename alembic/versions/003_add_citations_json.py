"""Add citations_json column to messages table

Revision ID: 003_citations
Revises: 002_chunks
Create Date: 2026-02-21 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003_citations"
down_revision: str | None = "002_chunks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("citations_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "citations_json")
