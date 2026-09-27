"""Tests for the browser_session_revocations migration (mm1a2b3c4d5e).

Verifies the table's shape (``workspace_id`` leading the primary key, the
purge index, no foreign keys) and that a downgrade drops it cleanly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.engine import Engine

from omnigent.db.utils import (
    _build_alembic_config,
    clear_engine_cache,
    get_or_create_engine,
)

_TABLE = "browser_session_revocations"
_PREVIOUS_HEAD = "ll1a2b3c4d5e"


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    """Fresh SQLite DB with the full migration chain applied; cleaned up after."""
    engine = get_or_create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    try:
        yield engine
    finally:
        clear_engine_cache()


def test_table_shape(db_engine: Engine) -> None:
    """Columns, primary key, purge index and no foreign keys."""
    inspector = sa.inspect(db_engine)
    columns = {c["name"]: c for c in inspector.get_columns(_TABLE)}
    assert set(columns) == {"workspace_id", "sid", "user_id", "revoked_at", "expires_at"}
    assert not any(c["nullable"] for c in columns.values())
    assert inspector.get_pk_constraint(_TABLE)["constrained_columns"] == ["workspace_id", "sid"]
    indexes = {i["name"]: i["column_names"] for i in inspector.get_indexes(_TABLE)}
    assert indexes["ix_browser_session_revocations_expires_at"] == [
        "workspace_id",
        "expires_at",
        "sid",
    ]
    assert inspector.get_foreign_keys(_TABLE) == []


def test_downgrade_drops_table(tmp_path: Path) -> None:
    """Downgrading one step removes the table; re-upgrade restores it."""
    uri = f"sqlite:///{tmp_path / 'downgrade.db'}"
    engine = get_or_create_engine(uri)
    try:
        assert _TABLE in sa.inspect(engine).get_table_names()
        config = _build_alembic_config(uri)
        with engine.begin() as conn:
            config.attributes["connection"] = conn
            command.downgrade(config, _PREVIOUS_HEAD)
        assert _TABLE not in sa.inspect(engine).get_table_names()
        with engine.begin() as conn:
            config.attributes["connection"] = conn
            command.upgrade(config, "mm1a2b3c4d5e")
        assert _TABLE in sa.inspect(engine).get_table_names()
    finally:
        engine.dispose()
        clear_engine_cache()
