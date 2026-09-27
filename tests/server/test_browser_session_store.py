"""Tests for :class:`BrowserSessionRevocationStore` (durable logout record)."""

from __future__ import annotations

import time

import pytest
from sqlalchemy import select

from omnigent.db.db_models import SqlBrowserSessionRevocation
from omnigent.server.browser_session_store import BrowserSessionRevocationStore


@pytest.fixture
def store(db_uri: str) -> BrowserSessionRevocationStore:
    return BrowserSessionRevocationStore(db_uri)


def _sids(store: BrowserSessionRevocationStore) -> set[str]:
    with store._session("test_list_browser_session_revocations") as session:
        return set(session.scalars(select(SqlBrowserSessionRevocation.sid)))


def test_revoked_sid_is_reported_until_it_lapses(store: BrowserSessionRevocationStore) -> None:
    now = int(time.time())
    store.revoke("sid-a", user_id="alice", expires_at=now + 3600)

    assert store.is_revoked("sid-a")
    assert not store.is_revoked("sid-unknown")


def test_revocation_is_shared_across_store_instances(db_uri: str) -> None:
    """A second store on the same database (another replica, or a restart) sees it."""
    BrowserSessionRevocationStore(db_uri).revoke(
        "sid-a", user_id="alice", expires_at=int(time.time()) + 3600
    )

    assert BrowserSessionRevocationStore(db_uri).is_revoked("sid-a")


def test_revoke_is_idempotent(store: BrowserSessionRevocationStore) -> None:
    expires_at = int(time.time()) + 3600
    store.revoke("sid-a", user_id="alice", expires_at=expires_at)
    store.revoke("sid-a", user_id="alice", expires_at=expires_at)

    assert _sids(store) == {"sid-a"}


def test_lapsed_rows_are_ignored_and_purged(
    store: BrowserSessionRevocationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows past the session's absolute expiry stop counting and are deleted."""
    now = int(time.time())
    store.revoke("sid-old", user_id="alice", expires_at=now + 10)
    monkeypatch.setattr(time, "time", lambda: now + 11)

    assert not store.is_revoked("sid-old")
    store.revoke("sid-new", user_id="bob", expires_at=now + 3600)
    assert _sids(store) == {"sid-new"}


def test_already_lapsed_session_is_not_recorded(store: BrowserSessionRevocationStore) -> None:
    store.revoke("sid-a", user_id="alice", expires_at=int(time.time()) - 1)

    assert _sids(store) == set()


def test_overlong_sid_is_never_stored(store: BrowserSessionRevocationStore) -> None:
    sid = "x" * 65
    store.revoke(sid, user_id="alice", expires_at=int(time.time()) + 3600)

    assert not store.is_revoked(sid)
    assert _sids(store) == set()
