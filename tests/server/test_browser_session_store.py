"""Tests for :class:`BrowserSessionRevocationStore` (durable logout record)."""

from __future__ import annotations

import time

import pytest
from sqlalchemy import select

from omnigent.db.db_models import SqlBrowserSessionRevocation, workspace_scope
from omnigent.server.browser_session_store import BrowserSessionRevocationStore


@pytest.fixture
def store(db_uri: str) -> BrowserSessionRevocationStore:
    return BrowserSessionRevocationStore(db_uri)


def _sids(store: BrowserSessionRevocationStore) -> set[str]:
    with store._session("test_list_browser_session_revocations") as session:
        return set(session.scalars(select(SqlBrowserSessionRevocation.sid)))


def _rows(store: BrowserSessionRevocationStore) -> set[tuple[int, str]]:
    """Every row's (workspace_id, sid), across all workspaces."""
    with store._session("test_list_all_browser_session_revocations") as session:
        rows = session.execute(
            select(SqlBrowserSessionRevocation.workspace_id, SqlBrowserSessionRevocation.sid)
        )
        return {(workspace_id, sid) for workspace_id, sid in rows}


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


def test_lookup_is_scoped_to_the_workspace(store: BrowserSessionRevocationStore) -> None:
    """A logout in one workspace does not end a same-sid session in another."""
    expires_at = int(time.time()) + 3600
    with workspace_scope(1):
        store.revoke("sid-a", user_id="alice", expires_at=expires_at)

    with workspace_scope(1):
        assert store.is_revoked("sid-a")
    with workspace_scope(2):
        assert not store.is_revoked("sid-a")
    assert not store.is_revoked("sid-a")  # default workspace 0


def test_pruning_is_scoped_to_the_workspace(
    store: BrowserSessionRevocationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A logout purges lapsed rows only in its own workspace."""
    now = int(time.time())
    for workspace_id in (1, 2):
        with workspace_scope(workspace_id):
            store.revoke("sid-old", user_id="alice", expires_at=now + 10)
    monkeypatch.setattr(time, "time", lambda: now + 11)

    with workspace_scope(1):
        store.revoke("sid-new", user_id="bob", expires_at=now + 3600)

    assert _rows(store) == {(1, "sid-new"), (2, "sid-old")}
    with workspace_scope(2):
        store.revoke("sid-other", user_id="carol", expires_at=now + 3600)
    assert _rows(store) == {(1, "sid-new"), (2, "sid-other")}
