"""Sliding renewal of browser session cookies in accounts and OIDC modes.

A session cookie used after half of its idle window
(``*_SESSION_TTL_HOURS``) is re-minted on the response so an active user
is not signed out mid-use, bounded by the absolute lifetime
(``*_SESSION_MAX_LIFETIME_HOURS``) measured from the login's
``auth_time``. Bearer callers, delegated/machine tokens and legacy
cookies without ``auth_time`` keep their fixed expiry, and logout ends
the whole renewal chain.

Most tests drive :class:`SessionRenewalMiddleware` over a minimal app
whose only route authenticates through the real provider; the login /
logout / attribute tests go through the real accounts and OIDC routers.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, Response

from omnigent.server.accounts_config import AccountsConfig
from omnigent.server.admin_list import AdminList
from omnigent.server.auth import (
    _SESSION_RENEWAL_KEY,
    SessionRenewalMiddleware,
    UnifiedAuthProvider,
)
from omnigent.server.browser_session_store import BrowserSessionRevocationStore
from omnigent.server.oidc import OIDCConfig, hmac_digest, mint_session_cookie
from tests.server.test_accounts import _build_accounts_app, _login

_SECRET = bytes.fromhex("5e" * 32)
_TTL_HOURS = 8
_TTL = _TTL_HOURS * 3600
_USER = "alice@example.com"
_HTTP = "http://localhost:8000"
_HTTPS = "https://omnigent.example.com"
# Delegated (``scope``) tokens are confined to an allowlist, so the probe
# lives under an allowed prefix; plain session tokens are accepted anywhere.
_PROBE = "/v1/sessions/probe"

_MODES = [
    pytest.param("accounts", _HTTP, id="accounts-http"),
    pytest.param("accounts", _HTTPS, id="accounts-https"),
    pytest.param("oidc", _HTTP, id="oidc-http"),
    pytest.param("oidc", _HTTPS, id="oidc-https"),
]


@pytest.fixture(autouse=True)
def _clear_ambient_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's exported OIDC / max-lifetime vars out of these tests."""
    for name in (
        "OMNIGENT_OIDC_ISSUER",
        "OMNIGENT_ACCOUNTS_SESSION_MAX_LIFETIME_HOURS",
        "OMNIGENT_OIDC_SESSION_MAX_LIFETIME_HOURS",
    ):
        monkeypatch.delenv(name, raising=False)


# ── Helpers ───────────────────────────────────────────────────────


def _config(source: str, origin: str, *, max_hours: int = 720) -> AccountsConfig | OIDCConfig:
    if source == "accounts":
        return AccountsConfig(
            cookie_secret=_SECRET,
            session_ttl_hours=_TTL_HOURS,
            base_url=origin,
            init_admin_password=None,
            invite_ttl_seconds=3600,
            magic_ttl_seconds=600,
            session_max_lifetime_hours=max_hours,
        )
    return OIDCConfig(
        issuer="https://idp.example.com",
        client_id="cid",
        client_secret="client-secret",
        redirect_uri=f"{origin}/auth/callback",
        cookie_secret=_SECRET,
        scopes="openid email profile",
        session_ttl_hours=_TTL_HOURS,
        logout_redirect_uri=None,
        allowed_domains=None,
        provider_type="oidc",
        authorization_endpoint="https://idp.example.com/authorize",
        token_endpoint="https://idp.example.com/token",
        jwks_uri="https://idp.example.com/jwks",
        userinfo_endpoint=None,
        allow_invites=False,
        session_max_lifetime_hours=max_hours,
    )


def _provider(source: str, origin: str, *, max_hours: int = 720) -> UnifiedAuthProvider:
    config = _config(source, origin, max_hours=max_hours)
    if isinstance(config, AccountsConfig):
        return UnifiedAuthProvider(source="accounts", accounts_config=config)
    return UnifiedAuthProvider(source="oidc", oidc_config=config)


def _cookie_name(origin: str) -> str:
    return "__Host-ap_session" if origin.startswith("https://") else "ap_session"


def _session_jwt(
    *,
    remaining: int,
    logged_in_ago: int = 3600,
    secret: bytes = _SECRET,
    **claims: Any,
) -> str:
    """Sign a browser session JWT as it looks *logged_in_ago* s after login.

    ``remaining`` is the seconds left before ``exp``. Keyword claims
    override the defaults; a ``None`` value drops the claim (e.g.
    ``auth_time=None`` for a legacy cookie).
    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": _USER,
        "iat": now + remaining - _TTL,
        "exp": now + remaining,
        "provider": "accounts",
        "account_generation": "gen-1",
        "auth_time": now - logged_in_ago,
        "sid": "sid-abc",
    }
    payload.update(claims)
    payload = {key: value for key, value in payload.items() if value is not None}
    return jwt.encode(payload, secret, algorithm="HS256")


def _decode(token: str, secret: bytes = _SECRET) -> dict[str, Any]:
    return jwt.decode(token, secret, algorithms=["HS256"])


def _session_set_cookies(resp: httpx.Response, cookie_name: str) -> list[str]:
    return [h for h in resp.headers.get_list("set-cookie") if h.startswith(f"{cookie_name}=")]


def _parse_set_cookie(header: str) -> tuple[str, str, dict[str, str | bool]]:
    """Split a ``Set-Cookie`` header into name, value and lowercased attributes."""
    first, *attrs = (part.strip() for part in header.split(";"))
    name, _, value = first.partition("=")
    parsed: dict[str, str | bool] = {}
    for attr in attrs:
        key, sep, attr_value = attr.partition("=")
        parsed[key.lower()] = attr_value.lower() if sep else True
    return name, value, parsed


def _renewed_token(resp: httpx.Response, cookie_name: str) -> str:
    headers = _session_set_cookies(resp, cookie_name)
    assert len(headers) == 1, resp.headers.get_list("set-cookie")
    _, value, _ = _parse_set_cookie(headers[0])
    return value


def _cookie(cookie_name: str, token: str) -> dict[str, str]:
    return {"Cookie": f"{cookie_name}={token}"}


def _probe_app(provider: UnifiedAuthProvider) -> FastAPI:
    """A minimal app whose routes authenticate through *provider*."""
    app = FastAPI()

    @app.get(_PROBE)
    def probe(request: Request) -> Response:
        user = provider.get_user_id(request)
        if user is None:
            return JSONResponse(status_code=401, content={"error": "not authenticated"})
        return JSONResponse(content={"user": user})

    @app.get("/v1/sessions/missing")
    def missing(request: Request) -> Response:
        assert provider.get_user_id(request) is not None
        return JSONResponse(status_code=404, content={"error": "not found"})

    @app.get("/v1/sessions/logout-race")
    def logout_race(request: Request) -> Response:
        # Authenticate (marking the cookie for renewal), then let a logout of
        # the same session land before the response starts.
        assert provider.get_user_id(request) is not None
        provider.end_browser_session(request)
        return JSONResponse(content={"ok": True})

    app.add_middleware(SessionRenewalMiddleware, auth_provider=provider)
    return app


@pytest.fixture
def probe() -> Iterator[Any]:
    """Factory: ``probe(source, origin, max_hours=...)`` -> (client, provider)."""
    clients: list[TestClient] = []

    def _make(
        source: str, origin: str, *, max_hours: int = 720
    ) -> tuple[TestClient, UnifiedAuthProvider]:
        provider = _provider(source, origin, max_hours=max_hours)
        client = TestClient(_probe_app(provider), base_url=origin)
        clients.append(client)
        return client, provider

    yield _make
    for client in clients:
        client.close()


class _Conn:
    """HTTPConnection stand-in for provider-level checks."""

    def __init__(self, *, cookies: dict[str, str], scope_type: str = "http") -> None:
        self.cookies = cookies
        self.headers: dict[str, str] = {}
        self.scope: dict[str, Any] = {"type": scope_type}
        self.url = httpx.URL(f"http://testserver{_PROBE}")


# ── Contract 1: renewal past the halfway point only ───────────────


@pytest.mark.parametrize(("source", "origin"), _MODES)
def test_cookie_past_halfway_is_renewed(probe: Any, source: str, origin: str) -> None:
    """A cookie with under half its idle window left gets a fresh JWT.

    The renewed JWT carries a full idle window from now and keeps the
    identity claims and the original login time / session id.
    """
    client, _ = probe(source, origin)
    name = _cookie_name(origin)
    old = _session_jwt(remaining=_TTL // 2 - 60, provider=source)

    resp = client.get(_PROBE, headers=_cookie(name, old))

    assert resp.status_code == 200, resp.text
    renewed = _decode(_renewed_token(resp, name))
    before = _decode(old)
    now = int(time.time())
    for claim in ("sub", "provider", "account_generation", "auth_time", "sid"):
        assert renewed[claim] == before[claim], claim
    assert abs(renewed["exp"] - (now + _TTL)) <= 5
    assert renewed["iat"] >= before["iat"] + _TTL // 2
    assert renewed["exp"] <= renewed["auth_time"] + 720 * 3600


@pytest.mark.parametrize(("source", "origin"), _MODES)
def test_cookie_before_halfway_is_not_renewed(probe: Any, source: str, origin: str) -> None:
    """A cookie still in the first half of its idle window is left alone."""
    client, _ = probe(source, origin)
    name = _cookie_name(origin)

    resp = client.get(_PROBE, headers=_cookie(name, _session_jwt(remaining=_TTL // 2 + 60)))

    assert resp.status_code == 200, resp.text
    assert _session_set_cookies(resp, name) == []


def test_error_response_is_not_renewed(probe: Any) -> None:
    """Renewal rides on successful responses only."""
    client, _ = probe("accounts", _HTTP)
    name = _cookie_name(_HTTP)

    resp = client.get("/v1/sessions/missing", headers=_cookie(name, _session_jwt(remaining=600)))

    assert resp.status_code == 404
    assert _session_set_cookies(resp, name) == []


# ── Contract 2/3: absolute lifetime ──────────────────────────────


@pytest.mark.parametrize(("source", "origin"), _MODES)
def test_renewal_is_capped_at_absolute_lifetime(probe: Any, source: str, origin: str) -> None:
    """Near the absolute limit, renewal extends only up to auth_time + max."""
    client, _ = probe(source, origin, max_hours=10)
    name = _cookie_name(origin)
    old = _session_jwt(remaining=1800, logged_in_ago=int(8.5 * 3600))

    resp = client.get(_PROBE, headers=_cookie(name, old))

    assert resp.status_code == 200, resp.text
    header = _session_set_cookies(resp, name)[0]
    renewed = _decode(_parse_set_cookie(header)[1])
    assert renewed["exp"] == _decode(old)["auth_time"] + 10 * 3600
    max_age = int(str(_parse_set_cookie(header)[2]["max-age"]))
    assert abs(max_age - (renewed["exp"] - int(time.time()))) <= 5


def test_session_pinned_at_absolute_lifetime_is_not_renewed(probe: Any) -> None:
    """A cookie whose exp already equals auth_time + max is not re-minted."""
    client, _ = probe("oidc", _HTTP, max_hours=10)
    name = _cookie_name(_HTTP)
    # auth_time = now - 9.5h, exp = now + 0.5h = auth_time + 10h.
    token = _session_jwt(remaining=1800, logged_in_ago=int(9.5 * 3600))

    resp = client.get(_PROBE, headers=_cookie(name, token))

    assert resp.status_code == 200, resp.text
    assert _session_set_cookies(resp, name) == []


@pytest.mark.parametrize(("source", "origin"), _MODES)
def test_past_absolute_lifetime_is_unauthenticated(probe: Any, source: str, origin: str) -> None:
    """Past auth_time + max the cookie is rejected outright and not renewed.

    The token's own ``exp`` is still in the future (e.g. the operator
    lowered the max lifetime), so only the absolute check rejects it.
    """
    client, _ = probe(source, origin, max_hours=10)
    name = _cookie_name(origin)
    token = _session_jwt(remaining=3600, logged_in_ago=10 * 3600 + 60)

    resp = client.get(_PROBE, headers=_cookie(name, token))

    assert resp.status_code == 401
    assert _session_set_cookies(resp, name) == []


# ── Contract 4: only plain cookie sessions renew ─────────────────


@pytest.mark.parametrize(("source", "origin"), _MODES)
def test_bearer_request_is_never_renewed(probe: Any, source: str, origin: str) -> None:
    """The same aged JWT presented as a Bearer authenticates but never slides."""
    client, _ = probe(source, origin)

    resp = client.get(_PROBE, headers={"Authorization": f"Bearer {_session_jwt(remaining=600)}"})

    assert resp.status_code == 200, resp.text
    assert resp.headers.get_list("set-cookie") == []


@pytest.mark.parametrize(
    "delegation",
    [
        pytest.param({"grant_id": "grant-1"}, id="grant_id"),
        pytest.param({"scope": "sessions"}, id="scope"),
        pytest.param({"grant_id": "grant-1", "scope": "sessions"}, id="grant_id+scope"),
    ],
)
@pytest.mark.parametrize("source", ["accounts", "oidc"])
def test_delegated_token_in_cookie_is_never_renewed(
    probe: Any, source: str, delegation: dict[str, str]
) -> None:
    """A delegated / machine token is never renewed, even from the cookie."""
    client, _ = probe(source, _HTTP)
    name = _cookie_name(_HTTP)

    resp = client.get(_PROBE, headers=_cookie(name, _session_jwt(remaining=600, **delegation)))

    assert resp.status_code == 200, resp.text
    assert _session_set_cookies(resp, name) == []


@pytest.mark.parametrize("source", ["accounts", "oidc"])
def test_legacy_cookie_without_auth_time_is_honoured_but_not_renewed(
    probe: Any, source: str
) -> None:
    """Pre-renewal cookies keep working until their exp, then lapse."""
    client, _ = probe(source, _HTTP)
    name = _cookie_name(_HTTP)
    legacy = _session_jwt(remaining=600, auth_time=None, sid=None)

    resp = client.get(_PROBE, headers=_cookie(name, legacy))

    assert resp.status_code == 200, resp.text
    assert _session_set_cookies(resp, name) == []


def test_websocket_handshake_is_never_marked_for_renewal() -> None:
    """WebSocket handshakes authenticate but are never renewed."""
    provider = _provider("accounts", _HTTP)
    token = _session_jwt(remaining=600)
    conn = _Conn(cookies={"ap_session": token}, scope_type="websocket")

    assert provider.get_user_id(conn) == _USER  # type: ignore[arg-type]
    assert _SESSION_RENEWAL_KEY not in conn.scope


# ── Contract 5: renewed cookie matches login's attributes ────────


def _attributes(header: str) -> tuple[str, dict[str, str | bool]]:
    name, _, attrs = _parse_set_cookie(header)
    assert "max-age" in attrs, header
    return name, {k: v for k, v in attrs.items() if k not in ("max-age", "expires")}


@pytest.mark.parametrize("origin", [_HTTP, _HTTPS])
def test_accounts_renewed_cookie_matches_login_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    """Accounts mode: the renewed cookie is attribute-identical to login's."""
    for client in _build_accounts_app(
        tmp_path, monkeypatch, init_admin_password="admin-pw-12345", base_url=origin
    ):
        name = _cookie_name(origin)
        login = client.post(
            "/auth/login", json={"username": "admin", "password": "admin-pw-12345"}
        )
        assert login.status_code == 200, login.text
        login_header = _session_set_cookies(login, name)[0]
        secret = bytes.fromhex(os.environ["OMNIGENT_ACCOUNTS_COOKIE_SECRET"])
        login_claims = _decode(_parse_set_cookie(login_header)[1], secret)

        aged = _session_jwt(
            remaining=600,
            secret=secret,
            sub=login_claims["sub"],
            account_generation=login_claims["account_generation"],
            sid=login_claims["sid"],
        )
        resp = client.get("/auth/me", headers=_cookie(name, aged))

        assert resp.status_code == 200, resp.text
        renewed_header = _session_set_cookies(resp, name)[0]
        assert _attributes(renewed_header) == _attributes(login_header)
        login_name, login_attrs = _attributes(login_header)
        assert login_name == name
        assert login_attrs.get("secure", False) is origin.startswith("https://")
        assert login_attrs["httponly"] is True
        assert login_attrs["samesite"] == "lax"
        assert login_attrs["path"] == "/"
        # The renewed session still authenticates through account checks.
        renewed = _parse_set_cookie(renewed_header)[1]
        again = client.get("/auth/me", headers=_cookie(name, renewed))
        assert again.status_code == 200, again.text
        assert again.json()["id"] == "admin"


def _oidc_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, origin: str
) -> tuple[TestClient, UnifiedAuthProvider]:
    """The real OIDC router + a probe route, with the IdP exchange stubbed."""
    import omnigent.server.routes.auth as oidc_routes

    monkeypatch.setenv("HOME", str(tmp_path))
    provider = _provider("oidc", origin)

    async def _fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"id_token": "stub"})

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    monkeypatch.setattr(oidc_routes, "_resolve_oidc_email", lambda token_json, config: _USER)
    admins = tmp_path / "admins"
    admins.write_text("")
    app = _probe_app(provider)
    app.include_router(
        oidc_routes.create_auth_router(provider, None, AdminList(admins)), prefix="/auth"
    )
    return TestClient(app, base_url=origin), provider


def _oidc_login(client: TestClient, origin: str) -> httpx.Response:
    """Complete ``/auth/callback`` with a valid state cookie."""
    from omnigent.server.routes.auth import _AUTH_STATE_COOKIE_PLAIN, _AUTH_STATE_COOKIE_SECURE

    state_cookie = (
        _AUTH_STATE_COOKIE_SECURE if origin.startswith("https://") else _AUTH_STATE_COOKIE_PLAIN
    )
    state_jwt = jwt.encode(
        {
            "state": "state-xyz",
            "code_verifier": "verifier",
            "return_to": "/",
            "exp": int(time.time()) + 300,
        },
        _SECRET,
        algorithm="HS256",
    )
    return client.get(
        "/auth/callback?code=auth-code&state=state-xyz",
        headers=_cookie(state_cookie, state_jwt),
        follow_redirects=False,
    )


@pytest.mark.parametrize("origin", [_HTTP, _HTTPS])
def test_oidc_renewed_cookie_matches_login_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    """OIDC mode: the renewed cookie is attribute-identical to the callback's."""
    client, _ = _oidc_client(monkeypatch, tmp_path, origin)
    name = _cookie_name(origin)
    with client:
        login = _oidc_login(client, origin)
        assert login.status_code == 302, login.text
        login_header = _session_set_cookies(login, name)[0]
        login_claims = _decode(_parse_set_cookie(login_header)[1])
        assert login_claims["auth_time"] == login_claims["iat"]

        aged = _session_jwt(remaining=600, provider="oidc", sid=login_claims["sid"])
        resp = client.get(_PROBE, headers=_cookie(name, aged))

        assert resp.status_code == 200, resp.text
        renewed_header = _session_set_cookies(resp, name)[0]
        assert _attributes(renewed_header) == _attributes(login_header)
        login_name, login_attrs = _attributes(login_header)
        assert login_name == name
        assert login_attrs.get("secure", False) is origin.startswith("https://")


def test_login_response_is_not_double_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Logging in with an aged cookie yields only the login's own cookie."""
    client, _ = _oidc_client(monkeypatch, tmp_path, _HTTP)
    name = _cookie_name(_HTTP)
    with client:
        client.cookies.set(name, _session_jwt(remaining=600, provider="oidc"))
        login = _oidc_login(client, _HTTP)
        assert login.status_code == 302, login.text
        assert len(_session_set_cookies(login, name)) == 1


# ── Contract 6: logout ends the session for good ─────────────────


@pytest.mark.parametrize("origin", [_HTTP, _HTTPS])
def test_accounts_logout_ends_renewed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    """Accounts logout clears the cookie and no renewal of it survives."""
    for client in _build_accounts_app(
        tmp_path, monkeypatch, init_admin_password="admin-pw-12345", base_url=origin
    ):
        name = _cookie_name(origin)
        _login(client, "admin", "admin-pw-12345")
        secret = bytes.fromhex(os.environ["OMNIGENT_ACCOUNTS_COOKIE_SECRET"])
        login_claims = _decode(client.cookies[name], secret)
        aged = _session_jwt(
            remaining=600,
            secret=secret,
            sub="admin",
            account_generation=login_claims["account_generation"],
            sid=login_claims["sid"],
        )
        renewed = _renewed_token(client.get("/auth/me", headers=_cookie(name, aged)), name)

        logout = client.post("/auth/logout", headers=_cookie(name, aged))

        assert logout.status_code == 204
        headers = _session_set_cookies(logout, name)
        assert len(headers) == 1
        _, value, attrs = _parse_set_cookie(headers[0])
        assert value in ("", '""')
        assert attrs["max-age"] == "0"
        for token in (aged, renewed):
            after = client.get("/auth/me", headers=_cookie(name, token))
            assert after.status_code == 401
            assert _session_set_cookies(after, name) == []
        # Recorded durably, where other replicas and restarts will see it.
        durable = BrowserSessionRevocationStore(f"sqlite:///{tmp_path}/test.db")
        assert durable.is_revoked(login_claims["sid"])


@pytest.mark.parametrize("origin", [_HTTP, _HTTPS])
def test_oidc_logout_ends_renewed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    """OIDC logout clears the cookie and no renewal of it survives."""
    client, _ = _oidc_client(monkeypatch, tmp_path, origin)
    name = _cookie_name(origin)
    with client:
        aged = _session_jwt(remaining=600, provider="oidc")
        renewed = _renewed_token(client.get(_PROBE, headers=_cookie(name, aged)), name)

        logout = client.get("/auth/logout", headers=_cookie(name, renewed), follow_redirects=False)

        assert logout.status_code == 302
        headers = _session_set_cookies(logout, name)
        assert len(headers) == 1
        assert _parse_set_cookie(headers[0])[2]["max-age"] == "0"
        for token in (aged, renewed):
            after = client.get(_PROBE, headers=_cookie(name, token))
            assert after.status_code == 401
            assert _session_set_cookies(after, name) == []


@pytest.mark.parametrize(("source", "origin"), _MODES)
def test_request_racing_logout_gets_no_renewal(probe: Any, source: str, origin: str) -> None:
    """A logout landing between authentication and response start wins.

    The request was authenticated and marked for renewal, but renewal
    re-checks the session at response time and attaches nothing.
    """
    client, _ = probe(source, origin)
    name = _cookie_name(origin)
    aged = _session_jwt(remaining=600)

    resp = client.get("/v1/sessions/logout-race", headers=_cookie(name, aged))

    assert resp.status_code == 200, resp.text
    assert _session_set_cookies(resp, name) == []
    assert client.get(_PROBE, headers=_cookie(name, aged)).status_code == 401


def test_browser_logout_leaves_bearer_copy_valid(probe: Any) -> None:
    """Logout ends the cookie session only; a CLI holding the same JWT keeps it.

    The OIDC CLI-ticket flow hands one JWT to both the CLI and the
    browser, so a browser logout must not sign the CLI out.
    """
    client, provider = probe("oidc", _HTTP)
    name = _cookie_name(_HTTP)
    token = _session_jwt(remaining=600)
    provider.end_browser_session(_Conn(cookies={name: token}))  # type: ignore[arg-type]

    assert client.get(_PROBE, headers=_cookie(name, token)).status_code == 401
    bearer = client.get(_PROBE, headers={"Authorization": f"Bearer {token}"})
    assert bearer.status_code == 200
    assert bearer.headers.get_list("set-cookie") == []


# ── Contract 7: credential cache ─────────────────────────────────


def test_cache_entry_never_outlives_token_exp() -> None:
    """A cached identity expires no later than the token itself."""
    provider = _provider("oidc", _HTTP)
    token = _session_jwt(remaining=120)

    assert provider.get_user_id(_Conn(cookies={"ap_session": token})) == _USER  # type: ignore[arg-type]

    _, deadline, _ = provider._cookie_cache[hmac_digest(token, _SECRET)]
    assert deadline - time.monotonic() <= 121


def test_renewed_token_gets_its_own_validated_cache_entry(probe: Any) -> None:
    """The successor is validated and cached on its own claims, not the old ones."""
    client, provider = probe("oidc", _HTTP)
    name = _cookie_name(_HTTP)
    old = _session_jwt(remaining=600)
    renewed = _renewed_token(client.get(_PROBE, headers=_cookie(name, old)), name)
    renewed_key = hmac_digest(renewed, _SECRET)
    assert renewed_key not in provider._cookie_cache

    resp = client.get(_PROBE, headers=_cookie(name, renewed))

    assert resp.status_code == 200
    assert _session_set_cookies(resp, name) == []  # fresh: not due again
    user, deadline, claims = provider._cookie_cache[renewed_key]
    assert user == _USER
    assert claims["exp"] == _decode(renewed)["exp"]
    assert deadline - time.monotonic() <= _TTL + 1
    old_claims = provider._cookie_cache[hmac_digest(old, _SECRET)][2]
    assert old_claims["exp"] < claims["exp"]


def test_cache_hit_still_offers_renewal(probe: Any) -> None:
    """An aged cookie served from the cache is still renewed."""
    client, provider = probe("oidc", _HTTP)
    name = _cookie_name(_HTTP)
    old = _session_jwt(remaining=600)
    client.get(_PROBE, headers=_cookie(name, old))
    assert hmac_digest(old, _SECRET) in provider._cookie_cache

    resp = client.get(_PROBE, headers=_cookie(name, old))

    assert len(_session_set_cookies(resp, name)) == 1


def test_expired_cache_entries_are_pruned() -> None:
    """Caching a new token drops entries whose token has expired."""
    provider = _provider("oidc", _HTTP)
    provider._cookie_cache["stale"] = (_USER, time.monotonic() - 1, {})

    provider.get_user_id(_Conn(cookies={"ap_session": _session_jwt(remaining=600)}))  # type: ignore[arg-type]

    assert "stale" not in provider._cookie_cache


def test_logout_evicts_cached_entries_of_the_session() -> None:
    """Logout drops every cached token of the session; none is served again."""
    provider = _provider("oidc", _HTTP)
    first = _session_jwt(remaining=600)
    second = _session_jwt(remaining=_TTL - 10)
    other = _session_jwt(remaining=600, sid="sid-other")
    for token in (first, second, other):
        assert provider.get_user_id(_Conn(cookies={"ap_session": token})) == _USER  # type: ignore[arg-type]

    provider.end_browser_session(_Conn(cookies={"ap_session": second}))  # type: ignore[arg-type]

    assert hmac_digest(first, _SECRET) not in provider._cookie_cache
    assert hmac_digest(second, _SECRET) not in provider._cookie_cache
    assert hmac_digest(other, _SECRET) in provider._cookie_cache
    assert provider.get_user_id(_Conn(cookies={"ap_session": first})) is None  # type: ignore[arg-type]
    assert provider.get_user_id(_Conn(cookies={"ap_session": other})) == _USER  # type: ignore[arg-type]


# ── Contract 8: configuration ────────────────────────────────────


def test_login_cookie_stamps_auth_time_and_fresh_sid() -> None:
    """Every login mints its own session id, with auth_time = iat."""
    first = _decode(mint_session_cookie(_USER, _SECRET, _TTL_HOURS, "accounts"))
    second = _decode(mint_session_cookie(_USER, _SECRET, _TTL_HOURS, "accounts"))

    assert first["auth_time"] == first["iat"]
    assert isinstance(first["sid"], str) and first["sid"]
    assert first["sid"] != second["sid"]


def _accounts_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIGENT_ACCOUNTS_COOKIE_SECRET", secrets.token_hex(32))
    monkeypatch.setenv("OMNIGENT_ACCOUNTS_BASE_URL", _HTTP)
    monkeypatch.delenv("OMNIGENT_ACCOUNTS_SESSION_TTL_HOURS", raising=False)


def _oidc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # GitHub skips discovery, so from_env needs no network.
    monkeypatch.setenv("OMNIGENT_OIDC_ISSUER", "https://github.com")
    monkeypatch.setenv("OMNIGENT_OIDC_CLIENT_ID", "cid")
    monkeypatch.setenv("OMNIGENT_OIDC_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OMNIGENT_OIDC_REDIRECT_URI", f"{_HTTP}/auth/callback")
    monkeypatch.setenv("OMNIGENT_OIDC_COOKIE_SECRET", secrets.token_hex(32))
    monkeypatch.delenv("OMNIGENT_OIDC_SESSION_TTL_HOURS", raising=False)


_CONFIG_SOURCES = [
    pytest.param(_accounts_env, AccountsConfig.from_env, "OMNIGENT_ACCOUNTS", id="accounts"),
    pytest.param(_oidc_env, OIDCConfig.from_env, "OMNIGENT_OIDC", id="oidc"),
]


@pytest.mark.parametrize(("set_env", "from_env", "prefix"), _CONFIG_SOURCES)
def test_max_lifetime_defaults_to_thirty_days(
    monkeypatch: pytest.MonkeyPatch, set_env: Any, from_env: Any, prefix: str
) -> None:
    set_env(monkeypatch)

    config = from_env()

    assert config.session_ttl_hours == 8
    assert config.session_max_lifetime_hours == 720
    assert config.session_max_lifetime_seconds == 720 * 3600


@pytest.mark.parametrize(("set_env", "from_env", "prefix"), _CONFIG_SOURCES)
def test_max_lifetime_reads_env(
    monkeypatch: pytest.MonkeyPatch, set_env: Any, from_env: Any, prefix: str
) -> None:
    set_env(monkeypatch)
    monkeypatch.setenv(f"{prefix}_SESSION_MAX_LIFETIME_HOURS", "48")

    assert from_env().session_max_lifetime_hours == 48


@pytest.mark.parametrize(("set_env", "from_env", "prefix"), _CONFIG_SOURCES)
def test_max_lifetime_default_never_undercuts_long_ttl(
    monkeypatch: pytest.MonkeyPatch, set_env: Any, from_env: Any, prefix: str
) -> None:
    """An existing idle window above 30 days keeps working after upgrade."""
    set_env(monkeypatch)
    monkeypatch.setenv(f"{prefix}_SESSION_TTL_HOURS", "1000")

    assert from_env().session_max_lifetime_hours == 1000


@pytest.mark.parametrize("value", ["4", "forever"])
@pytest.mark.parametrize(("set_env", "from_env", "prefix"), _CONFIG_SOURCES)
def test_max_lifetime_invalid_fails_loud(
    monkeypatch: pytest.MonkeyPatch, set_env: Any, from_env: Any, prefix: str, value: str
) -> None:
    """A non-integer, or a max shorter than the idle window, fails at startup."""
    set_env(monkeypatch)
    monkeypatch.setenv(f"{prefix}_SESSION_MAX_LIFETIME_HOURS", value)

    with pytest.raises(RuntimeError, match=f"{prefix}_SESSION_MAX_LIFETIME_HOURS"):
        from_env()


# ── Production wiring ────────────────────────────────────────────


def test_create_app_renews_oidc_session_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_app`` installs renewal in OIDC mode (accounts: see above)."""
    from omnigent.db.utils import get_or_create_engine
    from omnigent.runtime import init as init_runtime
    from omnigent.runtime import telemetry
    from omnigent.runtime.agent_cache import AgentCache
    from omnigent.runtime.caps import RuntimeCaps
    from omnigent.server.app import create_app
    from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from omnigent.stores.comment_store.sqlalchemy_store import SqlAlchemyCommentStore
    from omnigent.stores.conversation_store.sqlalchemy_store import (
        SqlAlchemyConversationStore,
    )
    from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
    from omnigent.stores.host_store import HostStore
    from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMNIGENT_DATA_DIR", str(tmp_path / ".omnigent"))
    db_url = f"sqlite:///{tmp_path}/oidc.db"
    get_or_create_engine(db_url)
    telemetry.init()
    agent_store = SqlAlchemyAgentStore(db_url)
    conversation_store = SqlAlchemyConversationStore(db_url)
    file_store = SqlAlchemyFileStore(db_url)
    comment_store = SqlAlchemyCommentStore(db_url)
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    agent_cache = AgentCache(artifact_store=artifact_store, cache_dir=tmp_path / "cache")
    init_runtime(
        agent_cache=agent_cache,
        caps=RuntimeCaps(),
        agent_store=agent_store,
        file_store=file_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
        comment_store=comment_store,
    )
    app = create_app(
        agent_store=agent_store,
        file_store=file_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
        agent_cache=agent_cache,
        comment_store=comment_store,
        permission_store=SqlAlchemyPermissionStore(db_url),
        host_store=HostStore(db_url),
        auth_provider=_provider("oidc", _HTTP),
        account_store=None,
    )
    name = _cookie_name(_HTTP)
    with TestClient(app) as client:
        resp = client.get("/v1/me", headers=_cookie(name, _session_jwt(remaining=600)))

        assert resp.status_code == 200, resp.text
        assert resp.json()["user_id"] == _USER
        assert _decode(_renewed_token(resp, name))["sid"] == "sid-abc"


# ── Review round 1 regressions ───────────────────────────────────


class _Clock:
    """Replace ``time.time`` with a controllable wall clock.

    Starts at the real time; ``tick`` seconds are added after every read,
    so ``tick=1`` models a clock that crosses a second boundary between
    any two reads.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, tick: int = 0) -> None:
        self.now = float(int(time.time()))
        self.tick = tick
        monkeypatch.setattr(time, "time", self)

    def __call__(self) -> float:
        value = self.now
        self.now += self.tick
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_request_outliving_its_token_is_not_renewed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token valid at authentication but expired by response start is not renewed."""
    provider = _provider("oidc", _HTTP)
    clock = _Clock(monkeypatch)
    app = FastAPI()

    @app.get(_PROBE)
    def slow(request: Request) -> Response:
        assert provider.get_user_id(request) == _USER
        clock.advance(30)  # the token expires while the handler runs
        return JSONResponse(content={"ok": True})

    app.add_middleware(SessionRenewalMiddleware, auth_provider=provider)
    with TestClient(app) as client:
        resp = client.get(_PROBE, headers=_cookie("ap_session", _session_jwt(remaining=10)))

    assert resp.status_code == 200
    assert _session_set_cookies(resp, "ap_session") == []


def test_logout_with_expired_cookie_still_ends_the_session(db_uri: str) -> None:
    """Logout records the sid even when its cookie has just expired.

    A request that authenticated just before expiry may still be renewing
    the session; that renewal must find the session ended.
    """
    store = BrowserSessionRevocationStore(db_uri)
    provider = _provider("oidc", _HTTP)
    provider.set_session_revocation_store(store)
    expired = _session_jwt(remaining=-30)

    provider.end_browser_session(_Conn(cookies={"ap_session": expired}))  # type: ignore[arg-type]

    assert store.is_revoked("sid-abc")
    assert "sid-abc" in provider._ended_sessions
    assert provider.renew_browser_session(_decode(_session_jwt(remaining=600))) is None


def test_logout_ignores_a_forged_expired_cookie(db_uri: str) -> None:
    """Skipping exp verification at logout still verifies the signature."""
    store = BrowserSessionRevocationStore(db_uri)
    provider = _provider("oidc", _HTTP)
    provider.set_session_revocation_store(store)
    forged = _session_jwt(remaining=-30, secret=b"x" * 32)

    provider.end_browser_session(_Conn(cookies={"ap_session": forged}))  # type: ignore[arg-type]

    assert not store.is_revoked("sid-abc")
    assert provider._ended_sessions == {}


def test_cache_hit_past_wall_clock_exp_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached token is rejected once wall time passes its exp.

    The cache deadline is monotonic, so a wall-clock jump can leave it in
    the future after the JWT has expired.
    """
    provider = _provider("oidc", _HTTP)
    clock = _Clock(monkeypatch)
    token = _session_jwt(remaining=30)
    assert provider.get_user_id(_Conn(cookies={"ap_session": token})) == _USER  # type: ignore[arg-type]
    key = hmac_digest(token, _SECRET)

    clock.advance(45)
    late = _Conn(cookies={"ap_session": token})

    assert provider._cookie_cache[key][1] > time.monotonic()
    assert provider.get_user_id(late) is None  # type: ignore[arg-type]
    assert _SESSION_RENEWAL_KEY not in late.scope
    assert key not in provider._cookie_cache


def test_capped_renewal_exp_is_exact_when_the_clock_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renewal reads the clock once, so a capped exp never overshoots by a second."""
    provider = _provider("accounts", _HTTP, max_hours=10)
    payload = _decode(_session_jwt(remaining=1800, logged_in_ago=int(8.5 * 3600)))
    _Clock(monkeypatch, tick=1)  # every read crosses a second boundary

    renewed = provider.renew_browser_session(payload)

    assert renewed is not None
    token, max_age = renewed
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["exp"] == payload["auth_time"] + 10 * 3600
    assert claims["exp"] == claims["iat"] + max_age


@pytest.mark.parametrize("source", ["accounts", "oidc"])
def test_logout_on_one_replica_ends_the_session_on_others(
    db_uri: str, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Replicas sharing a database honour each other's logouts, and so does a restart.

    Accounts mode checks every request. An OIDC replica that already cached
    the cookie may accept it until the cache recheck window ends, but never
    renews it.
    """

    def replica() -> UnifiedAuthProvider:
        provider = _provider(source, _HTTP)
        provider.set_session_revocation_store(BrowserSessionRevocationStore(db_uri))
        if source == "accounts":
            provider.set_account_check(lambda user, generation: True)
        return provider

    replica_a, replica_b = replica(), replica()
    name = _cookie_name(_HTTP)
    aged = _session_jwt(remaining=600)
    headers = _cookie(name, aged)
    with TestClient(_probe_app(replica_b)) as client_b:
        assert len(_session_set_cookies(client_b.get(_PROBE, headers=headers), name)) == 1

        replica_a.end_browser_session(_Conn(cookies={name: aged}))  # type: ignore[arg-type]

        resp = client_b.get(_PROBE, headers=headers)
        assert _session_set_cookies(resp, name) == []
        if source == "accounts":
            assert resp.status_code == 401
        else:
            assert resp.status_code == 200  # cached, inside the recheck window
            real_monotonic = time.monotonic
            monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 61)
            assert client_b.get(_PROBE, headers=headers).status_code == 401

    restarted = replica()
    assert restarted.get_user_id(_Conn(cookies={name: aged})) is None  # type: ignore[arg-type]
    assert restarted.renew_browser_session(_decode(aged)) is None


def test_account_generation_change_blocks_response_time_renewal() -> None:
    """A generation bump between authentication and response start yields no cookie."""
    provider = _provider("accounts", _HTTP)
    current = {"generation": "gen-1"}
    provider.set_account_check(lambda user, generation: generation == current["generation"])
    payload = _decode(_session_jwt(remaining=600))
    assert provider.renew_browser_session(payload) is not None

    current["generation"] = "gen-2"

    assert provider.renew_browser_session(payload) is None
