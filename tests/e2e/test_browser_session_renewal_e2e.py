"""An active browser session slides forward instead of expiring mid-use.

A real ``omnigent server`` runs in accounts mode. The test signs in through
``POST /auth/login`` as the web form does, then presents a session cookie aged
past half of its idle window (re-signed with the server's own cookie secret,
keeping the login's identity and session id, so the test need not wait hours).
The server must answer with a fresh cookie that keeps the original login time,
accept that cookie without renewing it again, and after logout accept neither.

Usage::

    python -m pytest tests/e2e/test_browser_session_renewal_e2e.py -v --timeout=300
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import jwt
import pytest

from tests._helpers.compat import apply_server_env, compat_server_cwd, server_executable
from tests._helpers.live_server import find_free_port

_REPO_ROOT = Path(__file__).resolve().parents[2]

_COOKIE_SECRET_HEX = "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4"
_ADMIN_USERNAME = "admin"
_ADMIN_PASSWORD = "session-renewal-test-pw"
_TTL_SECONDS = 8 * 3600
_SERVER_HEALTH_TIMEOUT_S = 60.0


def _await_health(base_url: str, log_path: Path) -> None:
    deadline = time.monotonic() + _SERVER_HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=2, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    tail = log_path.read_text()[-3000:] if log_path.exists() else "(no log)"
    raise RuntimeError(f"accounts server did not become healthy. Log:\n{tail}")


@pytest.fixture()
def accounts_server(tmp_path: Path) -> Iterator[str]:
    """Run a real ``omnigent server`` subprocess with accounts auth enabled."""
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    log_path = tmp_path / "server.log"

    env = {**os.environ}
    env["OMNIGENT_AUTH_PROVIDER"] = "accounts"
    env["OMNIGENT_ACCOUNTS_COOKIE_SECRET"] = _COOKIE_SECRET_HEX
    env["OMNIGENT_ACCOUNTS_BASE_URL"] = base_url
    env["OMNIGENT_ACCOUNTS_INIT_ADMIN_USERNAME"] = _ADMIN_USERNAME
    env["OMNIGENT_ACCOUNTS_INIT_ADMIN_PASSWORD"] = _ADMIN_PASSWORD
    env["OMNIGENT_ACCOUNTS_SESSION_TTL_HOURS"] = str(_TTL_SECONDS // 3600)
    for name in ("OMNIGENT_OIDC_ISSUER", "OMNIGENT_ACCOUNTS_SESSION_MAX_LIFETIME_HOURS"):
        env.pop(name, None)
    apply_server_env(env, _REPO_ROOT)

    log_handle = open(log_path, "w")  # noqa: SIM115 -- handle lives for the subprocess
    proc = subprocess.Popen(
        [
            server_executable(),
            "-m",
            "omnigent.cli",
            "server",
            "--port",
            str(port),
            "--database-uri",
            f"sqlite:///{tmp_path / 'e2e.db'}",
            "--artifact-location",
            str(artifact_dir),
        ],
        env=env,
        cwd=compat_server_cwd(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    try:
        _await_health(base_url, log_path)
        yield base_url
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_handle.close()


def _session_cookies(resp: httpx.Response) -> list[str]:
    return [h for h in resp.headers.get_list("set-cookie") if h.startswith("ap_session=")]


def _me(base_url: str, token: str) -> httpx.Response:
    return httpx.get(
        f"{base_url}/v1/me",
        headers={"Cookie": f"ap_session={token}"},
        timeout=10,
        trust_env=False,
    )


def test_active_browser_session_is_renewed_until_logout(accounts_server: str) -> None:
    """Login → aged cookie renewed → renewed cookie works → logout ends it."""
    secret = bytes.fromhex(_COOKIE_SECRET_HEX)
    login = httpx.post(
        f"{accounts_server}/auth/login",
        json={"username": _ADMIN_USERNAME, "password": _ADMIN_PASSWORD},
        timeout=10,
        trust_env=False,
    )
    assert login.status_code == 200, login.text
    login_claims = jwt.decode(login.cookies["ap_session"], secret, algorithms=["HS256"])
    assert login_claims["auth_time"] == login_claims["iat"]

    # The same session five hours after login: three hours of its idle window left.
    now = int(time.time())
    aged = jwt.encode(
        {
            **login_claims,
            "auth_time": now - 5 * 3600,
            "iat": now - 5 * 3600,
            "exp": now + 3 * 3600,
        },
        secret,
        algorithm="HS256",
    )
    renewed_resp = _me(accounts_server, aged)
    assert renewed_resp.status_code == 200, renewed_resp.text
    assert renewed_resp.json()["user_id"] == _ADMIN_USERNAME
    set_cookies = _session_cookies(renewed_resp)
    assert len(set_cookies) == 1, renewed_resp.headers.get_list("set-cookie")
    assert "httponly" in set_cookies[0].lower()
    renewed = renewed_resp.cookies["ap_session"]
    renewed_claims = jwt.decode(renewed, secret, algorithms=["HS256"])
    for claim in ("sub", "provider", "account_generation", "sid"):
        assert renewed_claims[claim] == login_claims[claim], claim
    assert renewed_claims["auth_time"] == now - 5 * 3600
    assert renewed_claims["exp"] >= now + _TTL_SECONDS - 60

    fresh = _me(accounts_server, renewed)
    assert fresh.status_code == 200, fresh.text
    assert _session_cookies(fresh) == []

    logout = httpx.post(
        f"{accounts_server}/auth/logout",
        headers={"Cookie": f"ap_session={renewed}"},
        timeout=10,
        trust_env=False,
    )
    assert logout.status_code == 204
    for token in (renewed, aged):
        after = _me(accounts_server, token)
        assert after.status_code == 401, after.text
        assert _session_cookies(after) == []
