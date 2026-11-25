"""Per-launch bearer token for the loopback WebSocket transport.

Loopback is reachable by every local process and user, so the WebSocket server
requires a secret. The server writes it to an owner-only file; local clients read
that file. Browsers, which cannot set headers on WebSocket upgrades, pass it as
the ``token`` query parameter.
"""
from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

TOKEN_ENV = "GAMST_WS_TOKEN"
TOKEN_FILE_ENV = "GAMST_WS_TOKEN_FILE"


def token_path() -> Path:
    configured = os.environ.get(TOKEN_FILE_ENV)
    return Path(configured).expanduser() if configured else Path.home() / ".gamst" / "websocket-token"


def issue_token(path: Path | None = None) -> str:
    """Create a fresh token readable only by the current user."""
    path = path or token_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    os.replace(temporary, path)
    return token


def load_token(path: Path | None = None) -> str:
    """Token for local clients: explicit environment value, then the token file."""
    if os.environ.get(TOKEN_ENV):
        return os.environ[TOKEN_ENV]
    path = path or token_path()
    try:
        return path.read_text().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"WebSocket token not found at {path}; start the server first") from exc


def request_token(path: str, headers) -> str | None:
    authorization = headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        return authorization[7:].strip()
    values = parse_qs(urlsplit(path).query).get("token")
    return values[0] if values else None


def authorized(expected: str, path: str, headers) -> bool:
    supplied = request_token(path, headers)
    return supplied is not None and hmac.compare_digest(supplied.encode(), expected.encode())


def connect_local(url: str = "ws://127.0.0.1:8765", **options):
    """``websockets`` client connection carrying the local token."""
    from websockets.asyncio.client import connect
    return connect(url, additional_headers={"Authorization": "Bearer " + load_token()}, **options)
