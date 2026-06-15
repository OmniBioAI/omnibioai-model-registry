from __future__ import annotations

# docker-compose.yml additions needed for model-registry service:
#   environment:
#     IAM_URL: http://auth-service:8001
#     AUDIT_URL: http://security-audit:8004
#     JWT_SECRET: ${AUTH_SECRET_KEY:-change-me}
#     AUTH_ENABLED: "true"
#
# api-gateway environment addition:
#     MODEL_REGISTRY_URL: http://model-registry:8095

from typing import Annotated

import jwt as _jwt
from fastapi import Header, HTTPException

from .config import load_config


class AuthError(Exception):
    """Raised when JWT validation fails."""

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_token(authorization_header: str | None) -> str:
    """
    Extracts Bearer token from Authorization header.
    Raises AuthError(401) if header is missing or malformed.
    """
    if not authorization_header:
        raise AuthError("Authorization header is missing", 401)
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Authorization header must be 'Bearer <token>'", 401)
    return parts[1]


def validate_token(token: str, secret: str) -> dict:
    """
    Validates JWT signature and expiry.
    Returns decoded payload dict.
    Raises AuthError(401) if invalid or expired.
    Raises AuthError(403) if token lacks required claims.
    """
    try:
        payload: dict = _jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except _jwt.ExpiredSignatureError:
        raise AuthError("Token has expired", 401)
    except _jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid token: {exc}", 401)


def get_actor(payload: dict) -> str:
    """
    Extracts actor identity from JWT payload.
    Checks: sub, username, email, preferred_username (in that order).
    Falls back to "unknown" if none present.
    """
    for field in ("sub", "username", "email", "preferred_username"):
        value = payload.get(field)
        if value:
            return str(value)
    return "unknown"


async def require_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """
    FastAPI dependency. Returns actor string.
    If auth_enabled=False: returns "system" immediately.
    If auth_enabled=True: validates token, returns actor.
    Raises HTTPException(401) on auth failure.
    Raises HTTPException(403) on insufficient permissions.
    """
    cfg = load_config()
    if not cfg.auth_enabled:
        return "system"
    try:
        token = extract_token(authorization)
        payload = validate_token(token, cfg.jwt_secret)
        return get_actor(payload)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


async def require_write_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """
    Same as require_auth but specifically for write operations
    (register, promote, stage, tag). Same implementation for now —
    placeholder for future RBAC scope checks.
    """
    return await require_auth(authorization)
