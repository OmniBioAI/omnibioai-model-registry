from __future__ import annotations

"""Model Registry IAM Integration.

Centralized authentication via the IAM client (omnibioai-iam-client, the
same package/pattern already adopted by omnibioai-lims's
core/authentication.py -- "PR10" there -- and omnibioai-api-gateway).
Local HS256-only JWT decoding (the old validate_token()) is removed
entirely; a token is now verified by AsyncIAMClient.get_user(), which
does local RS256/JWKS-or-HS256 signature verification followed by an
authoritative revocation check against omnibioai-auth (see that
package's client.py for the current implementation) -- no JWT decoding
happens locally in this service anymore, matching the "do not duplicate
IAM logic" instruction.

Every authenticated call is authorized against the existing `model.use`
IAM permission -- never by role name, username, or a hardcoded admin
check (the old require_write_auth was a bare copy of require_auth with
no permission check at all; this replaces both with real enforcement).
No new permission name is introduced.

Organization scope: organization_id is extracted from the verified
identity and included in every model_access audit event (see below), but
is NOT used to filter or partition registry data in this PR -- confirmed
scope. The registry's data model has no tenant/organization concept
anywhere (omr_runs/omr_params/omr_metrics/omr_tags/omr_version_tags have
no organization_id column, and models/versions are keyed only by
task/model_name/version). Retrofitting real per-org data isolation is a
follow-up, not attempted here:

    Model Registry multi-tenant isolation requires a dedicated data model
    migration:
    - define an ownership model for models/artifacts
    - add organization_id where appropriate
    - decide a migration/backfill strategy for already-registered models
    - enforce organization-level filtering at the query layer

Audit events reuse the existing AuditClient (audit_client.py) -- already
wired into service/app/main.py's register/promote/set_tag/set_stage
routes (fire-and-forget POST to AUDIT_URL, matching the omnibioai-
security-audit producer convention) -- rather than a new audit module.
Those 4 existing call sites already cover the "model registration"/
"model update" event types; this module adds the "model access
success"/"model access denied" events the IAM layer itself introduces.
"model deletion" has no wiring here -- no delete/archive-removal endpoint
exists anywhere in this service's API today (confirmed by inspection);
AuditClient's generic log_event() needs no new plumbing to cover one if
it's ever added.

require_auth/require_write_auth keep their existing `str` return contract
(the actor identifier) unchanged, so none of this repo's ~14 existing
route handlers (which pass this value straight through to storage, e.g.
omr_runs.actor) need to change -- only this module does.
"""

from typing import Annotated

from fastapi import Header, HTTPException
from iam_client import AsyncIAMClient
from iam_client.models import UserContext

from .audit_client import AuditClient
from .config import load_config

MODEL_USE_PERMISSION = "model.use"


class AuthError(Exception):
    """Raised when JWT validation or authorization fails."""

    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_token(authorization_header: str | None) -> str:
    """Extracts a Bearer token from an Authorization header. This is the
    exact header omnibioai-api-gateway forwards a caller's verified JWT
    through unchanged (app/routes/gateway.py sets
    upstream_headers["Authorization"] = f"Bearer {token}"), so a request
    arriving via the gateway and one arriving directly are handled
    identically here -- no gateway-specific header or trust boundary
    needed. Raises AuthError(401) if the header is missing or malformed.
    """
    if not authorization_header:
        raise AuthError("Authorization header is missing", 401)
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Authorization header must be 'Bearer <token>'", 401)
    return parts[1]


async def verify_and_authorize(token: str, action: str) -> UserContext:
    """Verifies `token` via the centralized IAM client (signature +
    revocation -- see module docstring) and enforces the model.use
    permission. Raises AuthError(401) for any invalid/expired/revoked
    token, AuthError(403) if the identity lacks model.use. Logs a
    model_access_success or model_access_denied audit event on every
    outcome, including organization_id when the identity carries one.

    A fresh AsyncIAMClient per call, not a module-level singleton -- its
    httpx.AsyncClient binds its connection pool to whichever asyncio event
    loop first uses it, and a FastAPI request handler already runs inside
    its own request-scoped loop; reusing one client across requests risks
    reusing connections bound to a closed loop. Same reasoning
    omnibioai-lims' core/authentication.py documents for its own (WSGI
    -bridged) equivalent.
    """
    cfg = load_config()
    audit = AuditClient(cfg.audit_url)
    client = AsyncIAMClient(base_url=cfg.iam_url, redis_url=cfg.iam_redis_url)
    try:
        user = await client.get_user(token, cfg.jwt_secret)
    except Exception:
        user = None
    finally:
        await client.http.aclose()

    if user is None or not user.valid:
        audit.log_event(
            "model_access_denied", "unknown", action,
            metadata={"reason": "invalid_token"},
        )
        raise AuthError("Invalid, expired, or revoked token", 401)

    if MODEL_USE_PERMISSION not in (user.permissions or []):
        audit.log_event(
            "model_access_denied", _actor_identifier(user), action,
            metadata={"reason": "missing_permission", "organization_id": user.org_id},
        )
        raise AuthError(f"Missing required permission: {MODEL_USE_PERMISSION}", 403)

    audit.log_event(
        "model_access_success", _actor_identifier(user), action,
        metadata={"organization_id": user.org_id},
    )
    return user


def _actor_identifier(user: UserContext) -> str:
    """Preserves the old get_actor()'s string-actor contract for the
    ~14 existing route handlers that store this value directly (e.g.
    omr_runs.actor VARCHAR(128)) -- email first, falling back to
    user_id, mirroring that function's old field-precedence intent
    without needing a raw, unverified JWT payload dict anymore."""
    return user.email or str(user.user_id) or "unknown"


async def require_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """FastAPI dependency. Returns the actor string (unchanged contract).

    If auth_enabled=False: returns "system" immediately, matching the
    prior local-only behavior exactly (used by tests / any deployment
    that deliberately disables auth).
    If auth_enabled=True: verifies the token via the centralized IAM
    client, enforces model.use, returns the actor identifier.
    Raises HTTPException(401) on a missing/invalid/expired/revoked token,
    HTTPException(403) if the identity lacks model.use.
    """
    cfg = load_config()
    if not cfg.auth_enabled:
        return "system"
    try:
        token = extract_token(authorization)
        user = await verify_and_authorize(token, action="model_access")
        return _actor_identifier(user)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


async def require_write_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Same enforcement as require_auth -- both are gated by the same
    model.use permission (no separate write-scoped permission exists;
    per this task's explicit instruction, no new permission name is
    introduced). Kept as a distinct function rather than a plain alias
    since call sites already Depends() on this specific name and it
    documents intent (a write action) at each call site even though the
    enforcement itself is identical."""
    return await require_auth(authorization)


async def require_write_auth_with_context(
    authorization: Annotated[str | None, Header()] = None,
) -> UserContext:
    """PR14.2B-3: same enforcement as require_write_auth, but returns the
    full UserContext instead of just the actor string -- for the one call
    site that needs organization_id (api_register(), for usage-event
    emission). Additive, not a signature change to require_write_auth
    itself: every other require_write_auth-dependent route
    (promote/set_tag/patch_version/set_stage) is unaffected, unchanged,
    still gets a bare str.

    When auth_enabled=False, returns a synthetic "system" UserContext
    with org_id=None -- api_register()'s usage-emission call already
    treats organization_id=None as "skip emission, don't block the
    request" the same way every other PR14.2B producer does.
    """
    cfg = load_config()
    if not cfg.auth_enabled:
        return UserContext(
            user_id="system", email="system", roles=[], permissions=[MODEL_USE_PERMISSION],
            valid=True, org_id=None,
        )
    try:
        token = extract_token(authorization)
        return await verify_and_authorize(token, action="model_access")
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
