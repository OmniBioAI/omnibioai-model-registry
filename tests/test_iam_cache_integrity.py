"""HIPAA PHI P1-5: IAM Redis cache-integrity regression tests.

Companion to TestVerifyAndAuthorize in test_model_registry.py (which mocks
AsyncIAMClient wholesale and never exercises the real Redis-backed cache
layer at all). These tests prove two things that class cannot:

1. verify_and_authorize() actually wires cfg.jwt_secret through to
   AsyncIAMClient's cache_secret kwarg -- a silently dropped/renamed kwarg
   here would leave the cache unsigned with every existing test still
   green, since they never touch the real AsyncIAMClient.
2. Once cache_secret IS configured, the real (unmocked) AsyncIAMClient
   cache layer actually rejects forged/tampered/malformed/wrong-key
   entries and accepts only entries it signed itself -- the same
   authorization bypass omnibioai-tes's PR #22 fixed for itself. Before
   this fix, this exact scenario (a synthetic, never-issued token's cache
   entry, planted directly in Redis by an unrelated party) was accepted
   and returned as a trusted identity.

Mirrors the FakeAsyncRedis pattern from omnibioai-iam-client's own
tests/conftest.py (`raw_set` = a forger planting a value with independent
Redis write access, bypassing AsyncIAMClient entirely) rather than
inventing a new one.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class FakeAsyncRedis:
    """Minimal dict-backed stand-in for redis.asyncio's client, covering
    only the calls AsyncIAMClient's cache layer makes (get/setex/delete)."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._store[key] = value

    async def delete(self, key: str):
        self._store.pop(key, None)

    def raw_set(self, key: str, value: str) -> None:
        """Plant a value directly, bypassing AsyncIAMClient entirely --
        simulates an attacker (or another container on the shared,
        unauthenticated Redis network) with independent write access."""
        self._store[key] = value


def _make_client(fake_redis: FakeAsyncRedis, **kwargs):
    mock_http = AsyncMock()
    with patch("iam_client.client.redis") as mock_redis_module, \
         patch("iam_client.client.httpx") as mock_httpx:
        mock_redis_module.from_url.return_value = fake_redis
        mock_httpx.AsyncClient.return_value = mock_http
        from iam_client.client import AsyncIAMClient

        client = AsyncIAMClient(
            base_url="http://test-iam", redis_url="redis://localhost", **kwargs
        )
        client.redis = fake_redis
        client.http = mock_http
    return client, mock_http


class TestVerifyAndAuthorizeConfiguresCacheSecret:
    """Proves the config -> AsyncIAMClient wiring itself, independent of
    the crypto (covered below) -- this is the one line that could
    silently regress (typo'd kwarg name, wrong config field, or a future
    edit that drops it) while every existing mocked test stays green."""

    @pytest.fixture(autouse=True)
    def _registry_root(self, monkeypatch):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")

    def test_cache_secret_kwarg_matches_configured_jwt_secret(self, monkeypatch):
        import omnibioai_model_registry.auth as auth_mod
        from iam_client.models import UserContext

        monkeypatch.setenv("JWT_SECRET", "hipaa-p1-5-test-secret")
        monkeypatch.setenv("IAM_URL", "http://iam.test")

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(
            return_value=UserContext(
                user_id="1", email="u@test.com", roles=[], permissions=["model.use"],
                valid=True, org_id="org-1",
            )
        )
        mock_client.http.aclose = AsyncMock()
        mock_ctor = MagicMock(return_value=mock_client)
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", mock_ctor)
        monkeypatch.setattr(auth_mod, "AuditClient", MagicMock(return_value=MagicMock()))

        asyncio.run(auth_mod.verify_and_authorize("sometoken", action="model_access"))

        assert mock_ctor.call_count == 1
        _, kwargs = mock_ctor.call_args
        assert kwargs.get("cache_secret") == "hipaa-p1-5-test-secret"

    def test_empty_jwt_secret_is_passed_through_not_silently_defaulted(self, monkeypatch):
        """If JWT_SECRET is unset in a given deployment, cache_secret must
        still be whatever cfg.jwt_secret resolves to (empty string here) --
        this test only guards against a future edit accidentally hardcoding
        a different value or skipping the kwarg for the falsy case; it is
        not a statement that an empty secret is itself acceptable in
        production (a deployment running with JWT_SECRET unset has larger
        problems than this cache)."""
        import omnibioai_model_registry.auth as auth_mod
        from iam_client.models import UserContext

        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.setenv("IAM_URL", "http://iam.test")

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(
            return_value=UserContext(
                user_id="1", email="u@test.com", roles=[], permissions=["model.use"],
                valid=True, org_id="org-1",
            )
        )
        mock_client.http.aclose = AsyncMock()
        mock_ctor = MagicMock(return_value=mock_client)
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", mock_ctor)
        monkeypatch.setattr(auth_mod, "AuditClient", MagicMock(return_value=MagicMock()))

        asyncio.run(auth_mod.verify_and_authorize("sometoken", action="model_access"))

        _, kwargs = mock_ctor.call_args
        assert kwargs.get("cache_secret") == ""


class TestIAMClientCacheIntegrity:
    """Exercises the real (unmocked) AsyncIAMClient cache layer this
    service now configures with cache_secret -- proves the actual
    trust-boundary behavior, not just that a kwarg was passed."""

    SECRET = "shared-jwt-secret-for-this-deployment"

    def test_forged_unsigned_entry_rejected_when_secret_configured(self):
        fake_redis = FakeAsyncRedis()
        client, _ = _make_client(fake_redis, cache_secret=self.SECRET)

        token = "synthetic-never-issued-token"
        forged = json.dumps({
            "user_id": "attacker", "email": "attacker@evil.example",
            "roles": ["admin"], "permissions": ["model.use", "model.resolve_ownership"],
            "org_id": "org-victim", "org_role": [], "schema_version": 1, "valid": True,
        })
        # Planted directly, bypassing AsyncIAMClient -- no MAC prefix at
        # all, the exact pre-hardening/unsigned shape.
        fake_redis.raw_set(client._cache_key(token), forged)

        result = asyncio.run(client.get_cached_user(token))

        assert result is None, "an unsigned forged entry must never be trusted"

    def test_tampered_signed_entry_rejected(self):
        fake_redis = FakeAsyncRedis()
        client, _ = _make_client(fake_redis, cache_secret=self.SECRET)
        token = "some-real-token"
        body = json.dumps({
            "user_id": "1", "email": "u@test.com", "roles": [], "permissions": ["model.use"],
            "org_id": "org-1", "org_role": [], "schema_version": 1, "valid": True,
        })
        mac = client._sign_cache_entry(token, body)
        # Flip the org_id after signing -- same MAC, different body.
        tampered_body = body.replace('"org-1"', '"org-attacker"')
        fake_redis.raw_set(client._cache_key(token), f"{mac}:{tampered_body}")

        result = asyncio.run(client.get_cached_user(token))

        assert result is None, "a signed-but-tampered entry must be rejected, not trusted"

    def test_malformed_entry_treated_as_cache_miss(self):
        fake_redis = FakeAsyncRedis()
        client, _ = _make_client(fake_redis, cache_secret=self.SECRET)
        token = "some-token"
        fake_redis.raw_set(client._cache_key(token), "not-json-and-no-colon-separator-garbage")

        result = asyncio.run(client.get_cached_user(token))

        assert result is None

    def test_entry_signed_with_wrong_secret_rejected(self):
        """Models a consumer sharing this Redis instance without this
        deployment's secret (or a pre-rollout consumer still using a
        stale/default secret) planting a MAC this client can't verify."""
        fake_redis = FakeAsyncRedis()
        victim, _ = _make_client(fake_redis, cache_secret=self.SECRET)
        attacker, _ = _make_client(fake_redis, cache_secret="attacker-does-not-know-this")

        token = "shared-token"
        body = json.dumps({
            "user_id": "attacker", "email": "a@evil.example", "roles": ["admin"],
            "permissions": ["model.resolve_ownership"], "org_id": None,
            "org_role": [], "schema_version": 1, "valid": True,
        })
        # Attacker signs with their own key and writes to the shared store.
        asyncio.run(attacker.set_cache(token, json.loads(body)))

        result = asyncio.run(victim.get_cached_user(token))

        assert result is None, "a MAC produced with a different secret must never verify"

    def test_valid_signed_entry_written_by_this_client_is_accepted(self):
        fake_redis = FakeAsyncRedis()
        client, _ = _make_client(fake_redis, cache_secret=self.SECRET)
        token = "legit-token"
        user = {
            "user_id": "1", "email": "u@test.com", "roles": ["member"],
            "permissions": ["model.use"], "org_id": "org-1", "org_role": [],
            "schema_version": 1, "valid": True,
        }

        asyncio.run(client.set_cache(token, user))
        result = asyncio.run(client.get_cached_user(token))

        assert result is not None
        assert result.user_id == "1"
        assert result.permissions == ["model.use"]

    def test_cache_miss_falls_through_to_authoritative_validation(self):
        """No cache entry at all -- must still resolve identity via the
        normal /auth/validate round trip, never treated as an error."""
        fake_redis = FakeAsyncRedis()
        client, mock_http = _make_client(fake_redis, cache_secret=self.SECRET)
        token = "brand-new-token"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "valid": True, "user_id": 7, "email": "real@test.com",
            "roles": ["member"], "permissions": ["model.use"], "org_id": 3,
        }
        mock_http.post = AsyncMock(return_value=mock_response)

        result = asyncio.run(client.validate_remote(token))

        assert result is not None
        assert result.user_id == "7"
        # And the now-cached entry is signed, not planted verbatim.
        cached_raw = asyncio.run(fake_redis.get(client._cache_key(token)))
        mac, sep, body = cached_raw.partition(":")
        assert sep == ":"
        assert mac == client._sign_cache_entry(token, body)

    def test_attacker_chosen_permissions_never_become_trusted_identity(self):
        """End-to-end: an attacker plants a forged entry granting
        themselves model.resolve_ownership (a privileged, narrowly-scoped
        permission -- see auth.py's MODEL_RESOLVE_OWNERSHIP_PERMISSION).
        With cache_secret configured, get_cached_user must reject it
        outright rather than ever returning that attacker-chosen
        permission set."""
        fake_redis = FakeAsyncRedis()
        client, _ = _make_client(fake_redis, cache_secret=self.SECRET)
        token = "attacker-chosen-token-string"
        forged = json.dumps({
            "user_id": "attacker", "email": "attacker@evil.example",
            "roles": ["org_admin"], "permissions": ["model.resolve_ownership"],
            "org_id": "any-org-attacker-wants", "org_role": [],
            "schema_version": 1, "valid": True,
        })
        fake_redis.raw_set(client._cache_key(token), forged)

        result = asyncio.run(client.get_cached_user(token))

        assert result is None
