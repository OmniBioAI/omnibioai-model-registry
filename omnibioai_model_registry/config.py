# File: omnibioai_model_registry/config.py
import os
from dataclasses import dataclass

from .errors import RegistryNotConfigured


@dataclass(frozen=True)
class RegistryConfig:
    root: str
    backend: str = "localfs"  # future: s3, azure_blob
    strict_verify: bool = True
    jwt_secret: str = ""       # from JWT_SECRET env var
    iam_url: str = ""          # from IAM_URL env var
    iam_redis_url: str = "redis://localhost:6379/0"  # from IAM_REDIS_URL, falling back to REDIS_URL
    audit_url: str = ""        # from AUDIT_URL env var
    auth_enabled: bool = False  # from AUTH_ENABLED env var ("true"/"false")
    # Security PR (filesystem path hardening): optional, operator-chosen
    # allowlist of server-local directory trees `register_model()`'s
    # `artifacts_dir` is permitted to be under. Comma-separated absolute
    # paths, from OMNIBIOAI_MODEL_REGISTRY_ARTIFACTS_ALLOWED_ROOTS.
    # Empty (the default) means unrestricted -- today's existing
    # behavior, preserved on purpose: this service has no established
    # convention for where training-output directories live on a given
    # deployment's filesystem, so picking a mandatory default here would
    # be inventing a policy rather than enforcing one a deployment
    # already has. Always enforced regardless of this setting:
    # artifacts_dir may never resolve to a location inside `root` itself
    # -- see api.py's register_model().
    artifacts_allowed_roots: tuple[str, ...] = ()


def load_config() -> RegistryConfig:
    root = (
        os.getenv("OMNIBIOAI_MODEL_REGISTRY_ROOT") or os.getenv("REGISTRY_ROOT") or ""
    ).strip()
    if not root:
        raise RegistryNotConfigured(
            "OMNIBIOAI_MODEL_REGISTRY_ROOT is not set. "
            "Example: export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/Desktop/machine/local_registry/model_registry"
        )
    backend = (
        os.getenv("OMNIBIOAI_MODEL_REGISTRY_BACKEND", "localfs").strip() or "localfs"
    )
    strict_verify = os.getenv(
        "OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "1"
    ).strip() not in {"0", "false", "False"}
    jwt_secret = os.getenv("JWT_SECRET", "").strip()
    iam_url = os.getenv("IAM_URL", "").strip()
    # Same fallback precedence as omnibioai-lims' core/authentication.py:
    # a dedicated IAM_REDIS_URL if the deployment wants the IAM client's
    # cache on a different Redis instance/db than this service's own
    # REDIS_URL (if any); otherwise share it.
    iam_redis_url = (
        os.getenv("IAM_REDIS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    ).strip()
    audit_url = os.getenv("AUDIT_URL", "").strip()
    auth_enabled = os.getenv("AUTH_ENABLED", "").strip().lower() == "true"
    artifacts_allowed_roots = tuple(
        p.strip()
        for p in os.getenv("OMNIBIOAI_MODEL_REGISTRY_ARTIFACTS_ALLOWED_ROOTS", "").split(",")
        if p.strip()
    )
    return RegistryConfig(
        root=root,
        backend=backend,
        strict_verify=strict_verify,
        jwt_secret=jwt_secret,
        iam_url=iam_url,
        iam_redis_url=iam_redis_url,
        audit_url=audit_url,
        auth_enabled=auth_enabled,
        artifacts_allowed_roots=artifacts_allowed_roots,
    )
