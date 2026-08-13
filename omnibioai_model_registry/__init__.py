# File: omnibioai_model_registry/__init__.py
from .api import (
    ModelRegistry,
    promote_model,
    register_model,
    resolve_model,
    verify_model_ref,
)
from .ownership import (
    OwnershipCheckResult,
    OwnershipRecord,
    backfill_legacy_ownership,
    check_model_ownership,
    read_ownership,
)
from .run import RunLogger

__all__ = [
    "ModelRegistry",
    "register_model",
    "resolve_model",
    "promote_model",
    "verify_model_ref",
    "RunLogger",
    "OwnershipRecord",
    "OwnershipCheckResult",
    "read_ownership",
    "check_model_ownership",
    "backfill_legacy_ownership",
]
