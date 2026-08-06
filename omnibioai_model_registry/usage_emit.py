"""
omnibioai_model_registry/usage_emit.py

PR14.2B-3: thin wrapper around omnibioai-usage-client for Model
Registry's usage events. Fail-open, defensively: UsageClient.
emit_usage_event() already swallows Redis/connection errors internally
and returns False rather than raising, but every call here is
additionally wrapped in a bare except Exception -- a usage-emission
failure must never fail model registration, the same posture
omnibioai's plugins/workflow_runner/usage_emit.py and omnibioai-rag's
ragbio/usage_emit.py take.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SERVICE = "model.registry"


def _client():
    from usage_client import UsageClient

    return UsageClient()


def emit_model_registered(*, organization_id, user_id=None, trace_id=None) -> None:
    if organization_id is None:
        logger.warning("usage_emit: skipping model.registered -- no organization_id")
        return

    try:
        _client().emit_usage_event(
            organization_id=organization_id,
            service=_SERVICE,
            resource="model.register",
            action="registered",
            quantity=1,
            unit="count",
            user_id=user_id,
            trace_id=trace_id,
        )
    except Exception:
        logger.warning("usage_emit: failed to emit model.registered", exc_info=True)
