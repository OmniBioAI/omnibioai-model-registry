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

PR14.3-4: operational visibility via structured logging only --
deliberately no new metrics dependency introduced into this repo (it
has no existing metrics abstraction, unlike Workbench's plugins/shared/
metrics.py or RAG's already-running prometheus_client), to keep this
PR's footprint small and avoid a third, inconsistent observability
pattern across the three producer repos. Every call to
extra={"usage_event_state": ...} uses the same field name so these log
lines are consistently queryable/aggregable by whatever log platform
this deployment already uses, without this repo needing to run its own
metrics endpoint.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SERVICE = "model.registry"


def _client():
    from usage_client import UsageClient

    return UsageClient()


def emit_model_registered(*, organization_id, user_id=None, trace_id=None) -> None:
    logger.info(
        "usage_emit: attempting model.registered",
        extra={"usage_event_state": "attempted", "resource": "model.register"},
    )
    if organization_id is None:
        logger.warning(
            "usage_emit: skipping model.registered -- no organization_id",
            extra={"usage_event_state": "skipped_missing_org", "resource": "model.register"},
        )
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
        logger.info(
            "usage_emit: model.registered emitted",
            extra={"usage_event_state": "succeeded", "resource": "model.register"},
        )
    except Exception:
        logger.warning(
            "usage_emit: failed to emit model.registered",
            extra={"usage_event_state": "failed_exception", "resource": "model.register"},
            exc_info=True,
        )
