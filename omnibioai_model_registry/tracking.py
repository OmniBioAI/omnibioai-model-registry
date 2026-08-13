"""
Pure DB read/write functions for MLflow-style run tracking.

Every function accepts `conn` as its first argument (connection-injection pattern)
so they can be exercised in unit tests with a real or mock connection without
touching module-level state.

All timestamps are stored as naive UTC datetimes (MySQL DATETIME has no tz).

PHASE 2C -- ORGANIZATION-SCOPED RUN TRACKING
==============================================

`organization_id`/`enforce_ownership` kwargs below mirror the exact
idiom Phase 2B established in api.py's ModelRegistry methods
(`enforce_ownership`/`requesting_org_id`) -- default False so CLI/direct
Python API callers with no IAM identity at all are unaffected, matching
Phase 2A/2B's own precedent. Only omnibioai_model_registry/service/app/
main.py's HTTP routes pass True, with organization_id/requesting_org_id
always the caller's already-verified UserContext.org_id (never a header,
body field, or query/path parameter).

WHY `omr_runs` GETS ITS OWN organization_id COLUMN (NOT derived from
ownership.json): a run's task/model_name has no required correspondence
to any registered model -- logging metrics/params *during* an
experiment, then registering the resulting model *afterward*, is the
whole point of this tracking layer, so ownership.json frequently won't
exist yet (or ever) for a given run's task/model_name. Model ownership
(ownership.json, Phase 2A) remains the single source of truth for
*model* ownership; this column is the equivalent, independent source of
truth for *run* ownership -- the two are deliberately not merged, and
neither can override the other. See check_run_ownership() below, and
db.py's _ALTER_DDL comment for the legacy-row migration reasoning.

`omr_params`/`omr_metrics`/`omr_tags` do NOT get their own
organization_id column -- they inherit their run's boundary via the
run_id foreign key (never duplicated onto every child row, which would
itself be a second, possibly-diverging source of truth for the same
run's org). `omr_version_tags` doesn't get one either: it's keyed
directly by (task, model_name, version) and IS a real model's
version-level metadata, so its boundary is the SAME authoritative
ownership.json Phase 2A/2B already established for models -- see
set_version_tag()/get_version_tags() below, which call
ownership.check_model_ownership() directly rather than inventing a
second ownership record for the same model.

ANTI-ENUMERATION: every denial below raises the exact same
`ModelNotFound(f"Run not found: {run_id}")` (or the model-ownership
equivalent for version tags) already raised for a genuinely missing
run/model in this module before Phase 2C -- a cross-org caller cannot
distinguish "doesn't exist" from "not yours".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .errors import ModelNotFound
from .ownership import check_model_ownership


# --------------------------------------------------------------------------- #
# Phase 2C -- run ownership decision (mirrors ownership.check_model_ownership) #
# --------------------------------------------------------------------------- #

STATUS_OWNED = "owned"
STATUS_UNOWNED = "unowned"
STATUS_LEGACY_UNOWNED = "legacy_unowned"

REASON_OWNED_BY_CALLER = "owned_by_caller"
REASON_OPEN_MODE_MATCH = "open_mode_match"
REASON_RUN_NOT_FOUND = "run_not_found"
REASON_LEGACY_UNOWNED = "legacy_unowned"
REASON_OWNED_BY_OTHER_ORG = "owned_by_other_org"


@dataclass(frozen=True)
class RunOwnershipCheckResult:
    allowed: bool
    reason: str
    # Whether the run row exists at all. A write call site (_ensure_run,
    # which pre-creates a run on its first log call) checks THIS rather
    # than `allowed` to distinguish "genuinely new run, nothing to check
    # against yet" (exists=False; always fine to create, attributed to
    # the caller) from a real denial (exists=True, allowed=False). Read
    # call sites just check `allowed` -- not-found and not-yours both
    # deny identically.
    exists: bool


def _evaluate_run_ownership(
    organization_id: Optional[str],
    ownership_status: str,
    *,
    requesting_org_id: Optional[str],
) -> RunOwnershipCheckResult:
    """Pure decision (no DB access) given an already-fetched
    organization_id/ownership_status pair -- lets get_run() reuse the row
    it already SELECTed instead of a second round-trip. See
    check_run_ownership() for the DB-touching wrapper used where nothing
    else already fetched the row (the write path, get_metric_history)."""
    if ownership_status == STATUS_LEGACY_UNOWNED:
        return RunOwnershipCheckResult(False, REASON_LEGACY_UNOWNED, exists=True)

    if requesting_org_id is not None and requesting_org_id == organization_id:
        return RunOwnershipCheckResult(True, REASON_OWNED_BY_CALLER, exists=True)

    if requesting_org_id is None and organization_id is None:
        # Both sides have no org context: the pre-existing AUTH_ENABLED=
        # false / fully-open dev-test mode, unchanged by Phase 2C. Not
        # the same as legacy_unowned -- this run was created in this
        # same no-org mode, not merely discovered without attribution.
        return RunOwnershipCheckResult(True, REASON_OPEN_MODE_MATCH, exists=True)

    return RunOwnershipCheckResult(False, REASON_OWNED_BY_OTHER_ORG, exists=True)


def check_run_ownership(
    conn, run_id: str, *, requesting_org_id: Optional[str]
) -> RunOwnershipCheckResult:
    """The DB-touching counterpart of _evaluate_run_ownership() -- used by
    _ensure_run()'s write-path pre-check and by get_metric_history()
    (neither otherwise fetches the omr_runs row). A run that doesn't
    exist at all is reported as allowed=False/exists=False; callers that
    mean "creatable" (only _ensure_run) explicitly branch on `.exists`
    rather than `.allowed` -- see its docstring above."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT organization_id, ownership_status FROM omr_runs WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        return RunOwnershipCheckResult(False, REASON_RUN_NOT_FOUND, exists=False)
    return _evaluate_run_ownership(
        row.get("organization_id"),
        row.get("ownership_status") or STATUS_UNOWNED,
        requesting_org_id=requesting_org_id,
    )


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ensure_run(
    conn,
    run_id: str,
    task: str,
    model_name: str,
    actor: Optional[str] = None,
    *,
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    """INSERT IGNORE — idempotent; safe to call before every log mutation.

    Phase 2C: when enforce_ownership=True, a run_id that already belongs
    to a different organization is denied before any write is attempted
    (ModelNotFound, matching every other Phase 2B/2C denial's message
    shape). A genuinely new run_id is never blocked here -- there is
    nothing to check against yet, same as Phase 2A/2B's brand-new-model
    carve-out -- and is recorded as owned by `requesting_org_id`
    (status="owned"), or "unowned" if that's None, mirroring
    ensure_model_ownership()'s exact organization_id-present-or-not
    logic. Attribution is recorded whenever requesting_org_id is passed,
    independent of enforce_ownership -- the flag only gates the
    pre-check, not whether attribution itself happens (same as
    register_model's organization_id parameter).
    """
    if enforce_ownership:
        check = check_run_ownership(conn, run_id, requesting_org_id=requesting_org_id)
        if check.exists and not check.allowed:
            raise ModelNotFound(f"Run not found: {run_id}")

    ownership_status = STATUS_OWNED if requesting_org_id else STATUS_UNOWNED
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT IGNORE INTO omr_runs
                (run_id, task, model_name, status, started_at, actor, organization_id, ownership_status)
            VALUES (%s, %s, %s, 'running', %s, %s, %s, %s)
            """,
            (run_id, task, model_name, _now_utc(), actor, requesting_org_id, ownership_status),
        )


# --------------------------------------------------------------------------- #
# Run lifecycle                                                                 #
# --------------------------------------------------------------------------- #

def create_run(
    conn,
    task: str,
    model_name: str,
    run_id: str,
    actor: Optional[str] = None,
    *,
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> dict:
    """Ensure the run row exists and return its current state."""
    _ensure_run(
        conn, run_id, task, model_name, actor,
        requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership,
    )
    return get_run(conn, run_id, requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership)


def finish_run(
    conn, run_id: str, *, requesting_org_id: Optional[str] = None, enforce_ownership: bool = False,
) -> None:
    """Mark a run as finished. Raises ModelNotFound if run_id is unknown
    or (Phase 2C, enforce_ownership=True) belongs to a different org --
    denied before the UPDATE, so a cross-org caller can't even flip its
    status."""
    if enforce_ownership:
        check = check_run_ownership(conn, run_id, requesting_org_id=requesting_org_id)
        if not check.allowed:
            raise ModelNotFound(f"Run not found: {run_id}")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE omr_runs SET status = 'finished', finished_at = %s WHERE run_id = %s",
            (_now_utc(), run_id),
        )
        if cur.rowcount == 0:
            raise ModelNotFound(f"Run not found: {run_id}")


# --------------------------------------------------------------------------- #
# Params                                                                        #
# --------------------------------------------------------------------------- #

def log_param(
    conn,
    run_id: str,
    key: str,
    value: Any,
    *,
    task: str = "",
    model_name: str = "",
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    _ensure_run(
        conn, run_id, task, model_name,
        requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_params (run_id, key_name, value_text)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE value_text = VALUES(value_text)
            """,
            (run_id, key, json.dumps(value)),
        )


def log_params(
    conn,
    run_id: str,
    params: Dict[str, Any],
    *,
    task: str = "",
    model_name: str = "",
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    for k, v in params.items():
        log_param(
            conn, run_id, k, v, task=task, model_name=model_name,
            requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership,
        )


# --------------------------------------------------------------------------- #
# Metrics                                                                       #
# --------------------------------------------------------------------------- #

def log_metric(
    conn,
    run_id: str,
    key: str,
    value: float,
    step: int = 0,
    *,
    task: str = "",
    model_name: str = "",
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    _ensure_run(
        conn, run_id, task, model_name,
        requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_metrics (run_id, key_name, value, step, ts_utc)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, key, value, step, _now_utc()),
        )


def log_metrics(
    conn,
    run_id: str,
    metrics: Dict[str, float],
    step: int = 0,
    *,
    task: str = "",
    model_name: str = "",
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    for k, v in metrics.items():
        log_metric(
            conn, run_id, k, v, step, task=task, model_name=model_name,
            requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership,
        )


# --------------------------------------------------------------------------- #
# Tags (run-level)                                                              #
# --------------------------------------------------------------------------- #

def set_tag(
    conn,
    run_id: str,
    key: str,
    value: str,
    *,
    task: str = "",
    model_name: str = "",
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    _ensure_run(
        conn, run_id, task, model_name,
        requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_tags (run_id, key_name, value_text)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE value_text = VALUES(value_text)
            """,
            (run_id, key, value),
        )


def set_tags(
    conn,
    run_id: str,
    tags: Dict[str, str],
    *,
    task: str = "",
    model_name: str = "",
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    for k, v in tags.items():
        set_tag(
            conn, run_id, k, v, task=task, model_name=model_name,
            requesting_org_id=requesting_org_id, enforce_ownership=enforce_ownership,
        )


# --------------------------------------------------------------------------- #
# Run query                                                                     #
# --------------------------------------------------------------------------- #

def get_run(
    conn, run_id: str, *, requesting_org_id: Optional[str] = None, enforce_ownership: bool = False,
) -> dict:
    """
    Return a full run snapshot: metadata + params + tags + metrics_summary.

    metrics_summary keeps the latest value (highest step, then latest insert)
    for each key — same semantics as MLflow's RunData.metrics.

    Phase 2C: when enforce_ownership=True, the ownership decision reuses
    the row this function already SELECTs (organization_id/
    ownership_status, both present on every omr_runs row since the
    Phase 2C migration) rather than a second round-trip -- see
    _evaluate_run_ownership(). A cross-org or legacy_unowned run raises
    the exact same ModelNotFound a genuinely missing run_id already did.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM omr_runs WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise ModelNotFound(f"Run not found: {run_id}")
        if enforce_ownership:
            check = _evaluate_run_ownership(
                row.get("organization_id"),
                row.get("ownership_status") or STATUS_UNOWNED,
                requesting_org_id=requesting_org_id,
            )
            if not check.allowed:
                raise ModelNotFound(f"Run not found: {run_id}")

        cur.execute(
            "SELECT key_name, value_text FROM omr_params WHERE run_id = %s",
            (run_id,),
        )
        params: Dict[str, Any] = {}
        for r in cur.fetchall():
            try:
                params[r["key_name"]] = json.loads(r["value_text"])
            except (json.JSONDecodeError, TypeError):
                params[r["key_name"]] = r["value_text"]

        cur.execute(
            "SELECT key_name, value_text FROM omr_tags WHERE run_id = %s",
            (run_id,),
        )
        tags: Dict[str, str] = {r["key_name"]: r["value_text"] for r in cur.fetchall()}

        # Highest-step value per key (latest-insert tie-breaker via id DESC)
        cur.execute(
            """
            SELECT key_name, value, step
            FROM omr_metrics
            WHERE run_id = %s
            ORDER BY step DESC, id DESC
            """,
            (run_id,),
        )
        metrics_summary: Dict[str, float] = {}
        for r in cur.fetchall():
            if r["key_name"] not in metrics_summary:
                metrics_summary[r["key_name"]] = r["value"]

    started = row["started_at"]
    finished = row["finished_at"]
    return {
        "run_id": row["run_id"],
        "task": row["task"],
        "model_name": row["model_name"],
        "status": row["status"],
        "started_at": started.isoformat() if started else None,
        "finished_at": finished.isoformat() if finished else None,
        "actor": row["actor"],
        "params": params,
        "tags": tags,
        "metrics_summary": metrics_summary,
    }


def get_metric_history(
    conn, run_id: str, key: str, *, requesting_org_id: Optional[str] = None, enforce_ownership: bool = False,
) -> List[dict]:
    """Return all (value, step, ts_utc) rows for a metric key, ordered by
    step ASC.

    Phase 2C: unlike get_run(), this function never otherwise touches
    omr_runs, so enforce_ownership=True costs a dedicated
    check_run_ownership() round-trip -- kept anyway (rather than trusting
    every caller to have already validated the run via get_run() in the
    same request) per the "don't assume route-level checks are
    sufficient, trace the underlying storage functions" principle.
    """
    if enforce_ownership:
        check = check_run_ownership(conn, run_id, requesting_org_id=requesting_org_id)
        if not check.allowed:
            raise ModelNotFound(f"Run not found: {run_id}")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT value, step, ts_utc
            FROM omr_metrics
            WHERE run_id = %s AND key_name = %s
            ORDER BY step ASC, id ASC
            """,
            (run_id, key),
        )
        return [
            {
                "value": r["value"],
                "step": r["step"],
                "ts_utc": r["ts_utc"].isoformat() if r["ts_utc"] else None,
            }
            for r in cur.fetchall()
        ]


def list_runs(
    conn, task: str, model_name: str, *, requesting_org_id: Optional[str] = None, enforce_ownership: bool = False,
) -> List[dict]:
    """List all runs for a (task, model_name), newest first.

    Phase 2C: when enforce_ownership=True, the org filter is applied in
    the WHERE clause itself -- an other-org or legacy_unowned run is
    never fetched at all, not fetched-then-hidden in the response
    (matches Phase 2B's /v1/models precedent). legacy_unowned rows never
    match either branch (their ownership_status is neither 'owned' nor
    'unowned'), so they're excluded unconditionally, same as
    check_run_ownership()'s REASON_LEGACY_UNOWNED denial.
    """
    with conn.cursor() as cur:
        if enforce_ownership:
            cur.execute(
                """
                SELECT run_id, status, started_at, finished_at
                FROM omr_runs
                WHERE task = %s AND model_name = %s
                  AND (
                        (ownership_status = %s AND organization_id = %s)
                        OR (ownership_status = %s AND organization_id IS NULL AND %s IS NULL)
                      )
                ORDER BY started_at DESC
                """,
                (
                    task, model_name,
                    STATUS_OWNED, requesting_org_id,
                    STATUS_UNOWNED, requesting_org_id,
                ),
            )
        else:
            cur.execute(
                """
                SELECT run_id, status, started_at, finished_at
                FROM omr_runs
                WHERE task = %s AND model_name = %s
                ORDER BY started_at DESC
                """,
                (task, model_name),
            )
        return [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                "finished_at": r["finished_at"].isoformat() if r["finished_at"] else None,
            }
            for r in cur.fetchall()
        ]


# --------------------------------------------------------------------------- #
# Version-level tags                                                            #
# --------------------------------------------------------------------------- #

def set_version_tag(
    conn,
    task: str,
    model_name: str,
    version: str,
    key: str,
    value: str,
    *,
    registry_root=None,
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> None:
    """Phase 2C: unlike omr_runs, omr_version_tags has no organization_id
    of its own -- it's keyed by (task, model_name, version), a real
    model's version, so its boundary is the SAME authoritative
    ownership.json Phase 2A/2B already established (never a second,
    possibly-diverging ownership record for the same model). When
    enforce_ownership=True, registry_root is required -- see
    ownership.check_model_ownership(). This is defense-in-depth: the one
    HTTP call site (PUT /v1/tags, main.py) already enforces model
    ownership via _require_model_ownership() before ever reaching here.
    """
    if enforce_ownership:
        check = check_model_ownership(registry_root, task, model_name, requesting_org_id=requesting_org_id)
        if not check.allowed:
            raise ModelNotFound(f"Model not found: task={task}, model_name={model_name}")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omr_version_tags (task, model_name, version, key_name, value_text)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE value_text = VALUES(value_text)
            """,
            (task, model_name, version, key, value),
        )


def get_version_tags(
    conn,
    task: str,
    model_name: str,
    version: str,
    *,
    registry_root=None,
    requesting_org_id: Optional[str] = None,
    enforce_ownership: bool = False,
) -> Dict[str, str]:
    """See set_version_tag()'s docstring -- same ownership.json-derived
    boundary, no independent organization_id for this table."""
    if enforce_ownership:
        check = check_model_ownership(registry_root, task, model_name, requesting_org_id=requesting_org_id)
        if not check.allowed:
            raise ModelNotFound(f"Model not found: task={task}, model_name={model_name}")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT key_name, value_text
            FROM omr_version_tags
            WHERE task = %s AND model_name = %s AND version = %s
            """,
            (task, model_name, version),
        )
        return {r["key_name"]: r["value_text"] for r in cur.fetchall()}
