# File: omnibioai_model_registry/ownership.py
"""
Phase 2A -- organization-ownership foundation for models/versions.

SOURCE OF TRUTH
----------------
The filesystem is authoritative for model/version identity today:
register_model()/resolve_model()/promote_model() (api.py) operate
entirely on task/models/model_name/versions/version directories. MySQL's
omr_* tables (db.py) are an optional, DB_HOST-gated side-store for
experiment-tracking metrics only (runs/params/metrics/tags) and never
represent models or versions themselves -- there is no "models" or
"versions" table, and the service is fully functional with DB_HOST unset
("filesystem-only mode", see README). Anchoring ownership in MySQL would
therefore make ownership unavailable in a fully-supported deployment mode
and would create two independent sources of truth for the same question
("who owns this model") -- exactly what the discovery audit warned
against. Ownership is instead recorded as a durable, write-once JSON file
alongside the other durable model-level state this repo already trusts
the filesystem for (aliases/*.json, audit/promotions.jsonl):
`ownership.json`, one per model (task/model_name) -- NOT one per version.

WHY MODEL-LEVEL, NOT VERSION-LEVEL
------------------------------------
Versions do not carry an independent ownership field at all. By
construction they inherit the parent model's single ownership.json --
this eliminates the "model=org-A, version=org-B" divergence risk
entirely, rather than merely preventing it with a runtime check.

WHERE organization_id COMES FROM
----------------------------------
Always the caller's already-verified IAM identity
(iam_client.models.UserContext.org_id, obtained via
auth.require_write_auth_with_context -- the same dependency PR14.2B-3
already introduced for usage-event emission). Never accepted from
request body/query/path parameters or any header (X-Organization-ID,
X-Team-ID, ...). This module itself never touches HTTP request data, only
already-resolved values passed in by its callers (api.py, cli/main.py).

WHAT THIS PHASE DOES NOT DO
------------------------------
No query-layer enforcement, no cross-org read/write blocking, no HF-push
ownership check, no model.use redesign, no public/private semantics.
Establishing the record is the entire scope -- see the root README's
Roadmap for what is deferred to Phase 2B/2C/2D.

---

PHASE 2B ADDENDUM -- ENFORCEMENT
==================================

check_model_ownership() below is the single centralized authorization
decision every route/storage-layer function uses, so
`if user.org_id != ownership.organization_id` is never duplicated ad
hoc. It is pure data (no FastAPI/HTTP dependency) -- callers translate
`not result.allowed` into whatever "not found"-shaped response that
specific route already uses for a genuinely missing model, so a denial
is anti-enumerating: indistinguishable from the model not existing at
all. See api.py (resolve_model/promote_model/register_model) and
service/app/main.py (routes that bypass ModelRegistry entirely) for the
call sites.

Matching rule (requesting_org_id is always a verified UserContext.org_id,
never client-supplied):

- status="owned": allowed only if requesting_org_id is not None and
  equals the record's organization_id.
- status="legacy_unowned": always denied, for every caller, including a
  None org_id -- never silently claimed. No admin-reassignment workflow
  exists yet (see README); resolving these is explicitly deferred.
- status="unowned" (AUTH_ENABLED=false at registration time): allowed
  only when requesting_org_id is ALSO None -- i.e. the whole deployment
  is running in the open, no-org dev/test mode this repo has always
  supported, unchanged. A real org_id reaching into a None-org model is
  treated the same as reaching into a different org's model (denied) --
  "unowned" is not "everyone's".
- ownership record does not exist at all (never registered or migrated):
  denied, same as "legacy_unowned" -- this should not normally happen
  post-Phase-2A, but is handled the same safe way if it does.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .audit.audit_log import now_utc_iso
from .package import layout as L
from .storage.localfs import LocalFS

SCHEMA_VERSION = 1

STATUS_OWNED = "owned"
# AUTH_ENABLED=false, or a real identity with no org membership, at the
# moment a genuinely NEW model is registered -- there was no org to
# record, not a guess.
STATUS_UNOWNED = "unowned"
# A model directory that already existed with no ownership.json --
# either pre-Phase-2A, or never touched since. The CURRENT registrant's
# org is deliberately never assigned here.
STATUS_LEGACY_UNOWNED = "legacy_unowned"


@dataclass(frozen=True)
class OwnershipRecord:
    schema_version: int
    task: str
    model_name: str
    organization_id: Optional[str]
    status: str
    registered_by: Optional[str]
    registered_at: Optional[str]
    discovered_at: Optional[str] = None
    note: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def read_ownership(
    registry_root: Path, task: str, model_name: str
) -> Optional[OwnershipRecord]:
    """The persisted ownership record for a model, or None if one has
    never been established (should not happen for any model that has
    been registered against, or migrated, since this PR)."""
    path = L.ownership_path(Path(registry_root), task, model_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # Forward-compatible: unknown extra keys from a future schema_version
    # are dropped rather than raising, known-missing keys fall back to
    # dataclass defaults (discovered_at/note).
    known = {f for f in OwnershipRecord.__dataclass_fields__}
    return OwnershipRecord(**{k: v for k, v in data.items() if k in known})


def ensure_model_ownership(
    backend: LocalFS,
    registry_root: Path,
    task: str,
    model_name: str,
    *,
    organization_id: Optional[str],
    actor: Optional[str],
    model_pre_existing: bool,
) -> OwnershipRecord:
    """Idempotently establishes -- but never overwrites -- the ownership
    record for a model. Call once per register_model() invocation, after
    the model's directories are guaranteed to exist.

    - model_pre_existing=False (a genuinely new task/model_name): the
      primary "registration" path. Ownership is recorded as the
      registering caller's verified organization_id (status="owned"), or
      status="unowned" if the caller has no organization_id -- never
      guessed, never left ambiguous.
    - model_pre_existing=True and no ownership.json yet: a pre-Phase-2A
      (or otherwise never-migrated) model. The CURRENT caller's org is
      deliberately NOT assigned here -- doing so would let any org "claim"
      an existing model just by registering a new version under it. This
      is recorded as status="legacy_unowned" with no organization_id and
      no registered_by, identical in shape and effect to what the
      standalone backfill_legacy_ownership() utility produces for the
      same model.
    - ownership.json already exists (either case above, from an earlier
      call -- including an earlier Phase-2A registration by this or a
      different org): left untouched and read back as-is. Ownership is
      write-once; the organization_id/actor passed to *this* call are
      discarded in that case.

    Concurrency: uses LocalFS.write_once_text (hardlink-based exclusive
    create), so two racing registrations for the same brand-new model
    cannot produce two different ownership records -- the loser reads
    back whatever the winner wrote.
    """
    path = L.ownership_path(Path(registry_root), task, model_name)

    if model_pre_existing:
        candidate = OwnershipRecord(
            schema_version=SCHEMA_VERSION,
            task=task,
            model_name=model_name,
            organization_id=None,
            status=STATUS_LEGACY_UNOWNED,
            registered_by=None,
            registered_at=None,
            discovered_at=now_utc_iso(),
            note=(
                "Pre-existing model discovered with no ownership record; "
                "requires manual assignment (see Phase 2B)."
            ),
        )
    else:
        candidate = OwnershipRecord(
            schema_version=SCHEMA_VERSION,
            task=task,
            model_name=model_name,
            organization_id=organization_id,
            status=STATUS_OWNED if organization_id else STATUS_UNOWNED,
            registered_by=actor,
            registered_at=now_utc_iso(),
        )

    created = backend.write_once_text(path, candidate.to_json())
    if created:
        return candidate

    existing = read_ownership(registry_root, task, model_name)
    if existing is None:
        # Extremely unlikely (file existed for write_once_text but
        # vanished before our read) -- surface rather than silently
        # returning a candidate that was never actually persisted.
        raise RuntimeError(
            f"ownership.json for {task}/{model_name} disappeared between "
            "write and read-back"
        )
    return existing


REASON_OWNED_BY_CALLER = "owned_by_caller"
REASON_OPEN_MODE_MATCH = "open_mode_match"
REASON_MODEL_NOT_FOUND = "model_not_found"
REASON_LEGACY_UNOWNED = "legacy_unowned"
REASON_OWNED_BY_OTHER_ORG = "owned_by_other_org"


@dataclass(frozen=True)
class OwnershipCheckResult:
    allowed: bool
    reason: str
    # The actual record, even when denied -- useful for server-side audit
    # logging (e.g. "org-B tried to touch org-A's model"). Callers MUST
    # NOT expose this to the client on a denial; only `allowed` may
    # influence the HTTP response, and only via the anti-enumerating
    # "not found"-shaped response the route already uses for a genuinely
    # missing model. None only when no record exists at all.
    ownership: Optional[OwnershipRecord]


def check_model_ownership(
    registry_root: Path,
    task: str,
    model_name: str,
    *,
    requesting_org_id: Optional[str],
) -> OwnershipCheckResult:
    """The single centralized Phase 2B authorization decision. See the
    module docstring's "PHASE 2B ADDENDUM" for the full matching rule.

    `requesting_org_id` must already be a verified IAM identity's org_id
    (UserContext.org_id) -- this function has no knowledge of HTTP
    request data, headers, or JWTs, and performs no verification of its
    own. It only ever reads ownership.json; it never writes.
    """
    record = read_ownership(registry_root, task, model_name)
    if record is None:
        return OwnershipCheckResult(False, REASON_MODEL_NOT_FOUND, None)

    if record.status == STATUS_LEGACY_UNOWNED:
        return OwnershipCheckResult(False, REASON_LEGACY_UNOWNED, record)

    # status is STATUS_OWNED or STATUS_UNOWNED here.
    if requesting_org_id is not None and requesting_org_id == record.organization_id:
        return OwnershipCheckResult(True, REASON_OWNED_BY_CALLER, record)

    if requesting_org_id is None and record.organization_id is None:
        # Both sides have no org context: the pre-existing AUTH_ENABLED=
        # false / fully-open dev-test mode, unchanged by Phase 2B. Not
        # the same as legacy_unowned -- this model was created in this
        # same no-org mode, not merely discovered without a record.
        return OwnershipCheckResult(True, REASON_OPEN_MODE_MATCH, record)

    return OwnershipCheckResult(False, REASON_OWNED_BY_OTHER_ORG, record)


def backfill_legacy_ownership(
    registry_root: Path, backend: Optional[LocalFS] = None
) -> dict:
    """Deterministic, repeatable, additive-only migration: walks every
    existing task/model_name directory under `registry_root` and writes
    an explicit status="legacy_unowned" ownership.json for any model that
    doesn't already have one.

    Never invents an organization from actor strings or any other weak
    signal -- actor identity is audit metadata, not a tenant key. Never
    touches versions/aliases/artifacts/audit -- additive only. Safe to
    re-run: already-migrated or already-owned models are left untouched
    (ownership.json is write-once).

    Returns {"scanned": N, "migrated": N, "already_had_ownership": N}.
    """
    backend = backend or LocalFS()
    root = Path(registry_root)
    tasks_root = root / "tasks"
    summary = {"scanned": 0, "migrated": 0, "already_had_ownership": 0}
    if not tasks_root.exists():
        return summary

    for task_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        models_root = task_dir / "models"
        if not models_root.exists():
            continue
        for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
            summary["scanned"] += 1
            task = task_dir.name
            model_name = model_dir.name
            if read_ownership(root, task, model_name) is not None:
                summary["already_had_ownership"] += 1
                continue
            ensure_model_ownership(
                backend,
                root,
                task,
                model_name,
                organization_id=None,
                actor=None,
                model_pre_existing=True,
            )
            summary["migrated"] += 1
    return summary
