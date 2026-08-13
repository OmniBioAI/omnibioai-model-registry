from __future__ import annotations

import json
import os
from fastapi.middleware.cors import CORSMiddleware
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from iam_client.models import UserContext

from omnibioai_model_registry import (
    ModelRegistry,
    promote_model,
    register_model,
    resolve_model,
    verify_model_ref,
)
from omnibioai_model_registry.errors import ModelRegistryError, PathTraversalError, RegistryNotConfigured
from omnibioai_model_registry.config import load_config
from omnibioai_model_registry.auth import (
    _actor_identifier,
    require_auth_with_context,
    require_write_auth_with_context,
)
from omnibioai_model_registry.ownership import check_model_ownership
from omnibioai_model_registry.package import layout as L
from omnibioai_model_registry.audit_client import AuditClient
from omnibioai_model_registry.hf_routes import router as hf_router
from omnibioai_model_registry.usage_emit import emit_model_registered
import logging as _logging

_audit = AuditClient(os.environ.get("AUDIT_URL", ""))
_logger = _logging.getLogger(__name__)


def _emit_usage_safe(fn, /, **kwargs) -> None:
    """PR14.2B-3: last line of defense around every usage-event emission
    call site in this file. usage_emit.py's own functions already catch
    everything reasonably catchable (Redis errors) internally -- this
    covers the unexpected case (a bug in that module itself). Usage
    emission must always be fail-open: never let it fail model
    registration.
    """
    try:
        fn(**kwargs)
    except Exception:
        _logger.warning("usage_emit call failed unexpectedly", exc_info=True)

try:
    from omnibioai_model_registry import db as _db
    from omnibioai_model_registry import tracking as _tracking
except ImportError:
    _db = None  # type: ignore
    _tracking = None  # type: ignore

# Keep API version in sync with package version
APP_VERSION = pkg_version("omnibioai-model-registry")
DEFAULT_PREFIX = "/v1"

app = FastAPI(title="OmniBioAI Model Registry Service", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5181",
        "http://localhost:5173",
        "http://127.0.0.1:5181",
        "http://127.0.0.1:5173",
        "http://localhost:5182",
        "http://127.0.0.1:5182",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.exception_handler(PathTraversalError)
def _handle_path_traversal_error(request: Request, exc: PathTraversalError) -> JSONResponse:
    """Global safety net for path_safety.py's PathTraversalError.

    Most routes already wrap their filesystem calls in
    `except Exception as e: _handle_registry_error(e)`, which already
    maps any ModelRegistryError (PathTraversalError included) to this
    same 400 shape -- this handler exists for the handful of routes
    (aliases/compare, which construct paths directly via package.layout
    outside of any local try/except) that don't, so a rejected
    identifier is *always* a safe, generic 400 everywhere in this
    service, never an unhandled 500 with a stack trace. `str(exc)`
    already contains only a field name (see path_safety.py) -- never a
    resolved filesystem path, request value, or other internal detail.
    """
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ==========================================================
# Shared registry instance for UI/API
# ==========================================================
registry = ModelRegistry.from_env()
app.include_router(hf_router, prefix=f"{DEFAULT_PREFIX}/hf", tags=["huggingface"])


@app.on_event("startup")
def _startup() -> None:
    _log = _logging.getLogger(__name__)
    try:
        cfg = load_config()
        if cfg.auth_enabled:
            _log.info("Auth enabled — JWT validation active")
        else:
            _log.warning("Auth disabled — running in open mode (AUTH_ENABLED != true)")
    except Exception:
        pass
    if _db is None:
        return
    try:
        conn = _db.get_connection()
        _db.init_tables(conn)
        conn.close()
        _log.info("DB tables initialised")
    except Exception as exc:
        _log.warning("DB init skipped (filesystem-only mode active): %s", exc)


# -------------------------
# Request/Response models
# -------------------------

class RegisterRequest(BaseModel):
    task: str
    model_name: str
    version: str
    artifacts_dir: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    set_alias: Optional[str] = "latest"
    actor: Optional[str] = None
    reason: Optional[str] = "api register"


class RegisterResponse(BaseModel):
    ok: bool
    task: str
    model_name: str
    version: str
    package_path: str
    alias_set: Optional[str] = None
    # Phase 2A: server-derived ownership, never accepted from the
    # request -- RegisterRequest above deliberately has no
    # organization_id field. See ownership.py.
    organization_id: Optional[str] = None
    ownership_status: Optional[str] = None


class PromoteRequest(BaseModel):
    task: str
    model_name: str
    alias: str
    version: str
    actor: Optional[str] = None
    reason: Optional[str] = None


class VerifyRequest(BaseModel):
    task: str
    ref: str


class VerifyResponse(BaseModel):
    ok: bool


class ResolveResponse(BaseModel):
    ok: bool
    path: str


class ShowResponse(BaseModel):
    ok: bool
    meta: Dict[str, Any]
    package_dir: str


# ── Phase-2 request / response models ────────────────────────────────────────

class LogMetricRequest(BaseModel):
    task: str
    model_name: str
    run_id: str
    key: str
    value: float
    step: int = 0
    ts_utc: Optional[str] = None


class LogParamRequest(BaseModel):
    task: str
    model_name: str
    run_id: str
    key: str
    value: Any


class LogBatchRequest(BaseModel):
    task: str
    model_name: str
    run_id: str
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)


class RunGetResponse(BaseModel):
    ok: bool
    run_id: str
    task: str = ""
    model_name: str = ""
    status: str = "running"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    actor: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    metrics_summary: Dict[str, float] = Field(default_factory=dict)
    metric_history: Dict[str, List[Any]] = Field(default_factory=dict)


class MetricPoint(BaseModel):
    value: float
    step: int
    ts_utc: Optional[str] = None


class MetricsResponse(BaseModel):
    ok: bool
    version_metrics: Dict[str, Any]
    run_history: Dict[str, List[MetricPoint]]


class AliasEntry(BaseModel):
    alias: str
    version: str
    updated_at: Optional[str] = None
    actor: Optional[str] = None


class AliasesResponse(BaseModel):
    ok: bool
    model_name: str
    aliases: List[AliasEntry]


class SetTagRequest(BaseModel):
    task: str
    model_name: str
    version: str
    key: str
    value: str


class PatchVersionRequest(BaseModel):
    task: str
    model_name: str
    version: str
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class SetStageRequest(BaseModel):
    task: str
    model_name: str
    version: str
    stage: str
    actor: Optional[str] = None
    reason: Optional[str] = None


class CompareResponse(BaseModel):
    ok: bool
    versions: Dict[str, Dict[str, Any]]


class ArtifactEntry(BaseModel):
    name: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None


class ArtifactsResponse(BaseModel):
    ok: bool
    version: str
    files: List[ArtifactEntry]


# -------------------------
# Helpers
# -------------------------

def _handle_registry_error(e: Exception):
    if isinstance(e, ModelRegistryError):
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=500, detail=str(e))


def _require_model_ownership(task: str, model_name: str, user: UserContext) -> None:
    """Phase 2B: single call site for every route below that touches a
    model's filesystem state directly (compare/aliases/tags/versions-patch
    /stage) rather than through ModelRegistry.resolve_model/promote_model
    /register_model (those already enforce ownership internally via
    ModelRegistry._enforce_ownership_or_not_found -- see api.py). Raises
    the same 404 shape those routes already use (or would use) for a
    genuinely missing model -- anti-enumerating: a cross-org caller cannot
    distinguish "doesn't exist" from "not yours". user.org_id is always
    the verified UserContext's own field, never a header/body/query value.
    """
    check = check_model_ownership(
        Path(registry.root), task, model_name, requesting_org_id=user.org_id
    )
    if not check.allowed:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: task={task}, model_name={model_name}",
        )


def _get_db_conn():
    """Return a DB connection or raise HTTP 503 if DB is not configured."""
    if _db is None:
        raise HTTPException(status_code=503, detail={"ok": False, "error": "database module not available"})
    try:
        return _db.get_connection()
    except RegistryNotConfigured as exc:
        raise HTTPException(status_code=503, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"ok": False, "error": f"database unavailable: {exc}"})


# -------------------------
# Core endpoints
# -------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "omnibioai-model-registry",
        "version": APP_VERSION,
    }


@app.post(f"{DEFAULT_PREFIX}/register", response_model=RegisterResponse)
def api_register(req: RegisterRequest, user=Depends(require_write_auth_with_context)):
    # PR14.2B-3: require_write_auth_with_context (not require_write_auth)
    # so organization_id is available for usage-event emission; Phase 2B
    # now reuses the same UserContext for ownership enforcement below.
    # actor is derived the same way require_auth's own
    # _actor_identifier() already does, preserving the exact str value
    # every other route/call site (register_model(actor=...),
    # _audit.log_event(...)) already expects.
    actor = _actor_identifier(user)
    try:
        out = register_model(
            task=req.task,
            model_name=req.model_name,
            version=req.version,
            artifacts_dir=req.artifacts_dir,
            metadata=req.metadata,
            set_alias=req.set_alias,
            actor=actor,
            reason=req.reason,
            # Phase 2A: server-derived only, from the same verified
            # UserContext PR14.2B-3 already resolved -- never from
            # req.metadata or any other client-supplied field.
            organization_id=user.org_id,
            # Phase 2B: registering a version under an EXISTING model now
            # requires belonging to its owning org (ModelNotFound -> 400,
            # no filesystem mutation). A brand-new model is never blocked
            # -- see ModelRegistry.register_model's own docstring.
            enforce_ownership=True,
        )
        _audit.log_event(
            "register_model", actor,
            f"{req.task}/{req.model_name}@{req.version}",
            task=req.task, model_name=req.model_name, version=req.version,
            metadata={
                "organization_id": out.get("organization_id"),
                "ownership_status": out.get("ownership_status"),
            },
        )
        _emit_usage_safe(emit_model_registered, organization_id=user.org_id, user_id=user.user_id)
        return RegisterResponse(**out)
    except Exception as e:
        _handle_registry_error(e)


@app.post(f"{DEFAULT_PREFIX}/promote")
def api_promote(req: PromoteRequest, user: UserContext = Depends(require_write_auth_with_context)):
    actor = _actor_identifier(user)
    try:
        promote_model(
            task=req.task,
            model_name=req.model_name,
            alias=req.alias,
            version=req.version,
            actor=actor,
            reason=req.reason,
            # Phase 2B: promote always targets an existing model -- caller
            # must belong to its owning org (ModelNotFound -> 400).
            enforce_ownership=True,
            requesting_org_id=user.org_id,
        )
        _audit.log_event(
            "promote_model", actor,
            f"{req.task}/{req.model_name}@{req.alias}",
            task=req.task, model_name=req.model_name, version=req.version,
            metadata={"organization_id": user.org_id},
        )
        return {"ok": True}
    except Exception as e:
        _handle_registry_error(e)


@app.get(f"{DEFAULT_PREFIX}/resolve", response_model=ResolveResponse)
def api_resolve(task: str, ref: str, verify: bool = True, user: UserContext = Depends(require_auth_with_context)):
    try:
        path = resolve_model(
            task=task, model_ref=ref, verify=verify,
            enforce_ownership=True, requesting_org_id=user.org_id,
        )
        return ResolveResponse(ok=True, path=str(path))
    except Exception as e:
        _handle_registry_error(e)


@app.post(f"{DEFAULT_PREFIX}/verify", response_model=VerifyResponse)
def api_verify(req: VerifyRequest, user: UserContext = Depends(require_auth_with_context)):
    # POST but read-only/data-bearing (acts as an existence+integrity
    # oracle for a task/ref pair) -- discovered during the Phase 1
    # route-by-route review and protected on the same basis as the
    # explicitly listed GET endpoints, per the "every non-informational
    # endpoint" objective.
    try:
        verify_model_ref(
            task=req.task, model_ref=req.ref,
            enforce_ownership=True, requesting_org_id=user.org_id,
        )
        return VerifyResponse(ok=True)
    except Exception as e:
        _handle_registry_error(e)


@app.get(f"{DEFAULT_PREFIX}/show", response_model=ShowResponse)
def api_show(task: str, ref: str, verify: bool = False, user: UserContext = Depends(require_auth_with_context)):
    try:
        vdir = ModelRegistry.from_env().resolve_model(
            task=task, model_ref=ref, verify=verify,
            enforce_ownership=True, requesting_org_id=user.org_id,
        )
        meta_path = Path(vdir) / "model_meta.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="model_meta.json not found")

        meta = json.loads(meta_path.read_text())
        return ShowResponse(ok=True, meta=meta, package_dir=str(vdir))
    except Exception as e:
        _handle_registry_error(e)


# ==========================================================
# UI SUPPORT ENDPOINT (FINAL CLEAN VERSION)
# ==========================================================

@app.get(f"{DEFAULT_PREFIX}/models")
def list_models(
    task: Optional[str] = Query(None, description="filter by task"),
    model_name: Optional[str] = Query(None, description="filter by model name"),
    metric_gte: Optional[str] = Query(None, description="metric filter, format key:threshold"),
    user: UserContext = Depends(require_auth_with_context),
):
    """
    UI endpoint for dashboard:
    returns list of models from filesystem registry
    """
    # Parse metric_gte once so invalid format returns 400 before scanning files
    _metric_key: Optional[str] = None
    _metric_threshold: Optional[float] = None
    if metric_gte is not None:
        try:
            _metric_key, _thresh_str = metric_gte.split(":", 1)
            _metric_threshold = float(_thresh_str)
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=400,
                detail={"ok": False, "error": f"Invalid metric_gte format '{metric_gte}'. Expected 'key:threshold'"},
            )

    root = Path(registry.root)

    models: List[Dict[str, Any]] = []

    for meta_file in root.glob("**/model_meta.json"):
        try:
            meta = json.loads(meta_file.read_text())

            # Optional filters (safe, UI-friendly)
            if task and meta.get("task") != task:
                continue
            if model_name and meta.get("model_name") != model_name:
                continue

            # Phase 2B: ownership-aware filtering in the same filesystem
            # walk that already reads this entry -- a denied model is
            # never appended to `models`, not fetched-then-hidden in the
            # response. Skip entries with no task/model_name (malformed
            # meta) the same way the pre-existing filters implicitly do.
            m_task = meta.get("task")
            m_model_name = meta.get("model_name")
            if not m_task or not m_model_name:
                continue
            check = check_model_ownership(
                root, m_task, m_model_name, requesting_org_id=user.org_id
            )
            if not check.allowed:
                continue

            if _metric_key is not None:
                metrics_path = meta_file.parent / "metrics.json"
                if not metrics_path.exists():
                    continue
                try:
                    m_data = json.loads(metrics_path.read_text())
                except Exception:
                    continue
                if m_data.get(_metric_key, -float("inf")) < _metric_threshold:
                    continue

            models.append(meta)

        except Exception:
            continue

    # 🔥 IMPORTANT: return FLAT LIST (UI-friendly, no nesting confusion)
    return models


# ==========================================================
# Phase-2: DB-backed run tracking (HTTP 503 if DB absent)
# ==========================================================

_VALID_STAGES = frozenset({"none", "staging", "production", "archived"})


@app.post(f"{DEFAULT_PREFIX}/runs/log-metric")
def api_log_metric(req: LogMetricRequest, user: UserContext = Depends(require_write_auth_with_context)):
    conn = _get_db_conn()
    try:
        _tracking.log_metric(
            conn, req.run_id, req.key, req.value, req.step,
            task=req.task, model_name=req.model_name,
            # Phase 2C: a run_id already owned by a different org is
            # denied (ModelNotFound -> 400) before any INSERT; a
            # genuinely new run_id is attributed to this org. See
            # tracking.py's module docstring for why this is its own
            # independent ownership record, not derived from
            # ownership.json.
            requesting_org_id=user.org_id, enforce_ownership=True,
        )
        _audit.log_event(
            "log_metric", _actor_identifier(user),
            f"{req.task}/{req.model_name}#{req.run_id}",
            task=req.task, model_name=req.model_name,
            metadata={"organization_id": user.org_id, "run_id": req.run_id, "key": req.key, "value": req.value},
        )
        return {"ok": True}
    except Exception as e:
        _handle_registry_error(e)
    finally:
        conn.close()


@app.post(f"{DEFAULT_PREFIX}/runs/log-param")
def api_log_param(req: LogParamRequest, user: UserContext = Depends(require_write_auth_with_context)):
    conn = _get_db_conn()
    try:
        _tracking.log_param(
            conn, req.run_id, req.key, req.value,
            task=req.task, model_name=req.model_name,
            requesting_org_id=user.org_id, enforce_ownership=True,
        )
        # Deliberately no `value` in audit metadata -- param values are
        # caller-supplied Any (unlike log-metric's plain float) and could
        # contain something sensitive; only the key name is recorded.
        _audit.log_event(
            "log_param", _actor_identifier(user),
            f"{req.task}/{req.model_name}#{req.run_id}",
            task=req.task, model_name=req.model_name,
            metadata={"organization_id": user.org_id, "run_id": req.run_id, "key": req.key},
        )
        return {"ok": True}
    except Exception as e:
        _handle_registry_error(e)
    finally:
        conn.close()


@app.post(f"{DEFAULT_PREFIX}/runs/log-batch")
def api_log_batch(req: LogBatchRequest, user: UserContext = Depends(require_write_auth_with_context)):
    conn = _get_db_conn()
    try:
        for m in req.metrics:
            _tracking.log_metric(
                conn, req.run_id, m["key"], float(m["value"]), int(m.get("step", 0)),
                task=req.task, model_name=req.model_name,
                requesting_org_id=user.org_id, enforce_ownership=True,
            )
        if req.params:
            _tracking.log_params(
                conn, req.run_id, req.params, task=req.task, model_name=req.model_name,
                requesting_org_id=user.org_id, enforce_ownership=True,
            )
        if req.tags:
            _tracking.set_tags(
                conn, req.run_id, req.tags, task=req.task, model_name=req.model_name,
                requesting_org_id=user.org_id, enforce_ownership=True,
            )
        _audit.log_event(
            "log_batch", _actor_identifier(user),
            f"{req.task}/{req.model_name}#{req.run_id}",
            task=req.task, model_name=req.model_name,
            metadata={
                "organization_id": user.org_id, "run_id": req.run_id,
                "metrics_count": len(req.metrics),
                "param_keys": list(req.params.keys()),
                "tag_keys": list(req.tags.keys()),
            },
        )
        return {"ok": True}
    except Exception as e:
        _handle_registry_error(e)
    finally:
        conn.close()


@app.get(f"{DEFAULT_PREFIX}/runs/get", response_model=RunGetResponse)
def api_run_get(task: str, model: str, run_id: str, user: UserContext = Depends(require_auth_with_context)):
    conn = _get_db_conn()
    try:
        data = _tracking.get_run(
            conn, run_id, requesting_org_id=user.org_id, enforce_ownership=True,
        )
        history: Dict[str, List[Any]] = {}
        for key in data.get("metrics_summary", {}).keys():
            history[key] = _tracking.get_metric_history(
                conn, run_id, key, requesting_org_id=user.org_id, enforce_ownership=True,
            )
        data["metric_history"] = history
        return RunGetResponse(ok=True, **data)
    except Exception as e:
        _handle_registry_error(e)
    finally:
        conn.close()


@app.get(f"{DEFAULT_PREFIX}/runs/list")
def api_runs_list(task: str, model: str, user: UserContext = Depends(require_auth_with_context)):
    conn = _get_db_conn()
    try:
        # Phase 2C: org-filtered at the query layer (tracking.list_runs)
        # -- another org's run is never fetched, not fetched-then-hidden.
        basic = _tracking.list_runs(
            conn, task, model, requesting_org_id=user.org_id, enforce_ownership=True,
        )
        result = []
        for r in basic:
            try:
                # Redundant with the list_runs filter above in the
                # common case, kept for defense-in-depth consistency
                # (don't rely solely on the caller having filtered
                # correctly upstream).
                full = _tracking.get_run(
                    conn, r["run_id"], requesting_org_id=user.org_id, enforce_ownership=True,
                )
                result.append(full)
            except Exception:
                result.append(r)
        return result
    except Exception as e:
        _handle_registry_error(e)
    finally:
        conn.close()


# ==========================================================
# Phase-2: filesystem-based endpoints (DB optional)
# ==========================================================

@app.get(f"{DEFAULT_PREFIX}/metrics", response_model=MetricsResponse)
def api_metrics(task: str, ref: str, user: UserContext = Depends(require_auth_with_context)):
    try:
        vdir = Path(registry.resolve_model(
            task=task, model_ref=ref, verify=False,
            enforce_ownership=True, requesting_org_id=user.org_id,
        ))
    except Exception as e:
        _handle_registry_error(e)

    version_metrics: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    metrics_path = vdir / "metrics.json"
    meta_path = vdir / "model_meta.json"

    if metrics_path.exists():
        try:
            version_metrics = json.loads(metrics_path.read_text())
        except Exception:
            pass
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    run_id: Optional[str] = (meta.get("lineage") or {}).get("run_id")
    run_history: Dict[str, list] = {}

    # Phase 2C: deliberately NOT run-ownership-gated here, unlike
    # /v1/runs/get|list. `run_id` is not caller-supplied on this route --
    # it's read out of `meta`, this model's OWN model_meta.json, only
    # reachable because resolve_model() above already enforced ownership
    # on (task, model_name). Gating this second, transitively-safe lookup
    # on omr_runs.organization_id too would incorrectly lock the model's
    # legitimate owner out of their own run history whenever that run
    # predates the Phase 2C migration (organization_id=NULL,
    # ownership_status='legacy_unowned' -- see db.py's _ALTER_DDL
    # comment) or was logged before this model existed at all (the
    # ordinary log-during-experiment-then-register-after workflow).
    if run_id and _db is not None:
        try:
            conn = _db.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT key_name FROM omr_metrics WHERE run_id = %s",
                        (run_id,),
                    )
                    keys = [r["key_name"] for r in cur.fetchall()]
                for key in keys:
                    run_history[key] = _tracking.get_metric_history(conn, run_id, key)
            finally:
                conn.close()
        except RegistryNotConfigured:
            # Filesystem fallback — read *.jsonl written by RunLogger.
            #
            # Security: model_name is derived from `vdir` itself (the
            # real, already-resolved, ownership-checked directory
            # resolve_model() returned above -- .../models/<model_name>/
            # versions/<version>), and `run_id` is routed through the
            # same path_safety validation every other run_dir() caller
            # gets. Deliberately NOT meta.get("task")/
            # meta.get("model_name"): those live inside model_meta.json,
            # itself part of the free-form `metadata` dict a caller
            # supplies at registration time (register_model() only ever
            # *defaults* task/model_name into it with setdefault(), it
            # never overwrites an attacker-supplied "task"/"model_name"
            # key already present in that dict) -- trusting them here
            # would let a same-org caller read an arbitrary
            # attacker-chosen *.jsonl path via a poisoned
            # model_meta.json rather than this model's real identity. A
            # malformed run_id (or one with simply no such directory)
            # yields no run history, same as today -- this fallback has
            # always been best-effort.
            real_model_name = vdir.parent.parent.name
            try:
                run_metrics_dir = L.run_dir(Path(registry.root), task, real_model_name, run_id) / "metrics"
            except PathTraversalError:
                run_metrics_dir = None
            if run_metrics_dir is not None and run_metrics_dir.exists():
                for jf in sorted(run_metrics_dir.glob("*.jsonl")):
                    entries = []
                    for line in jf.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            entries.append({"value": e.get("value", 0.0), "step": e.get("step", 0), "ts_utc": e.get("ts_utc")})
                        except Exception:
                            pass
                    if entries:
                        run_history[jf.stem] = entries
        except Exception:
            pass

    return MetricsResponse(ok=True, version_metrics=version_metrics, run_history=run_history)


@app.get(f"{DEFAULT_PREFIX}/aliases", response_model=AliasesResponse)
def api_aliases(task: str, model: str, user: UserContext = Depends(require_auth_with_context)):
    from omnibioai_model_registry.package.layout import aliases_root as _aliases_root

    # Phase 2B: alias resolution is model metadata retrieval -- reads the
    # filesystem directly, bypassing ModelRegistry.resolve_model entirely,
    # so it needs its own ownership check.
    _require_model_ownership(task, model, user)

    aliases_dir = _aliases_root(Path(registry.root), task, model)
    entries: List[AliasEntry] = []
    if aliases_dir.exists():
        for f in sorted(aliases_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                entries.append(AliasEntry(
                    alias=data.get("alias", f.stem),
                    version=data.get("version", ""),
                    updated_at=data.get("updated_at"),
                    actor=data.get("actor"),
                ))
            except Exception:
                continue
    return AliasesResponse(ok=True, model_name=model, aliases=entries)


@app.put(f"{DEFAULT_PREFIX}/tags")
def api_set_tag(req: SetTagRequest, user: UserContext = Depends(require_write_auth_with_context)):
    import tempfile

    actor = _actor_identifier(user)
    # Phase 2B: checked before either the DB write or the filesystem
    # write below, so a cross-org caller mutates nothing at all.
    _require_model_ownership(req.task, req.model_name, user)

    # Persist to DB when available (best-effort; never blocks the filesystem write)
    if _db is not None:
        try:
            conn = _db.get_connection()
            try:
                # Phase 2C: defense-in-depth -- _require_model_ownership()
                # above already gates this whole route, but
                # set_version_tag() re-derives from the same
                # ownership.json rather than trusting only the route.
                _tracking.set_version_tag(
                    conn, req.task, req.model_name, req.version, req.key, req.value,
                    registry_root=Path(registry.root), requesting_org_id=user.org_id,
                    enforce_ownership=True,
                )
            finally:
                conn.close()
        except Exception:
            pass

    # Always patch model_meta.json (atomic write)
    from omnibioai_model_registry.package.layout import version_dir as _vdir_fn
    vdir = _vdir_fn(Path(registry.root), req.task, req.model_name, req.version)
    meta_path = vdir / "model_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            tags = meta.get("tags") or {}
            tags[req.key] = req.value
            meta["tags"] = tags
            text = json.dumps(meta, indent=2) + "\n"
            fd, tmp = tempfile.mkstemp(dir=meta_path.parent, suffix=".tmp")
            try:
                os.write(fd, text.encode())
            finally:
                os.close(fd)
            os.replace(tmp, meta_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    _audit.log_event(
        "set_tag", actor,
        f"{req.task}/{req.model_name}@{req.version}",
        task=req.task, model_name=req.model_name, version=req.version,
        metadata={"key": req.key, "value": req.value, "organization_id": user.org_id},
    )
    return {"ok": True}


@app.post(f"{DEFAULT_PREFIX}/versions/patch")
def api_patch_version(req: PatchVersionRequest, user: UserContext = Depends(require_write_auth_with_context)):
    import tempfile

    # Phase 2B: checked before the meta_path existence probe below, so a
    # cross-org caller can't distinguish "not yours" from "genuinely
    # missing" -- both are the same 404.
    _require_model_ownership(req.task, req.model_name, user)

    from omnibioai_model_registry.package.layout import version_dir as _vdir_fn
    vdir = _vdir_fn(Path(registry.root), req.task, req.model_name, req.version)
    meta_path = vdir / "model_meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="model_meta.json not found")

    try:
        meta = json.loads(meta_path.read_text())
        if req.description is not None:
            meta["description"] = req.description
        if req.tags is not None:
            existing_tags = meta.get("tags") or {}
            existing_tags.update(req.tags)
            meta["tags"] = existing_tags
        text = json.dumps(meta, indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(dir=meta_path.parent, suffix=".tmp")
        try:
            os.write(fd, text.encode())
        finally:
            os.close(fd)
        os.replace(tmp, meta_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _audit.log_event(
        "patch_version", _actor_identifier(user),
        f"{req.task}/{req.model_name}@{req.version}",
        task=req.task, model_name=req.model_name, version=req.version,
        metadata={"organization_id": user.org_id},
    )
    return {"ok": True}


@app.post(f"{DEFAULT_PREFIX}/stage")
def api_set_stage(req: SetStageRequest, user: UserContext = Depends(require_write_auth_with_context)):
    import tempfile

    actor = _actor_identifier(user)
    if req.stage not in _VALID_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage '{req.stage}'. Must be one of: {sorted(_VALID_STAGES)}",
        )

    # Phase 2B: checked before the vdir existence probe below, so a
    # cross-org caller can't distinguish "not yours" from "genuinely
    # missing version" -- both are the same 404.
    _require_model_ownership(req.task, req.model_name, user)

    from omnibioai_model_registry.package.layout import version_dir as _vdir_fn
    vdir = _vdir_fn(Path(registry.root), req.task, req.model_name, req.version)
    if not vdir.exists():
        raise HTTPException(status_code=404, detail=f"Version not found: {req.version}")

    meta_path = vdir / "model_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["stage"] = req.stage
            text = json.dumps(meta, indent=2) + "\n"
            fd, tmp = tempfile.mkstemp(dir=meta_path.parent, suffix=".tmp")
            try:
                os.write(fd, text.encode())
            finally:
                os.close(fd)
            os.replace(tmp, meta_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if req.stage in ("staging", "production"):
        try:
            promote_model(
                task=req.task,
                model_name=req.model_name,
                alias=req.stage,
                version=req.version,
                actor=actor,
                reason=req.reason or "stage transition",
                # Defense in depth: this route already enforced ownership
                # above, but promote_model is the shared library call site
                # every other caller (including the standalone /v1/promote
                # route) also goes through -- keep it consistent here too.
                enforce_ownership=True,
                requesting_org_id=user.org_id,
            )
        except Exception as e:
            _handle_registry_error(e)

    _audit.log_event(
        "set_stage", actor,
        f"{req.task}/{req.model_name}@{req.stage}",
        task=req.task, model_name=req.model_name, version=req.version,
        metadata={"organization_id": user.org_id},
    )
    return {"ok": True, "stage": req.stage, "version": req.version}


@app.get(f"{DEFAULT_PREFIX}/compare", response_model=CompareResponse)
def api_compare(
    task: str,
    model: str,
    versions: List[str] = Query(default=[]),
    user: UserContext = Depends(require_auth_with_context),
):
    if len(versions) < 2:
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": "At least 2 versions required"},
        )
    # Phase 2B: compare reads version_dir metadata directly, bypassing
    # ModelRegistry.resolve_model -- own ownership check required. One
    # model per request (all versions belong to it), so a single check
    # covers every version listed.
    _require_model_ownership(task, model, user)

    from omnibioai_model_registry.package.layout import version_dir as _vdir_fn

    result: Dict[str, Dict[str, Any]] = {}
    for ver in versions:
        vdir = _vdir_fn(Path(registry.root), task, model, ver)
        entry: Dict[str, Any] = {}
        metrics_path = vdir / "metrics.json"
        meta_path = vdir / "model_meta.json"
        if metrics_path.exists():
            try:
                entry["metrics"] = json.loads(metrics_path.read_text())
            except Exception:
                entry["metrics"] = {}
        else:
            entry["metrics"] = {}
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                entry["created_at"] = m.get("created_at")
                entry["stage"] = m.get("stage", "none")
                entry["tags"] = m.get("tags") or {}
            except Exception:
                pass
        result[ver] = entry
    return CompareResponse(ok=True, versions=result)


@app.get(f"{DEFAULT_PREFIX}/artifacts", response_model=ArtifactsResponse)
def api_artifacts(task: str, ref: str, user: UserContext = Depends(require_auth_with_context)):
    try:
        vdir = Path(registry.resolve_model(
            task=task, model_ref=ref, verify=False,
            enforce_ownership=True, requesting_org_id=user.org_id,
        ))
    except Exception as e:
        _handle_registry_error(e)

    # Parse sha256sums.txt manually (avoids Cython extension)
    hashes: Dict[str, str] = {}
    manifest_path = vdir / "sha256sums.txt"
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                hashes[parts[-1]] = parts[0]

    files: List[ArtifactEntry] = []
    for f in sorted(vdir.iterdir()):
        if not f.is_file():
            continue
        size: Optional[int] = None
        try:
            size = f.stat().st_size
        except OSError:
            pass
        files.append(ArtifactEntry(name=f.name, sha256=hashes.get(f.name), size_bytes=size))

    return ArtifactsResponse(ok=True, version=vdir.name, files=files)


@app.get(f"{DEFAULT_PREFIX}/auth/status")
def api_auth_status():
    try:
        cfg = load_config()
        auth_enabled = cfg.auth_enabled
        iam_url = cfg.iam_url if auth_enabled else None
    except Exception:
        auth_enabled = False
        iam_url = None
    return {
        "auth_enabled": auth_enabled,
        "mode": "jwt" if auth_enabled else "open",
        "iam_url": iam_url,
    }