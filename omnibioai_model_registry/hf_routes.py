# File: omnibioai_model_registry/hf_routes.py
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from iam_client.models import UserContext
from pydantic import BaseModel

from .api import ModelRegistry
from .audit_client import AuditClient
from .auth import _actor_identifier, require_auth_with_context, require_write_auth_with_context

router = APIRouter()

_registry = ModelRegistry.from_env()
_audit = AuditClient(os.environ.get("AUDIT_URL", ""))

# In-memory job store. This service has no queue/worker process (see
# audit_client.py for the same fire-and-forget-thread pattern), so push
# jobs are tracked here and polled by job_id.
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        _JOBS.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None


class HFPushRequest(BaseModel):
    task: str
    model_name: str
    version: str
    repo_id: str
    token: Optional[str] = None
    description: Optional[str] = ""
    license: str = "apache-2.0"
    private: bool = True


class HFPushResponse(BaseModel):
    ok: bool
    job_id: str


class HFPushStatusResponse(BaseModel):
    ok: bool
    status: str
    url: Optional[str] = None
    error: Optional[str] = None


class HFSettingsResponse(BaseModel):
    has_default_token: bool
    default_namespace: str


def _run_push(
    job_id: str,
    vdir: Path,
    repo_id: str,
    private: bool,
    token: str,
    description: str,
    license_id: str,
) -> None:
    try:
        _set_job(job_id, status="running")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        # Upload the version directory as-is — it is never mutated so that
        # sha256sums.txt (the integrity manifest for this registered
        # version) stays valid. The model card is uploaded as a separate
        # file rather than written into vdir.
        api.upload_folder(folder_path=str(vdir), repo_id=repo_id, repo_type="model", token=token)

        card = (
            f"---\nlicense: {license_id}\n---\n\n"
            f"# {repo_id}\n\n{description}\n\n"
            "Pushed from [OmniBioAI Model Registry](https://github.com/man4ish/omnibioai).\n"
        )
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

        _set_job(job_id, status="success", url=f"https://huggingface.co/{repo_id}")
    except Exception as exc:
        _set_job(job_id, status="error", error=str(exc))


@router.post("/push", response_model=HFPushResponse)
def hf_push(req: HFPushRequest, user: UserContext = Depends(require_write_auth_with_context)):
    actor = _actor_identifier(user)
    token = (req.token or os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise HTTPException(
            status_code=400,
            detail="No HuggingFace token provided and HF_TOKEN is not configured on the server.",
        )

    # Phase 2B: resolve_model's own enforce_ownership check (reads
    # ownership.json, compares against user.org_id) runs before this
    # returns a vdir at all -- a cross-org caller gets the same
    # "not found" response a genuinely missing model/version would, and
    # nothing below (artifact read, HF API calls) ever executes. This is
    # the entire "authenticate -> resolve -> read ownership -> compare ->
    # deny -> only then push" sequence the HF push threat model requires,
    # in one call.
    try:
        vdir = Path(
            _registry.resolve_model(
                task=req.task, model_ref=f"{req.model_name}@{req.version}", verify=False,
                enforce_ownership=True, requesting_org_id=user.org_id,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Model version not found: {exc}")

    job_id = str(uuid.uuid4())
    # Phase 2D: the job itself now carries the same organization_id that
    # already gated the resolve_model() call above, so hf_push_status()
    # can enforce the same boundary when a different caller later polls
    # this job_id -- closes the gap hf_push_status()'s own docstring
    # flagged since Phase 1 ("deferred to the tenant-isolation phase").
    _set_job(
        job_id, status="queued", url=None, error=None, repo_id=req.repo_id,
        organization_id=user.org_id,
    )

    t = threading.Thread(
        target=_run_push,
        args=(job_id, vdir, req.repo_id, req.private, token, req.description or "", req.license),
        daemon=True,
    )
    t.start()

    _audit.log_event(
        "push_to_hf",
        actor,
        f"{req.task}/{req.model_name}@{req.version}",
        task=req.task,
        model_name=req.model_name,
        version=req.version,
        metadata={"repo_id": req.repo_id, "job_id": job_id, "organization_id": user.org_id},
    )
    return HFPushResponse(ok=True, job_id=job_id)


@router.get("/push/status/{job_id}", response_model=HFPushStatusResponse)
def hf_push_status(job_id: str, user: UserContext = Depends(require_auth_with_context)):
    # Phase 2D: the job's organization_id (set at creation in hf_push()
    # above, from the same verified UserContext that already gated the
    # push itself) must match the polling caller's own org_id -- a
    # cross-org caller gets the exact same 404 a genuinely unknown
    # job_id already returned, so this remains a non-enumerating
    # "unknown job_id" either way, never distinguishing "doesn't exist"
    # from "not yours". job_id is also unguessable (uuid4) and the
    # response never contains the HF token, so this closes the
    # previously-accepted "bounded enumeration/status-leak" gap this
    # docstring flagged since Phase 1, rather than merely narrowing it.
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    job_org_id = job.get("organization_id")
    if not (
        (user.org_id is not None and user.org_id == job_org_id)
        or (user.org_id is None and job_org_id is None)
    ):
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return HFPushStatusResponse(
        ok=True, status=job.get("status", "unknown"), url=job.get("url"), error=job.get("error")
    )


@router.get("/settings", response_model=HFSettingsResponse)
def hf_settings():
    return HFSettingsResponse(
        has_default_token=bool(os.environ.get("HF_TOKEN", "").strip()),
        default_namespace=os.environ.get("HF_DEFAULT_NAMESPACE", "omnibioai"),
    )
