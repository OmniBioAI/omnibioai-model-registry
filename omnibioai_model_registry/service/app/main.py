from __future__ import annotations

import json
import os
from fastapi.middleware.cors import CORSMiddleware
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from omnibioai_model_registry import (
    ModelRegistry,
    promote_model,
    register_model,
    resolve_model,
    verify_model_ref,
)
from omnibioai_model_registry.errors import ModelRegistryError

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
# ==========================================================
# Shared registry instance for UI/API
# ==========================================================
registry = ModelRegistry.from_env()


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


# -------------------------
# Helpers
# -------------------------

def _handle_registry_error(e: Exception):
    if isinstance(e, ModelRegistryError):
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=500, detail=str(e))


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
def api_register(req: RegisterRequest):
    try:
        out = register_model(
            task=req.task,
            model_name=req.model_name,
            version=req.version,
            artifacts_dir=req.artifacts_dir,
            metadata=req.metadata,
            set_alias=req.set_alias,
            actor=req.actor,
            reason=req.reason,
        )
        return RegisterResponse(**out)
    except Exception as e:
        _handle_registry_error(e)


@app.post(f"{DEFAULT_PREFIX}/promote")
def api_promote(req: PromoteRequest):
    try:
        promote_model(
            task=req.task,
            model_name=req.model_name,
            alias=req.alias,
            version=req.version,
            actor=req.actor,
            reason=req.reason,
        )
        return {"ok": True}
    except Exception as e:
        _handle_registry_error(e)


@app.get(f"{DEFAULT_PREFIX}/resolve", response_model=ResolveResponse)
def api_resolve(task: str, ref: str, verify: bool = True):
    try:
        path = resolve_model(task=task, model_ref=ref, verify=verify)
        return ResolveResponse(ok=True, path=str(path))
    except Exception as e:
        _handle_registry_error(e)


@app.post(f"{DEFAULT_PREFIX}/verify", response_model=VerifyResponse)
def api_verify(req: VerifyRequest):
    try:
        verify_model_ref(task=req.task, model_ref=req.ref)
        return VerifyResponse(ok=True)
    except Exception as e:
        _handle_registry_error(e)


@app.get(f"{DEFAULT_PREFIX}/show", response_model=ShowResponse)
def api_show(task: str, ref: str, verify: bool = False):
    try:
        vdir = ModelRegistry.from_env().resolve_model(
            task=task, model_ref=ref, verify=verify
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
):
    """
    UI endpoint for dashboard:
    returns list of models from filesystem registry
    """

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

            models.append(meta)

        except Exception:
            continue

    # 🔥 IMPORTANT: return FLAT LIST (UI-friendly, no nesting confusion)
    return models