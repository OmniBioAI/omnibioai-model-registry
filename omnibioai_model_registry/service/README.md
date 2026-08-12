# OmniBioAI Model Registry – REST Service

This directory contains the **minimal REST wrapper** for the OmniBioAI Model Registry.

It exposes the registry over HTTP while preserving:

* Immutable version storage
* Integrity verification (SHA256)
* Alias promotion workflow
* Storage abstraction
* Scientific provenance tracking

The REST service is a **thin wrapper** over the core `omnibioai_model_registry` Python library.

> No business logic exists in this layer.
> All lifecycle logic lives in the core library.

---

# Architecture

```
Client (Plugin / TES / UI)
        │
        ▼
FastAPI REST Service
        │
        ▼
omnibioai_model_registry (Core Library)
        │
        ▼
Filesystem / Object Store
```

This ensures:

* CLI and REST share identical behavior
* No duplication of logic
* Deterministic model resolution

---

# Environment Variable

The service requires:

```
OMNIBIOAI_MODEL_REGISTRY_ROOT
```

Example (local):

```bash
export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/Desktop/machine/local_registry/model_registry
```

Inside Docker, this must point to a mounted path.

---

# 5) Run Locally (No Docker, No Compose)

```bash
export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/Desktop/machine/local_registry/model_registry

cd ~/Desktop/machine/omnibioai-model-registry
pip install -e .
pip install -r service/requirements.txt

uvicorn service.app.main:app --host 0.0.0.0 --port 8095
```

### Test

```bash
curl -s http://127.0.0.1:8095/health | python -m json.tool
```

You should see:

```json
{
  "ok": true,
  "service": "omnibioai-model-registry",
  "version": "0.1.0",
  "registry_root_configured": true
}
```

---

# 6) Run With Docker (Recommended)

### Important Concept

`artifacts_dir` must be visible **inside the container**.

We use two mounts:

1. Registry root (persistent storage)
2. A shared “staging” directory where training outputs are written

---

## Prepare Directories

```bash
mkdir -p ~/Desktop/machine/local_registry/model_registry
mkdir -p ~/Desktop/machine/local_registry/staging
```

---

## Build

```bash
docker build -f service/Dockerfile -t omnibioai-model-registry-svc .
```

---

## Run

```bash
docker run --rm -p 8095:8095 \
  -e OMNIBIOAI_MODEL_REGISTRY_ROOT=/data/model_registry \
  -v ~/Desktop/machine/local_registry/model_registry:/data/model_registry \
  -v ~/Desktop/machine/local_registry/staging:/shared \
  omnibioai-model-registry-svc
```

Now:

* `/data/model_registry` → registry storage
* `/shared` → staging directory

Your API `artifacts_dir` must reference something like:

```
"/shared/model_pkg_001"
```

---

# 7) Example REST Calls

---

## Register

(Assumes you created `/shared/model_pkg` on host under staging mount)

```bash
curl -s -X POST http://127.0.0.1:8095/v1/register \
  -H "Content-Type: application/json" \
  -d '{
    "task":"celltype_classification_sc",
    "model_name":"human_pbmc",
    "version":"2026-02-13_001",
    "artifacts_dir":"/shared/model_pkg",
    "metadata":{"framework":"pytorch","model_type":"mlp"},
    "set_alias":"latest",
    "actor":"manish",
    "reason":"api smoke test"
  }' | python -m json.tool
```

---

## Resolve

```bash
curl -s "http://127.0.0.1:8095/v1/resolve?task=celltype_classification_sc&ref=human_pbmc@latest&verify=true" \
| python -m json.tool
```

---

## Show Metadata

```bash
curl -s "http://127.0.0.1:8095/v1/show?task=celltype_classification_sc&ref=human_pbmc@latest&verify=true" \
| python -m json.tool
```

---

## Promote

```bash
curl -s -X POST http://127.0.0.1:8095/v1/promote \
  -H "Content-Type: application/json" \
  -d '{
    "task":"celltype_classification_sc",
    "model_name":"human_pbmc",
    "alias":"production",
    "version":"2026-02-13_001",
    "actor":"manish",
    "reason":"validated"
  }' | python -m json.tool
```

---

## Verify

```bash
curl -s -X POST http://127.0.0.1:8095/v1/verify \
  -H "Content-Type: application/json" \
  -d '{"task":"celltype_classification_sc","ref":"human_pbmc@production"}' \
| python -m json.tool
```

---

# Security Model

This section describes the state as of the Phase 2A organization-
ownership hardening. It supersedes the old "v0.1 — no authentication,
assumes trusted internal network" description, which no longer reflects
the code.

* When `AUTH_ENABLED=true`, every non-informational endpoint — reads
  (`resolve`, `verify`, `show`, `models`, `runs/get`, `runs/list`,
  `metrics`, `aliases`, `compare`, `artifacts`,
  `hf/push/status/{job_id}`) and mutations alike — requires a valid
  Bearer JWT carrying the `model.use` IAM permission (`model.use` is the
  only permission this service checks; see the root README's
  [Authentication](../../README.md#authentication) section).
* The registry verifies the JWT itself via `omnibioai-iam-client`,
  independently of the API Gateway. It does not trust gateway-injected
  identity headers (`X-Organization-ID`, `X-Team-ID`, `X-User-ID`,
  `X-User-Email`) as a substitute for a verified token.
* The API Gateway remains a real enforcement layer in front of this
  service (it authenticates and permission-checks requests before
  forwarding them), but this service no longer depends on the Gateway,
  or on network topology, as its only line of defense.
* `AUTH_ENABLED=false` (the default) still runs the service fully open —
  no token required anywhere, every call attributed to a synthetic
  `system` actor. This is an explicit opt-in dev/test switch, not a
  production default; it is unchanged by the Phase 1 hardening.
* **Organization ownership is now recorded, but not yet enforced**
  (Phase 2A). `POST /v1/register` now records which organization owns a
  newly-registered model, derived only from the caller's verified IAM
  identity (`UserContext.org_id`) — never from a header, body field, or
  query/path parameter. This is a durable, write-once, server-controlled
  record (`ownership.json`, one per model). It does **not** yet gate
  anything: `model.use` is still a flat permission, not scoped to a
  resource or an org, and any authenticated caller holding it can still
  read or mutate any model regardless of who owns it. Legacy models
  (registered before this phase) are recorded as explicitly
  `legacy_unowned`, never guessed. See the root README's
  [Organization Ownership](../../README.md#organization-ownership-phase-2a)
  section for the full design and the deferred Phase 2B/2C work
  (query-layer enforcement, HF-push ownership check, tracking-table
  scoping, resource-scoped `model.use`).

