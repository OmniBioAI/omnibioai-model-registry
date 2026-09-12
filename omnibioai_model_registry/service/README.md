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
pip install -r omnibioai_model_registry/service/requirements.txt

uvicorn omnibioai_model_registry.service.app.main:app --host 0.0.0.0 --port 8095
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
  "version": "0.1.4"
}
```

(`version` tracks the installed `omnibioai-model-registry` package version —
it will differ if you're on a different release.)

---

# 6) Run With Docker (Recommended)

This directory's own `Dockerfile` is stale (a single-stage, API-only image
with a module path that no longer matches the code) and superseded by the
repo root's multi-stage `Dockerfile`, which also bundles the ModelHub UI
behind nginx. See the root README's
[Docker](../../README.md#docker) section for the current build/run
instructions, including the build-context caveat (it must be this repo's
*parent* directory).

`artifacts_dir` still needs to be visible **inside** whichever container
you run: mount the registry root for persistent storage, plus a shared
"staging" directory that training outputs are written to, and reference
that mount path (e.g. `/shared/model_pkg_001`) as `artifacts_dir` in your
`register` calls.

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

This section describes the state as of the Phase 2B organization-
ownership *enforcement* (built on top of Phase 2A's ownership records).
It supersedes both the old "v0.1 — no authentication, assumes trusted
internal network" description and the later "recorded but not yet
enforced" one — neither reflects the code anymore.

* When `AUTH_ENABLED=true`, every non-informational endpoint — reads
  (`resolve`, `verify`, `show`, `models`, `runs/get`, `runs/list`,
  `metrics`, `aliases`, `compare`, `artifacts`,
  `hf/push/status/{job_id}`) and mutations alike — requires a valid
  Bearer JWT carrying the `model.use` IAM permission; see the root
  README's [Authentication](../../README.md#authentication) section.
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
  production default.
* **Organization ownership is recorded *and enforced*.** `POST
  /v1/register` records which organization owns a newly-registered
  model, derived only from the caller's verified IAM identity
  (`UserContext.org_id`) — never from a header, body field, or
  query/path parameter. This is a durable, write-once, server-controlled
  record (`ownership.json`, one per model). As of Phase 2B, every
  model-bearing read/write route above — plus `POST /v1/hf/push` and its
  status-polling endpoint — independently checks the caller's verified
  `org_id` against the model's recorded owner and denies a mismatch
  (same "not found" response a genuinely missing model would return, so
  it doesn't leak which models exist). `model.use` itself is still a
  flat, non-resource-scoped permission; ownership enforcement sits
  alongside it, not inside a redesigned permission model. Legacy models
  (registered before Phase 2A shipped) are recorded as
  `legacy_unowned` and denied for *every* caller, including one with no
  org context — never guessed, and no longer just left unassigned.
* **Legacy-ownership resolution is real and in use** (Phase 2E):
  `POST /v1/ownership/resolve` lets a caller holding the separate
  `model.resolve_ownership` permission claim a `legacy_unowned` model
  into their own organization — write-once, idempotent for a repeat
  resolution to the same org, and always self-scoped (no
  caller-supplied target `organization_id` anywhere in the HTTP path).
  The equivalent CLI backfill/resolve commands (`omr migrate-ownership`,
  `omr resolve-ownership`) have been used against this registry's real
  data, not just exercised in tests. See the root README's
  [Organization Ownership and Enforcement](../../README.md#organization-ownership-and-enforcement-phase-2a-phase-2b)
  and
  [Legacy Ownership Resolution](../../README.md#legacy-ownership-resolution-phase-2e)
  sections for the full design, including the tracking-data (Phase 2C)
  and filesystem-path-safety hardening layered on top of this.

