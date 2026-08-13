# OmniBioAI ModelHub

**OmniBioAI ModelHub** is a production-grade experiment tracking and model lifecycle management system for AI/ML models within the OmniBioAI ecosystem — purpose-built for biomedical AI plugins.

It provides:

- **Experiment tracking** — log params, metrics, tags from training runs
- **Step-indexed metric history** with sparkline visualization
- **Immutable model versioning** (write-once)
- **Cryptographic integrity verification (SHA256)**
- **Staged promotion workflows** (latest → staging → production)
- **Alias management** with full audit trail
- **MySQL-backed run storage** with filesystem fallback
- **Plugin-first design** — `PluginRunClient` for TES container environments
- **Local-first, cloud-ready** storage abstraction
- **REST API (FastAPI) + CLI (`omr`) + Python SDK**
- **IAM-gated reads and writes** — every non-informational endpoint (mutations and reads alike) requires a valid JWT (via `omnibioai-iam-client`) carrying the `model.use` permission, when `AUTH_ENABLED=true`; the registry verifies this itself rather than relying solely on the API Gateway
- **One-click Hugging Face push** — package and upload a registered model version straight to the Hub
- **Usage metering + cross-service audit** — registration events are emitted to the platform usage pipeline and to the security-audit service, in addition to this repo's own local `promotions.jsonl` trail

The registry is implemented as a **standalone Python library** (package name: `omnibioai-model-registry`, CLI entrypoint: `omr`) and ships a self-contained FastAPI service.

---

## Status

- ✅ Experiment tracking (`RunLogger`, `PluginRunClient`)
- ✅ MySQL-backed metric + param storage
- ✅ Immutable and verifiable model storage
- ✅ Audit-ready promotion workflow
- ✅ 21 REST endpoints (tracking + registry + governance + Hugging Face push)
- ✅ 12 CLI commands
- ✅ IAM-gated reads and writes (`model.use` permission, `omnibioai-iam-client`, enforced independently at the registry)
- ✅ Usage metering + cross-service audit emission
- ✅ ModelHub UI with Experiments tab + metric sparklines
- ✅ Local-first, cloud-ready design
- ✅ Organization ownership recorded (server-derived from verified IAM identity) and enforced on every model-bearing read/write route, including Hugging Face push; see [Organization Ownership and Enforcement](#organization-ownership-and-enforcement-phase-2a-phase-2b)
- ✅ Run-tracking data (`omr_runs`/`omr_params`/`omr_metrics`/`omr_tags`/`omr_version_tags`) organization-isolated too; see [Tracking-Data Organization Isolation](#tracking-data-organization-isolation-phase-2c)
- ✅ Filesystem path traversal closed for every task/model/version/alias/run/metric identifier, centrally enforced; see [Filesystem Path Safety](#filesystem-path-safety)

---

## Why This Exists

Biomedical AI requires:

- **Reproducibility**
- **Auditability**
- **Governance**
- **Offline / air-gapped deployment**
- **Cross-infrastructure execution parity**

Traditional ML tooling often assumes cloud-first infrastructure, mutable artifacts, and weak provenance guarantees.

**OmniBioAI ModelHub is designed differently.**

> It treats AI models as **scientific artifacts** that must be **immutable, verifiable, and reproducible** across environments.

---

## Experiment Tracking

Two clients cover the two primary execution contexts.

### RunLogger (filesystem — local scripts, notebooks)

Writes directly to the registry root on the local filesystem. No network required.

```python
from omnibioai_model_registry import RunLogger, register_model

with RunLogger(task="celltype_sc", model_name="human_pbmc") as run:
    run.log_params({"lr": 0.001, "epochs": 50, "batch_size": 32})
    for epoch, acc in enumerate(training_curve):
        run.log_metric("accuracy", acc, step=epoch)
        run.log_metric("val_loss", loss, step=epoch)
    run.set_tag("team", "bioml")

register_model(
    task="celltype_sc",
    model_name="human_pbmc",
    version="2026-06-14_001",
    artifacts_dir="/path/to/artifacts",
    metadata={"lineage": {"run_id": run.run_id}},
)
```

Filesystem layout produced by `RunLogger`:

```
{registry_root}/tasks/{task}/models/{model_name}/runs/{run_id}/
    params.json          # {"lr": 0.001, "epochs": 50}
    tags.json            # {"team": "bioml"}
    metrics/
        accuracy.jsonl   # one JSON record per step
        val_loss.jsonl
```

### PluginRunClient (HTTP — TES container plugins)

Posts metrics and params to the ModelHub REST API. Designed for training jobs running inside TES-scheduled containers that cannot access the registry filesystem directly.

```python
import os
from omnibioai_model_registry import PluginRunClient

with PluginRunClient(
    task="celltype_sc",
    model_name="human_pbmc",
    registry_url=os.environ["MODEL_REGISTRY_BASE_URL"],
) as run:
    run.log_params({"lr": 0.001})
    run.log_metric("accuracy", 0.95, step=0)
    run.set_tag("plugin_version", "1.2.3")
```

Both clients share the same `log_params` / `log_metric` / `set_tag` interface. The storage backend is the only difference.

---

## Role in the OmniBioAI Architecture

OmniBioAI follows a **four-plane architecture**:

| Plane             | Responsibility                         |
| ----------------- | -------------------------------------- |
| **Control Plane** | UI, registries, metadata, governance   |
| **Compute Plane** | Workflow execution, HPC/cloud adapters |
| **Data Plane**    | Artifacts, datasets, outputs           |
| **AI Plane**      | Reasoning, RAG, agents, interpretation |

The **ModelHub** belongs to the **Control Plane** and provides AI artifact governance, deterministic inference references, promotion and audit workflows, and infrastructure-independent model resolution.

---

## Core Design Principles

### 1) Immutability
Each model version is **write-once**: no overwrites, no silent mutation, full historical trace. This guarantees scientific reproducibility.

### 2) Integrity Verification
Every model package includes a SHA256 manifest (`sha256sums.txt`) that hashes the package contents (excluding itself). This enables bit-level reproducibility, tamper detection, and trustworthy deployment in regulated environments.

### 3) Provenance-Friendly Metadata
Each model stores structured metadata via `model_meta.json`:
- training code version (git commit)
- dataset reference (e.g., DVC / object store ref)
- hyperparameters and preprocessing
- `lineage.run_id` linking back to the originating tracking run

### 4) Promotion Workflow
Models move through controlled stages:

```
latest → staging → production
```

All promotions are explicit, append-only, and audited (`audit/promotions.jsonl`).

### 5) Storage Abstraction
v0.2.0 supports a **local filesystem backend** (`localfs`) with a MySQL-backed tracking layer. S3 / Azure Blob backends are on the roadmap.

---

## Repository Structure

```
omnibioai-model-registry/
├── omnibioai_model_registry/
│   ├── api.py
│   ├── config.py            # Settings incl. AUTH_ENABLED/JWT_SECRET/IAM_URL
│   ├── refs.py
│   ├── errors.py
│   ├── run.py               # RunLogger — filesystem-based tracking
│   ├── plugin_client.py     # PluginRunClient — HTTP-based tracking for TES plugins
│   ├── db.py                # MySQL connection + table bootstrap
│   ├── tracking.py          # Pure-SQL tracking functions
│   ├── auth.py              # IAM integration — require_auth/require_write_auth,
│   │                         # model.use permission via omnibioai-iam-client
│   ├── audit_client.py      # Fire-and-forget AuditClient — POSTs to AUDIT_URL
│   │                         # (security-audit), separate from audit/'s local trail
│   ├── hf_routes.py         # Hugging Face push — POST /v1/hf/push, status, settings
│   ├── usage_emit.py        # Usage-metering wrapper around omnibioai-usage-client
│   ├── ownership.py         # Phase 2A — write-once model ownership.json,
│   │                         # legacy backfill (see Organization Ownership)
│   ├── storage/
│   ├── package/
│   ├── audit/                # Local audit trail — audit/promotions.jsonl
│   ├── cli/
│   └── service/
├── frontend/
│   └── omnibioai-model-registry-ui/   # ModelHub UI (React + TypeScript)
├── tests/
├── pyproject.toml
└── README.md
```

---

## Canonical Model Package Layout

Registered models follow a strict, portable structure:

```
<OMNIBIOAI_MODEL_REGISTRY_ROOT>/
tasks/<task>/models/<model_name>/
    versions/<version>/
        model.pt
        model_genes.txt
        label_map.json
        model_meta.json
        metrics.json
        feature_schema.json
        sha256sums.txt
    aliases/
        latest.json
        staging.json
        production.json
    audit/
        promotions.jsonl
    ownership.json
```

`ownership.json` (Phase 2A) is write-once and model-level, not per-version,
and enforced on every model-bearing route (Phase 2B) — see
[Organization Ownership and Enforcement](#organization-ownership-and-enforcement-phase-2a-phase-2b).

This guarantees deterministic loading, integrity validation, and cross-environment portability.

---

## Install, Build, and Use as a Python Package

### 1) Configure registry root

```bash
export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/local_registry/model_registry
```

### 2) Install (editable) for development

```bash
pip install -e .
```

Verify:

```bash
python -c "import omnibioai_model_registry as m; print('OK', m.__file__)"
omr --help
```

### 3) Build a wheel (distribution)

```bash
pip install build
python -m build
```

Artifacts are written to `dist/`:

- `dist/omnibioai_model_registry-0.2.0-py3-none-any.whl`
- `dist/omnibioai_model_registry-0.2.0.tar.gz`

Install the wheel:

```bash
pip install dist/*.whl
```

---

## CLI Usage (`omr`)

11 commands covering the full model lifecycle.

### Register a model package

```bash
omr register \
  --task celltype_sc \
  --model human_pbmc \
  --version 2026-06-14_001 \
  --artifacts /tmp/model_pkg \
  --set-alias latest
```

### Resolve a model reference

```bash
omr resolve --task celltype_sc --ref human_pbmc@latest
```

### Promote a version to production

```bash
omr promote --task celltype_sc --model human_pbmc --version 2026-06-14_001 --alias production
```

### Verify integrity

```bash
omr verify --task celltype_sc --ref human_pbmc@production
```

### Show metadata

```bash
omr show --task celltype_sc --ref human_pbmc@production --json
```

### List models for a task

```bash
omr list --task celltype_sc
```

### Show version metrics and run history

```bash
omr metrics --task celltype_sc --ref human_pbmc@latest
```

### List aliases

```bash
omr aliases --task celltype_sc --model human_pbmc
```

### Set a tag on a model version

```bash
omr tag --task celltype_sc --ref human_pbmc@2026-06-14_001 --key team --value bioml
```

### Set lifecycle stage

```bash
omr stage --task celltype_sc --model human_pbmc --version 2026-06-14_001 --stage production
```

Valid stages: `none`, `staging`, `production`, `archived`.

### Compare metrics across versions

```bash
omr compare --task celltype_sc --model human_pbmc --versions 2026-02-14_001 2026-06-14_001
```

### Backfill legacy ownership records (Phase 2A, operator/admin use)

```bash
omr migrate-ownership --json
```

Deterministic, repeatable, additive-only: writes an explicit
`status="legacy_unowned"` `ownership.json` for every pre-existing model
that has none yet. Never guesses an organization. Safe to re-run. As of
Phase 2B, a `legacy_unowned` model is denied for every caller through
every enforced route, not just left unassigned — see
[Organization Ownership and Enforcement](#organization-ownership-and-enforcement-phase-2a-phase-2b).

---

## Python API Usage

```python
from omnibioai_model_registry import register_model, resolve_model, promote_model

register_model(
    task="celltype_sc",
    model_name="human_pbmc",
    version="2026-06-14_001",
    artifacts_dir="/tmp/model_pkg",
    metadata={
        "framework": "pytorch",
        "model_type": "classifier",
        "provenance": {
            "git_commit": "abc123",
            "training_data_ref": "s3://bucket/datasets/pbmc_v1",
            "trainer_version": "0.2.0",
        },
    },
    set_alias="latest",
    actor="manish",
    reason="initial training",
)

# Resolve by alias (or version)
path = resolve_model("celltype_sc", "human_pbmc@latest", verify=True)
print("Resolved model dir:", path)

# Promote to production
promote_model(
    task="celltype_sc",
    model_name="human_pbmc",
    alias="production",
    version="2026-06-14_001",
    actor="manish",
    reason="validated metrics",
)
```

---

## REST Service (FastAPI)

### Run locally

```bash
pip install -e .
uvicorn omnibioai_model_registry.service.app.main:app --host 0.0.0.0 --port 8095
```

Health check:

```bash
curl -s http://127.0.0.1:8095/health | python -m json.tool
```

### Endpoints

**Registry**

| Method | Path            | Description                          |
| ------ | --------------- | ------------------------------------ |
| POST   | /v1/register    | Register a model version             |
| GET    | /v1/resolve     | Resolve a model reference to a path  |
| POST   | /v1/promote     | Promote a version to an alias        |
| POST   | /v1/verify      | Verify SHA256 integrity              |
| GET    | /v1/show        | Return model_meta.json for a ref     |
| GET    | /v1/models      | List all registered model versions   |

**Tracking** (requires MySQL — HTTP 503 if `DB_HOST` is unset)

| Method | Path                  | Description                        |
| ------ | --------------------- | ---------------------------------- |
| POST   | /v1/runs/log-metric   | Log a single metric point          |
| POST   | /v1/runs/log-param    | Log a single parameter             |
| POST   | /v1/runs/log-batch    | Log metrics, params, and tags      |
| GET    | /v1/runs/get          | Fetch a full run snapshot          |
| GET    | /v1/runs/list         | List runs for a (task, model)      |

**Governance**

| Method | Path                  | Description                                          |
| ------ | --------------------- | ---------------------------------------------------- |
| GET    | /v1/aliases           | List all aliases for a model                         |
| GET    | /v1/metrics           | Return version metrics + step history from DB/JSONL  |
| GET    | /v1/compare           | Compare metrics across two or more versions          |
| GET    | /v1/artifacts         | List files in a version package with SHA256 + sizes  |
| PUT    | /v1/tags              | Set a tag on a model version                         |
| POST   | /v1/versions/patch    | Patch description or tags on a version               |
| POST   | /v1/stage             | Set lifecycle stage (none/staging/production/archived)|
| GET    | /v1/auth/status       | Report whether auth is enabled and, if so, the caller's verified identity |

**Hugging Face**

| Method | Path                       | Description                                          |
| ------ | -------------------------- | ----------------------------------------------------- |
| POST   | /v1/hf/push                | Package a registered model version and push it to the Hugging Face Hub |
| GET    | /v1/hf/push/status/{job_id}| Poll an async push job's status                       |
| GET    | /v1/hf/settings             | Whether a default `HF_TOKEN`/namespace is configured  |

`POST /v1/hf/push` accepts an explicit `token` in the request body, falling
back to the `HF_TOKEN` env var if omitted — never required to be the
caller's own credential. See [Authentication](#authentication) below for
the gate on all mutating routes above, including these three.

### MySQL setup (optional)

When `DB_HOST` is set, the service bootstraps five tables on startup:

```
omr_runs          — run lifecycle (run_id, status, started_at, finished_at,
                     organization_id, ownership_status)  [Phase 2C]
omr_params        — key/value params per run
omr_metrics       — step-indexed metric values per run
omr_tags          — key/value tags per run
omr_version_tags  — key/value tags per model version
```

`organization_id`/`ownership_status` on `omr_runs` (Phase 2C) are added
via an idempotent `ALTER TABLE ... ADD COLUMN` migration run on every
startup (`db.py`'s `_ALTER_DDL`, guarded against MySQL error 1060 so
re-running is always a safe no-op) — see
[Tracking-Data Organization Isolation](#tracking-data-organization-isolation-phase-2c)
below for the full design and legacy-row migration reasoning.
`omr_params`/`omr_metrics`/`omr_tags`/`omr_version_tags` have no schema
change; they inherit their organization boundary from `omr_runs`/
`ownership.json` respectively, never a duplicated column.

Environment variables:

```bash
export DB_HOST=localhost
export DB_PORT=3306
export DB_USER=omr
export DB_PASSWORD=secret
export DB_NAME=model_registry
```

When `DB_HOST` is absent, the service runs in filesystem-only mode. Tracking endpoints return HTTP 503; all registry and governance endpoints remain fully functional.

---

## Authentication

Off by default; **every non-informational endpoint** — both mutations
(`register`, `promote`, `tags`, `versions/patch`, `stage`, the `hf/*`
routes, `runs/log-*`) and reads (`resolve`, `verify`, `show`, `models`,
`runs/get`, `runs/list`, `metrics`, `aliases`, `compare`, `artifacts`,
`hf/push/status/{job_id}`) — is IAM-gated when explicitly enabled:

```bash
export AUTH_ENABLED=true
export JWT_SECRET=...      # HS256 fallback secret, matches omnibioai-auth's SECRET_KEY
export IAM_URL=http://auth-service:8001
```

`AUTH_ENABLED=false` (the default) runs the service in open mode — no
token required, every call attributed to a synthetic `system` actor. When
enabled, `require_auth`/`require_write_auth` (`auth.py`) verify the
presented JWT via `omnibioai-iam-client`'s `AsyncIAMClient.get_user()`
(RS256/JWKS-or-HS256 signature check + revocation check against
`omnibioai-auth`, no local JWT decoding of its own) and require the
`model.use` permission — the same IAM pattern `omnibioai-lims` and
`omnibioai-api-gateway` use. **The registry performs this verification
itself, independently of the API Gateway** — it never trusts
gateway-injected identity headers (`X-Organization-ID`, `X-Team-ID`,
`X-User-ID`, `X-User-Email`, or similar) as a substitute for a verified
Bearer token. The Gateway remains a real enforcement layer of its own
(it authenticates and policy-checks every request before forwarding),
but the registry no longer depends on it as the *only* layer — a
request that reaches the service directly is now held to the same
standard as one that arrives via the Gateway. `GET /v1/auth/status`,
`GET /v1/hf/settings`, and `GET /health` remain public — they carry no
registry data, only service/mode metadata.

**Scope note:** authorization is still permission-based first —
`model.use` remains the existing flat, non-resource-scoped permission,
unchanged and not redesigned by Phase 2B. On top of that, every
model-bearing read/write route now independently enforces organization
ownership (see
[Organization Ownership](#organization-ownership-and-enforcement-phase-2a-phase-2b) below):
holding `model.use` is necessary but no longer sufficient — the caller's
verified `org_id` must also match the model's recorded owner (or, for
`omr_*` run-tracking routes as of Phase 2C, the run's own recorded
owner — see
[Tracking-Data Organization Isolation](#tracking-data-organization-isolation-phase-2c)).

### Organization Ownership and Enforcement (Phase 2A, Phase 2B)

Every newly registered model now has a durable, server-derived
organization-ownership record, stored as `ownership.json` at the model
root (`tasks/<task>/models/<model_name>/ownership.json` — one per model,
not per version; versions inherit their parent model's ownership by
construction, they carry no independent ownership field). This is
**not** part of the SHA256-hashed version manifest (that only covers
files inside a version directory) — it is a separate, model-level,
write-once file next to `aliases/` and `audit/`.

- **Source of truth**: the filesystem. MySQL's `omr_*` tables are an
  optional, `DB_HOST`-gated side-store for experiment-tracking metrics
  only (see below) and never represent models/versions themselves — the
  service is fully functional with `DB_HOST` unset, so ownership cannot
  depend on MySQL being configured. There is exactly one source of truth
  for ownership; it is not duplicated into MySQL or into
  `model_meta.json`.
- **Where `organization_id` comes from**: only the caller's
  already-verified IAM identity (`UserContext.org_id`, resolved by
  `require_write_auth_with_context` after JWT verification). Never from
  the request body, query/path parameters, or any header
  (`X-Organization-ID`, `X-Team-ID`, `X-User-ID`, `X-User-Email`, or a
  client-supplied `organization_id` key inside the free-form `metadata`
  dict) — none of those are ever consulted.
- **Write-once**: the very first successful registration of a given
  `task`/`model_name` establishes ownership permanently. Every later
  version registered under that same model inherits the existing record
  unchanged; ownership is never reassigned by a subsequent write. As of
  Phase 2B, a different org's attempt to register a version under an
  existing model is also actively **denied** (not just ignored) — see
  below.
- **Legacy models**: any model that already had a registered version
  before this phase shipped (or is otherwise touched for the first time
  post-Phase-2A with no `ownership.json` yet) is recorded as
  `status="legacy_unowned"` with `organization_id=null` — **never**
  guessed from actor strings or assigned to whichever org happens to
  touch it next. Resolving these into real ownership is an explicit,
  deferred, administrator/manual-assignment step (Phase 2B+), not
  something this phase attempts. Run `omr migrate-ownership` to
  proactively backfill this explicit marker for every legacy model in
  one deterministic, repeatable, additive-only pass (safe to re-run).
- **CLI**: `omr register --org-id <id>` lets an operator assign ownership
  explicitly for a brand-new model when registering out-of-band (the CLI
  has no JWT/IAM identity of its own); it has no effect on an
  already-owned or legacy model.
- **`POST /v1/register`'s response** now includes server-derived
  `organization_id` and `ownership_status` fields (`"owned"`,
  `"unowned"`, or `"legacy_unowned"`). No other endpoint's response
  shape changed in this phase.

**Phase 2B — enforcement.** Every model-bearing read/write route above
now independently checks
`verified_user.org_id == model_owner.organization_id` via
`ownership.check_model_ownership()`, the single centralized decision
function every route (or the underlying `ModelRegistry` method it goes
through) calls before touching filesystem state:

- **Reads**: `resolve`, `verify`, `show`, `models` (list — filtered
  during the same filesystem walk that already reads each entry, not
  fetched-then-hidden in the response), `compare`, `artifacts`,
  `aliases`, `metrics`.
- **Writes**: `register` (a version under an *existing* model requires
  belonging to its owner; a brand-new model is unaffected — see
  Write-once above), `promote`, `tags`, `versions/patch`, `stage`.
- **`POST /v1/hf/push`**: resolves the model/version and checks
  ownership *before* any Hugging Face API call — a cross-org push is
  denied before an artifact is ever read, let alone uploaded.
  **`GET /v1/hf/push/status/{job_id}`** is org-scoped too as of Phase 2D
  — the job itself now carries the `organization_id` of whoever started
  the push (recorded at `POST /v1/hf/push` time), and polling status
  requires the same match; previously any `model.use` holder could poll
  any job_id (a documented, bounded enumeration/status-leak gap, never a
  secret-disclosure one — job_id is an unguessable uuid4 and the response
  never contains the HF token — but explicitly flagged since Phase 1 as
  "deferred to the tenant-isolation phase").
- **Denial shape**: anti-enumerating by construction — a cross-org
  caller gets the exact same "not found" response (same status code,
  same message) a genuinely nonexistent model would return from that
  route. `legacy_unowned` models are denied for *every* caller,
  including one with no org context at all — never silently claimed by
  whoever touches them next (still no admin-reassignment workflow; see
  "Not yet implemented" below).
- **Audit**: `register_model`/`promote_model`/`set_tag`/`patch_version`/
  `set_stage`/`push_to_hf` events all carry `organization_id` directly in
  their metadata now, not only inferable by correlating with a separate
  `model_access_*` event.

**Not yet implemented — and why (Phase 2D audit conclusion):**

- **A legacy/unowned ownership-resolution workflow.** Phase 2D
  explicitly audited this (see the Phase 2D PR description for the full
  writeup) rather than building it, because this repo's own established
  convention — stated in `auth.py`'s own module docstring — is
  "authorize by permission, never by role," and neither candidate
  IAM signal already available to this service cleanly satisfies that
  without a cross-repo product decision:
  - `UserContext.org_role` containing `"org_admin"` (omnibioai-auth's
    standard per-org admin role, already present on the verified
    `UserContext` this service already trusts) would work mechanically
    and is naturally self-scoping (the admin's own `org_id` is exactly
    the org they'd resolve resources into) — but it's a *role* name, not
    a *permission*, a first-ever precedent-breaking pattern in this repo.
  - `manage_all_orgs` (the existing platform-admin *permission*, which
    genuinely does flow through the JWT `permissions` claim the same way
    `model.use` does) fits the "permission, not role" precedent — but
    it's platform-wide, not org-scoped, so using it here would require a
    target `organization_id` to come from somewhere *other* than the
    caller's own verified `org_id` for the first time ever in this
    ownership model, breaking the "organization_id never client-supplied"
    invariant every phase since 2A has held.

  Either is implementable, but picking one is a product/IAM decision,
  not something to invent unilaterally inside this repo.
- Resource-scoped `model.use` (authorization is still
  ownership-check-plus-flat-permission, not a redesigned permission
  model) — not needed for the above either way, since both candidate
  designs use an *existing* IAM signal rather than requiring `model.use`
  itself to change shape.
- A `platform`/public model namespace; model deletion.

**Path traversal — fixed, in a dedicated follow-up PR, not Phase 2D
itself.** Phase 2D identified (but by its own explicit charter did not
fix, since it is independent of tenant isolation — it would exist in a
single-tenant deployment too) that `task`/`model_name`/`version`/`alias`
request fields had no `..`-sequence sanitization anywhere in `refs.py`/
`package/layout.py`/`storage/localfs.py`. That gap is now closed — see
[Filesystem Path Safety](#filesystem-path-safety) below for the full
design (a centralized allowlist validator, applied once in
`package/layout.py`, inherited by every consumer).

### Tracking-Data Organization Isolation (Phase 2C)

Extends organization-scoped enforcement from the filesystem model
registry (Phase 2A/2B, above) to the optional MySQL tracking layer
(`omr_runs`/`omr_params`/`omr_metrics`/`omr_tags`/`omr_version_tags`),
without a second, possibly-diverging source of truth for the same
question.

- **Two independent ownership authorities, deliberately not merged.**
  `ownership.json` (Phase 2A) remains the sole authority for *model*
  ownership. `omr_runs.organization_id` (new) is the sole authority for
  *run* ownership — **not** derived from `ownership.json`, because a
  run's `task`/`model_name` has no required correspondence to any
  registered model at all: logging metrics/params *during* an
  experiment, then registering the resulting model *afterward*, is the
  whole point of this tracking layer, so `ownership.json` frequently
  won't exist yet (or ever) for a given run. Neither authority can
  override the other.
- **Where it lives**: `organization_id`/`ownership_status` columns
  directly on `omr_runs`, populated write-once (first `log-metric`/
  `log-param`/`log-batch` call for a given `run_id`) from the caller's
  verified `UserContext.org_id` — the exact same "record it once, never
  reassign" pattern Phase 2A established for models, just at the row
  level instead of a separate file. `omr_params`/`omr_metrics`/
  `omr_tags` get no column of their own; they inherit their boundary via
  the existing `run_id` foreign key. `omr_version_tags` gets no column
  either — it's keyed directly by `(task, model_name, version)`, a real
  model's version, so it's checked against `ownership.json` instead
  (`tracking.set_version_tag()`/`get_version_tags()` now accept the same
  `enforce_ownership`/`requesting_org_id` kwargs api.py's `ModelRegistry`
  methods already do).
- **Enforced routes**: `POST /v1/runs/log-metric|log-param|log-batch`,
  `GET /v1/runs/get|list`. `GET /v1/metrics`'s own run-history lookup is
  deliberately **not** run-ownership-gated — its `run_id` isn't
  caller-supplied, it's read out of the model's own `model_meta.json`,
  reachable only because `resolve_model()` already enforced *model*
  ownership on `(task, model_name)`; gating it a second time on
  `omr_runs.organization_id` too would incorrectly lock a model's
  legitimate owner out of their own run history whenever that run
  predates this migration (see below) or was logged before the model
  existed.
- **Legacy rows**: every `omr_runs` row that existed before this
  migration gets `organization_id=NULL, ownership_status='legacy_unowned'`
  automatically (MySQL applies a column's `DEFAULT` to every existing row
  on `ALTER TABLE ... ADD COLUMN`) — denied for every caller, including
  one with no org context, identical to Phase 2A/2B's `legacy_unowned`
  model policy. **Deliberately not** backfilled by matching `(task,
  model_name)` against `ownership.json`: pre-Phase-2A this service
  enforced no uniqueness on those strings at all, so two different
  orgs' legacy runs could coincidentally share the same `task`/
  `model_name` — a string-match backfill risked attributing one org's
  historical run data to a different org. Never guessed from the
  `actor` column either.
- **Anti-enumeration**: every denial (cross-org, legacy, or genuinely
  missing) raises the exact same "Run not found" `ModelNotFound`
  already used pre-Phase-2C for a missing `run_id` — same status code,
  same message shape.
- **`GET /v1/runs/list`** filters at the SQL layer (`WHERE ... AND
  (organization_id/ownership_status ...)`), not fetch-all-then-hide.
- **Concurrency**: run creation is `INSERT IGNORE` on `run_id`'s
  `PRIMARY KEY` — atomic at the MySQL engine level, the same reasoning
  `ownership.json`'s `os.link`-based write-once relies on, just via a
  different primitive.
- **Audit**: `log_metric`/`log_param`/`log_batch` now emit audit events
  (none existed pre-Phase-2C) carrying `organization_id` — `log_param`'s
  event deliberately never includes the logged `value` (caller-supplied
  `Any`, unlike a metric's plain float) to avoid a param value that
  happens to be a secret ending up in the audit trail.

### Filesystem Path Safety

A standalone security hardening pass, independent of (and layered
*underneath*, not instead of) the organization-ownership enforcement
above — this section is about **where on disk** a request is allowed to
touch, not **which organization** is allowed to touch it. This is not a
HIPAA compliance claim or certification of any kind; it is a description
of what this specific hardening pass does and does not cover.

**The invariant**: no user-controlled model/task/version/alias/run/
metric identifier may cause any filesystem operation to resolve outside
the configured `OMNIBIOAI_MODEL_REGISTRY_ROOT`, except the CLI (which
already has direct filesystem access and is not reachable over HTTP/JWT
at all — the same trust boundary Phase 2A's `omr register --org-id`
precedent already established).

**Design — allowlist, not blocklist.** `path_safety.py`'s
`safe_component()` requires every identifier (task, model_name, version,
alias, run_id, metric_key) to match a strict character allowlist
(letters, digits, `_`, `-`, and interior `.`; must start and end with an
alphanumeric character; ≤128 characters) *before* it is ever joined into
a `Path`. Rejecting known-bad patterns one at a time (`..`, absolute
paths, encoded variants, backslashes, repeated separators, ...) is
inherently incomplete — there is always another variant; a component
that satisfies the allowlist is *lexically* incapable of forming a path
separator, a traversal sequence, or an absolute-path prefix on any
platform, which is a stronger guarantee than "reject `..` after the
fact". Legitimate identifiers this system's own docs/tests already use
(`human_pbmc`, `2026-02-13_001`, `staging`) are unaffected.

**Where it's applied**: once, centrally, inside `package/layout.py`'s
path-builder functions — the single choke point every filesystem path in
this codebase is already constructed through (confirmed by inspection:
no other module builds a task/model/version/alias/run path by hand).
Every consumer (`api.py`, `ownership.py`, `service/app/main.py`,
`run.py`, `cli/main.py`) inherits this automatically, with no
per-call-site change needed. `refs.py`'s `parse_model_ref()` also
validates model_name/selector directly, for an earlier, clearer error —
defense-in-depth, since every path it feeds into validates independently
anyway.

**Defense-in-depth**: `path_safety.py`'s `assert_contained()` is a
second, independent layer at the handful of actual mutating/returning
operations in `api.py` (`register_model`'s copy destination,
`promote_model`'s alias write, `resolve_model`'s returned version
directory) — re-verifies against the real, symlink-resolved filesystem
location, catching a hypothetical future bug in the allowlist itself,
not serving as the primary defense.

**Symlink policy**: symlink-*aware*, not symlink-*hostile*.
`assert_contained()` uses `Path.resolve()` (the same mechanism
`config.py`/`run.py`'s own `_resolve_registry_root()` already uses for
`registry_root` itself), so a symlinked registry root — a plausible
legitimate deployment layout, e.g. network-mounted storage — is
*followed*, not rejected, as long as the real, final location is still
under the real, resolved root. A symlink placed *inside* the managed
tree that points outside the root is rejected, because its real
destination fails that same containment check. This service does not
itself create any symlinks anywhere in the tree it manages, so there is
no legitimate in-repo symlink layout to special-case beyond "the root
itself may be one."

**Invalid-path error behavior**: a rejected identifier raises
`PathTraversalError` (a `ValidationError`/`ModelRegistryError`), which
every route either already maps to a generic HTTP 400 via the existing
`_handle_registry_error` handler, or — for the couple of routes
(`aliases`, `compare`) that construct paths directly without a local
`try`/`except` — a dedicated FastAPI exception handler in
`service/app/main.py` catches directly, so the outcome is identical
either way. The error message names only the *field* (e.g. `"Invalid
model_name"`) — never the submitted value, a resolved filesystem path,
internal directory structure, or a stack trace.

**`artifacts_dir` trust boundary** (`register_model()`'s server-local
source directory, not an identifier — a fixed character allowlist
doesn't apply to it the way it does to task/model_name/version, since it
legitimately needs to be anywhere a training job wrote its output on the
server's own filesystem):

- **Always enforced, no configuration**: `artifacts_dir` may never
  resolve to a location *inside* the registry root itself. Without this,
  any authenticated `model.use` holder could point `artifacts_dir` at an
  *already-registered* version directory — including one belonging to a
  different organization — and have it copied into a brand-new
  registration they own, then read it back via their own,
  legitimately-authorized `/v1/artifacts` or `/v1/resolve`: a complete,
  silent bypass of every Phase 2A–2D ownership check through a path no
  ownership check ever inspects. Never a legitimate use case under any
  deployment, so this is a structural rule, not a policy choice.
- **Optional, operator-configured**: `OMNIBIOAI_MODEL_REGISTRY_ARTIFACTS_ALLOWED_ROOTS`
  (comma-separated absolute paths) further restricts *which* server-local
  directory trees `artifacts_dir` may be under. Unset (the default) is
  intentionally unrestricted, preserving every existing deployment's
  behavior — this service has no established convention for where
  training-output directories live on a given server, so picking a
  mandatory default would be inventing a policy rather than enforcing
  one a deployment already has; operators who want this narrowed can opt
  in.

**Not covered by this hardening pass** (unchanged, and out of scope by
this PR's own charter): SQL parameterization (`tracking.py`'s queries
were already parameterized, not string-built, and untouched here);
resource exhaustion / disk-fill from a very large `artifacts_dir` when
no allowlist is configured; anything about *which organization* may
access a given, validly-contained path — that remains entirely governed
by Phase 2A–2D's `check_model_ownership()`/`check_run_ownership()`,
unmodified and unweakened by this pass.

### Observability side effects of every write

- **Audit** — `audit_client.py`'s `AuditClient` fire-and-forget-POSTs an
  event to `AUDIT_URL` (the security-audit service) on
  register/promote/set_tag/set_stage, in addition to this repo's own
  local `audit/promotions.jsonl` trail described under "Core Design
  Principles" above — two separate audit records, not one.
- **Usage metering** — `usage_emit.py` wraps `omnibioai-usage-client` to
  emit a `model.register` usage event on every registration
  (`service="model-registry"`), fail-open by design: a metering failure
  never blocks or fails the registration itself.

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Relationship to OmniBioAI Ecosystem

The ModelHub is a **control-plane component** of OmniBioAI.

Companion repositories:

- **omnibioai** → AI-powered bioinformatics workbench
- **omnibioai-tes** → execution orchestration across local/HPC/cloud
- **omnibioai-rag** → reasoning and literature intelligence
- **omnibioai-lims** → laboratory data management
- **omnibioai-workflow-bundles** → reproducible pipelines
- **omnibioai-sdk** → Python client access

The **ModelHub** provides the AI artifact governance layer shared by all.

---

## Roadmap

### Done (v0.2.0)

- Experiment tracking with `RunLogger` + `PluginRunClient`
- MySQL-backed run/metric/param/tag storage
- ModelHub UI with Experiments tab + metric sparklines
- Stage management (`none` → `staging` → `production` → `archived`)
- Alias listing, metric comparison, artifact browser endpoints
- **HIPAA hardening Phase 1** — every non-informational read endpoint
  now independently requires IAM authentication (`model.use`), closing
  the previously-unauthenticated read-path gap; see
  [Authentication](#authentication). Tenant/organization isolation is
  explicitly **not** part of this phase.
- **HIPAA hardening Phase 2A** — durable, server-derived organization
  ownership for newly registered models; see
  [Organization Ownership](#organization-ownership-and-enforcement-phase-2a-phase-2b).
  Establishes *who owns this model*, server-controlled and
  IAM-derived — it does **not** yet *enforce* that ownership anywhere.
  `model.use` is unchanged (still flat, not resource-scoped). Legacy
  models remain explicitly `legacy_unowned`, not guessed.
- **HIPAA hardening Phase 2B** — turns Phase 2A's ownership record into
  real enforcement; see
  [Organization Ownership and Enforcement](#organization-ownership-and-enforcement-phase-2a-phase-2b).
  Every model-bearing read/write route (including `POST /v1/hf/push`)
  now independently denies a caller whose verified `org_id` doesn't
  match the model's owner, anti-enumerating by construction.
  `legacy_unowned` models remain unclaimed by design — still no
  admin-reassignment workflow (deliberately not invented in this phase
  either; needs its own product/audit decision). `model.use` remains
  unchanged (flat, not resource-scoped) — org enforcement sits alongside
  it, not inside a redesigned permission model.
- **HIPAA hardening Phase 2C** — extends organization isolation to the
  optional MySQL tracking layer; see
  [Tracking-Data Organization Isolation](#tracking-data-organization-isolation-phase-2c).
  `omr_runs` gets its own independent `organization_id`/
  `ownership_status` columns (never derived from `ownership.json` — a
  run's task/model_name has no required correspondence to a registered
  model); `omr_params`/`omr_metrics`/`omr_tags` inherit via the `run_id`
  FK; `omr_version_tags` is checked against `ownership.json` instead,
  since it IS a real model's version. Legacy rows (pre-migration)
  become `legacy_unowned` automatically via the column's `ALTER TABLE`
  default, denied for everyone, never guessed from `actor` or
  backfilled by string-matching `task`/`model_name` against
  `ownership.json` (a real, considered, and rejected option — see the
  section above for why).
- **HIPAA hardening Phase 2D** — final tenant-isolation audit across
  every model/run/version-tag/HF-push route from Phases 2A–2C (no
  bypass, alternate-org-source, gateway-header-trust, mutation-after-
  denial, or enumeration-oracle found), plus one real fix it surfaced:
  `GET /v1/hf/push/status/{job_id}` is now org-scoped too (see
  [Organization Ownership and Enforcement](#organization-ownership-and-enforcement-phase-2a-phase-2b)
  above). Audited legacy-ownership resolution end to end and
  deliberately did **not** build it — see "Not yet implemented" above
  for the two candidate IAM designs considered and why neither was
  adopted without a product decision. Re-confirmed the pre-existing
  path-traversal finding is unrelated to tenant isolation (doesn't
  bypass any ownership check) and left it for its own dedicated PR.
- **Filesystem path traversal & storage boundary hardening** — the
  dedicated follow-up PR Phase 2D flagged; see
  [Filesystem Path Safety](#filesystem-path-safety). Centralized
  allowlist validation for every task/model/version/alias/run/metric
  identifier (`path_safety.py`, applied once in `package/layout.py`),
  defense-in-depth containment checks, a closed `artifacts_dir`
  cross-org-ingestion vector (unrelated to but discovered alongside the
  originally-flagged issue), and a fixed `/v1/metrics` fallback that
  previously trusted attacker-influenceable `model_meta.json` fields
  for its own path construction. Symlink-aware, not symlink-hostile.
  Independent of, and layered underneath, Phase 2A–2D's organization
  enforcement — not a tenant-isolation change.

### Near Term

- S3 / Azure Blob storage backends
- Step-history sparklines in UI pulled from DB (currently single-point)
- Model signature validation (input/output schema enforcement)
- **HIPAA hardening Phase 2E (product/IAM decision needed first)** —
  either adopt `org_role` containing `"org_admin"` as a legitimate
  role-based signal for this service (a first, deliberate exception to
  "authorize by permission, never by role"), or register + issue a new
  org-scoped permission via omnibioai-auth's permission registry and
  have this service check it instead — then build the actual
  legacy/unowned ownership-resolution workflow (model *and* run) on top
  of whichever is chosen. Resource-scoped `model.use`; `platform`/public
  model namespace; model deletion remain separately unscoped.

### Mid Term

- Parallel coordinates plot for hyperparameter search
- Auto-link `run_id` → model version in UI (Registered As chip)
- Pagination + filtering on `GET /v1/models` and `GET /v1/runs/list`
- Promotion policies (metric threshold gates)

### Long Term

- Regulatory-ready audit and lineage export (PDF/CSV)
- Enterprise biomedical AI governance platform
- Deeper LIMS integration (sample → dataset → run → model chain)
