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
- **IAM-gated writes** — every mutating endpoint requires a valid JWT (via `omnibioai-iam-client`) carrying the `model.use` permission, when `AUTH_ENABLED=true`
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
- ✅ 11 CLI commands
- ✅ IAM-gated writes (`model.use` permission, `omnibioai-iam-client`)
- ✅ Usage metering + cross-service audit emission
- ✅ ModelHub UI with Experiments tab + metric sparklines
- ✅ Local-first, cloud-ready design

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
```

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
omr_runs          — run lifecycle (run_id, status, started_at, finished_at)
omr_params        — key/value params per run
omr_metrics       — step-indexed metric values per run
omr_tags          — key/value tags per run
omr_version_tags  — key/value tags per model version
```

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

Off by default; every mutating endpoint (`register`, `promote`, `tags`,
`versions/patch`, `stage`, the `hf/*` routes, `runs/log-*`) is
IAM-gated when explicitly enabled:

```bash
export AUTH_ENABLED=true
export JWT_SECRET=...      # HS256 fallback secret, matches omnibioai-auth's SECRET_KEY
export IAM_URL=http://auth-service:8001
```

`AUTH_ENABLED=false` (the default) runs the service in open mode — no
token required, every call attributed to a synthetic `system` actor. When
enabled, `require_write_auth` (`auth.py`) verifies the presented JWT via
`omnibioai-iam-client`'s `AsyncIAMClient.get_user()` (RS256/JWKS-or-HS256
signature check + revocation check against `omnibioai-auth`, no local JWT
decoding of its own) and requires the `model.use` permission — the same
IAM pattern `omnibioai-lims` and `omnibioai-api-gateway` use. `GET
/v1/auth/status` reports whether auth is on and, if so, the caller's
resolved identity — useful for a client to detect which mode it's
talking to.

**Scope note:** authorization is permission-based, not org-scoped. The
verified identity's `organization_id` is attached to every audit/usage
event this service emits, but nothing in the data model
(`omr_runs`/`omr_params`/`omr_metrics`/`omr_tags`/`omr_version_tags`, or
models/versions themselves) has an `organization_id` column — every
organization currently shares one flat model namespace. Per-org isolation
is real future work (see Roadmap), not something `AUTH_ENABLED=true`
already provides.

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

### Near Term

- S3 / Azure Blob storage backends
- Step-history sparklines in UI pulled from DB (currently single-point)
- Model signature validation (input/output schema enforcement)
- Per-organization data isolation — `model.use` permission gating (see [Authentication](#authentication)) is real today, but the data model has no `organization_id` anywhere; every org currently shares one flat model namespace

### Mid Term

- Parallel coordinates plot for hyperparameter search
- Auto-link `run_id` → model version in UI (Registered As chip)
- Pagination + filtering on `GET /v1/models` and `GET /v1/runs/list`
- Promotion policies (metric threshold gates)

### Long Term

- Regulatory-ready audit and lineage export (PDF/CSV)
- Enterprise biomedical AI governance platform
- Deeper LIMS integration (sample → dataset → run → model chain)
