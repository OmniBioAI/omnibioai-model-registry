# syntax=docker/dockerfile:1
# ── Stage 1: Build React UI ────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM node:20-bookworm-slim AS ui-builder
WORKDIR /ui
COPY omnibioai-model-registry/frontend/omnibioai-model-registry-ui/package*.json ./
RUN npm ci
COPY omnibioai-model-registry/frontend/omnibioai-model-registry-ui/ ./
RUN npm run build

# ── Stage 2: Python API + nginx ───────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS backend
LABEL org.opencontainers.image.source=https://github.com/man4ish/omnibioai

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY omnibioai-model-registry/pyproject.toml .

# Model Registry IAM integration: pyproject.toml declares omnibioai-iam-client
# as a pinned git+https dependency (private repo -- GitHub Packages has no
# PyPI-format registry). `pip install .` re-resolves ALL declared deps
# including direct-URL ones (it does NOT trust a same-named package already
# being installed the way it does for plain version-range requirements), so
# the token must be available for pip's own git clone here. Uses a BuildKit
# secret mount (not ARG) -- ARG values get echoed into BuildKit's progress
# output for the RUN instruction that uses them, leaking the token into
# build logs. A secret mount is never printed and never persists in any
# image layer.
RUN --mount=type=secret,id=github_token \
    git config --global url."https://$(cat /run/secrets/github_token)@github.com/".insteadOf "https://github.com/" \
 && pip install --no-cache-dir -U pip && pip install --no-cache-dir . \
 && git config --global --unset url."https://$(cat /run/secrets/github_token)@github.com/".insteadOf
COPY omnibioai-model-registry/omnibioai_model_registry/ ./omnibioai_model_registry/

COPY --from=ui-builder /ui/dist /usr/share/nginx/html

RUN printf 'server {\n\
    listen 5176;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location /_svc/modelregistry/ { alias /usr/share/nginx/html/; try_files $uri $uri/ /_svc/modelregistry/index.html; }\n\
    location / { try_files $uri $uri/ /index.html; }\n\
    location /v1/ { proxy_pass http://127.0.0.1:8095; proxy_set_header Host $host; }\n\
    location /health { proxy_pass http://127.0.0.1:8095; }\n\
    location /docs { proxy_pass http://127.0.0.1:8095; }\n\
}\n' > /etc/nginx/sites-available/default

ENV HOST=0.0.0.0 PORT=8095 PYTHONUNBUFFERED=1
ENV MODEL_REGISTRY_APP=omnibioai_model_registry.service.app.main:app
EXPOSE 8095 5176
CMD ["bash", "-c", "nginx && python -m uvicorn omnibioai_model_registry.service.app.main:app --host 0.0.0.0 --port 8095"]
