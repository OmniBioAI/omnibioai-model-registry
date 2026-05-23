# ── Stage 1: Build React UI ────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM node:20-bookworm-slim AS ui-builder
WORKDIR /ui
COPY frontend/omnibioai-model-registry-ui/package*.json ./
RUN npm ci
COPY frontend/omnibioai-model-registry-ui/ ./
RUN npm run build

# ── Stage 2: Python API + nginx ───────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS backend
LABEL org.opencontainers.image.source=https://github.com/man4ish/omnibioai

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .
COPY omnibioai_model_registry/ ./omnibioai_model_registry/

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
