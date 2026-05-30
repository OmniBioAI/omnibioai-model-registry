FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install deps
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir .

# Copy package (non-src layout)
COPY omnibioai_model_registry /app/omnibioai_model_registry

ENV HOST=0.0.0.0
ENV PORT=8095
ENV MODEL_REGISTRY_APP=omnibioai_model_registry.service.app.main:app

EXPOSE 8095

CMD ["python", "-m", "uvicorn", "omnibioai_model_registry.service.app.main:app", "--host", "0.0.0.0", "--port", "8095"]


