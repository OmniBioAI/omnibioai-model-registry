"""
plugin_client.py — lightweight HTTP client for the OmniBioAI Model Registry.

Lets training jobs running outside the registry service log params, metrics,
and tags back to the registry's REST API without importing the full SDK.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from uuid import uuid4


class PluginClient:
    """
    HTTP client that mirrors the RunLogger interface but POSTs to a remote
    registry service instead of writing to the local filesystem.

    Usage::

        with PluginClient(registry_url="http://registry:8000") as client:
            client.start(task="celltype_sc", model_name="human_pbmc")
            client.log_params({"lr": 0.001, "epochs": 50})
            for step, acc in enumerate(curve):
                client.log_metric("accuracy", acc, step=step)
            client.set_tag("team", "bioml")
    """

    def __init__(self, registry_url: Optional[str] = None) -> None:
        url = registry_url or os.getenv("OMNIBIOAI_REGISTRY_URL", "http://localhost:8000")
        self._url = url.rstrip("/")
        self._run_id: Optional[str] = None
        self._task: str = ""
        self._model_name: str = ""

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, task: str = "", model_name: str = "") -> str:
        self._task = task
        self._model_name = model_name
        self._run_id = uuid4().hex
        self._post(
            "/v1/runs/start",
            {"run_id": self._run_id, "task": task, "model_name": model_name},
        )
        return self._run_id

    def finish(self) -> Optional[str]:
        """Notify the registry that the run is complete and return the run_id."""
        if self._run_id:
            self._post("/v1/runs/finish", {"run_id": self._run_id})
        return self._run_id

    # ── params ────────────────────────────────────────────────────────────────

    def log_param(self, key: str, value: Any) -> None:
        self._post(
            "/v1/runs/log_param",
            {"run_id": self._run_id, "key": key, "value": value},
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    # ── metrics ───────────────────────────────────────────────────────────────

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self._post(
            "/v1/runs/log_metric",
            {"run_id": self._run_id, "key": key, "value": value, "step": step},
        )

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    # ── tags ──────────────────────────────────────────────────────────────────

    def set_tag(self, key: str, value: str) -> None:
        self._post(
            "/v1/runs/set_tag",
            {"run_id": self._run_id, "key": key, "value": value},
        )

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "PluginClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()

    # ── internal ──────────────────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Optional[dict]:
        """POST JSON payload to endpoint; returns parsed response or None on error."""
        url = self._url + endpoint
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return {"error": str(exc), "code": exc.code}
        except Exception:
            return None
