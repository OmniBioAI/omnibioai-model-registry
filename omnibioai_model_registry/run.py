# File: omnibioai_model_registry/run.py
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from .package.layout import (
    run_metric_log_path,
    run_params_path,
    run_tags_path,
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_registry_root(registry_root: Optional[str | Path]) -> Path:
    if registry_root is not None:
        return Path(registry_root).expanduser().resolve()
    from .config import load_config
    cfg = load_config()
    return Path(cfg.root).expanduser().resolve()


class RunLogger:
    """
    Workbench plugin SDK entry point for logging training runs.

    Each instance represents one training run, identified by ``run_id``
    (auto-generated as a UUID hex string if not provided).

    Filesystem layout produced::

        {registry_root}/tasks/{task}/models/{model_name}/runs/{run_id}/
            params.json        -> {"lr": 0.001, "epochs": 50}
            tags.json          -> {"team": "bioml"}
            metrics/
                accuracy.jsonl -> {"key":"accuracy","value":0.95,"step":0,"ts_utc":"..."}
                                  {"key":"accuracy","value":0.97,"step":1,"ts_utc":"..."}

    Usage::

        with RunLogger(task="celltype_sc", model_name="human_pbmc") as run:
            run.log_params({"lr": 0.001, "epochs": 50})
            for step, acc in enumerate(training_curve):
                run.log_metric("accuracy", acc, step=step)
            run.set_tag("team", "bioml")

        # Link the run to the registered model version:
        register_model(..., metadata={"lineage": {"run_id": run.run_id, ...}})
    """

    def __init__(
        self,
        task: str,
        model_name: str,
        run_id: Optional[str] = None,
        registry_root: Optional[str | Path] = None,
    ) -> None:
        self._root = _resolve_registry_root(registry_root)
        self._task = task
        self._model_name = model_name
        self._run_id: str = run_id if run_id is not None else uuid4().hex
        self._params: Dict[str, Any] = {}
        self._tags: Dict[str, str] = {}

    # ── property ──────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    # ── params ────────────────────────────────────────────────────────────────

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value
        _atomic_write_json(
            run_params_path(self._root, self._task, self._model_name, self._run_id),
            self._params,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        self._params.update(params)
        _atomic_write_json(
            run_params_path(self._root, self._task, self._model_name, self._run_id),
            self._params,
        )

    # ── metrics ───────────────────────────────────────────────────────────────

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        _append_jsonl(
            run_metric_log_path(
                self._root, self._task, self._model_name, self._run_id, key
            ),
            {"key": key, "value": value, "step": step, "ts_utc": _now_utc_iso()},
        )

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    # ── tags ──────────────────────────────────────────────────────────────────

    def set_tag(self, key: str, value: str) -> None:
        self._tags[key] = value
        _atomic_write_json(
            run_tags_path(self._root, self._task, self._model_name, self._run_id),
            self._tags,
        )

    def set_tags(self, tags: Dict[str, str]) -> None:
        self._tags.update(tags)
        _atomic_write_json(
            run_tags_path(self._root, self._task, self._model_name, self._run_id),
            self._tags,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def finish(self) -> str:
        """Flush final state to disk and return the run_id."""
        if self._params:
            _atomic_write_json(
                run_params_path(self._root, self._task, self._model_name, self._run_id),
                self._params,
            )
        if self._tags:
            _atomic_write_json(
                run_tags_path(self._root, self._task, self._model_name, self._run_id),
                self._tags,
            )
        return self._run_id

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()
