# File: omnibioai_model_registry/package/layout.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..path_safety import safe_component

REQUIRED_FILES = [
    "model.pt",
    "model_genes.txt",
    "label_map.json",
    "model_meta.json",
    "metrics.json",
    "feature_schema.json",
    "sha256sums.txt",
]

HASHED_FILES = [
    "model.pt",
    "model_genes.txt",
    "label_map.json",
    "model_meta.json",
    "metrics.json",
    "feature_schema.json",
]


@dataclass(frozen=True)
class PackagePaths:
    version_dir: Path

    @property
    def meta_path(self) -> Path:
        return self.version_dir / "model_meta.json"

    @property
    def manifest_path(self) -> Path:
        return self.version_dir / "sha256sums.txt"


def task_root(registry_root: Path, task: str) -> Path:
    return registry_root / "tasks" / safe_component(task, "task")


def model_root(registry_root: Path, task: str, model_name: str) -> Path:
    return task_root(registry_root, task) / "models" / safe_component(model_name, "model_name")


def versions_root(registry_root: Path, task: str, model_name: str) -> Path:
    return model_root(registry_root, task, model_name) / "versions"


def version_dir(registry_root: Path, task: str, model_name: str, version: str) -> Path:
    return versions_root(registry_root, task, model_name) / safe_component(version, "version")


def aliases_root(registry_root: Path, task: str, model_name: str) -> Path:
    return model_root(registry_root, task, model_name) / "aliases"


def alias_path(registry_root: Path, task: str, model_name: str, alias: str) -> Path:
    return aliases_root(registry_root, task, model_name) / f"{safe_component(alias, 'alias')}.json"


def audit_root(registry_root: Path, task: str, model_name: str) -> Path:
    return model_root(registry_root, task, model_name) / "audit"


def ownership_path(registry_root: Path, task: str, model_name: str) -> Path:
    """Phase 2A: the single, model-level (not per-version) ownership
    record. Sits as a flat file directly under the model root, a peer of
    versions/ and aliases/ -- there is exactly one per model, unlike
    aliases (many) or versions (many), so a subdirectory would be
    unwarranted structure."""
    return model_root(registry_root, task, model_name) / "ownership.json"


def promotions_log_path(registry_root: Path, task: str, model_name: str) -> Path:
    return audit_root(registry_root, task, model_name) / "promotions.jsonl"


# ── Run entity paths ──────────────────────────────────────────────────────────

def runs_root(registry_root: Path, task: str, model_name: str) -> Path:
    return model_root(registry_root, task, model_name) / "runs"


def run_dir(registry_root: Path, task: str, model_name: str, run_id: str) -> Path:
    return runs_root(registry_root, task, model_name) / safe_component(run_id, "run_id")


def run_params_path(registry_root: Path, task: str, model_name: str, run_id: str) -> Path:
    return run_dir(registry_root, task, model_name, run_id) / "params.json"


def run_tags_path(registry_root: Path, task: str, model_name: str, run_id: str) -> Path:
    return run_dir(registry_root, task, model_name, run_id) / "tags.json"


def run_metric_log_path(
    registry_root: Path, task: str, model_name: str, run_id: str, metric_key: str
) -> Path:
    return (
        run_dir(registry_root, task, model_name, run_id)
        / "metrics" / f"{safe_component(metric_key, 'metric_key')}.jsonl"
    )


# ── Version-level extras ──────────────────────────────────────────────────────

def version_tags_path(registry_root: Path, task: str, model_name: str, version: str) -> Path:
    return version_dir(registry_root, task, model_name, version) / "tags.json"
