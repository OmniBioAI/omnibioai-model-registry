# File: omnibioai_model_registry/api.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .audit.audit_log import PromotionEvent, append_promotion_event, now_utc_iso
from .config import RegistryConfig, load_config
from .errors import ModelNotFound, ValidationError, VersionAlreadyExists
from .package import layout as L
from .package.manifest import verify_sha256_manifest, write_sha256_manifest
from .package.validate import validate_package_files
from .refs import parse_model_ref
from .storage.localfs import LocalFS


@dataclass
class ModelRegistry:
    cfg: RegistryConfig
    backend: LocalFS  # v1 local; later swap to interface

    @classmethod
    def from_env(cls) -> "ModelRegistry":
        cfg = load_config()
        if cfg.backend != "localfs":
            raise ValueError(
                f"Unsupported backend '{cfg.backend}' (v1 supports only localfs)."
            )
        return cls(cfg=cfg, backend=LocalFS())

    @property
    def root(self) -> Path:
        return Path(self.cfg.root).expanduser().resolve()

    def _ensure_model_dirs(self, task: str, model_name: str) -> None:
        self.backend.ensure_dirs(L.versions_root(self.root, task, model_name))
        self.backend.ensure_dirs(L.aliases_root(self.root, task, model_name))
        self.backend.ensure_dirs(L.audit_root(self.root, task, model_name))

    def register_model(
        self,
        task: str,
        model_name: str,
        version: str,
        artifacts_dir: str | Path,
        metadata: Dict[str, Any],
        set_alias: Optional[str] = "latest",
        actor: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:

        artifacts_dir = Path(artifacts_dir).resolve()
        if not artifacts_dir.exists():
            raise ValidationError(f"artifacts_dir does not exist: {artifacts_dir}")

        self._ensure_model_dirs(task, model_name)

        dst = L.version_dir(self.root, task, model_name, version)
        if self.backend.exists(dst):
            raise VersionAlreadyExists(
                f"Model version already exists: {task}/{model_name}/{version}"
            )

        self.backend.copy_tree(artifacts_dir, dst)

        # ------------------------------------------------------
        # ✅ MINIMAL CHANGE: add MLflow-style lineage metadata
        # ------------------------------------------------------
        meta = dict(metadata)
        meta.setdefault("task", task)
        meta.setdefault("model_name", model_name)
        meta.setdefault("version", version)
        meta.setdefault("created_at", now_utc_iso())

        meta.setdefault(
            "lineage",
            {
                "dataset": metadata.get("dataset"),
                "run_id": metadata.get("run_id"),
                "metrics": metadata.get("metrics", {}),
                "parent_version": metadata.get("parent_version"),
            },
        )
        # ------------------------------------------------------

        self.backend.atomic_write_text(
            dst / "model_meta.json", json.dumps(meta, indent=2) + "\n"
        )

        validate_package_files(dst)

        hashes = write_sha256_manifest(
            dst, dst / "sha256sums.txt", include_files=L.REQUIRED_FILES
        )

        if set_alias:
            self.promote_model(
                task,
                model_name,
                set_alias,
                version,
                actor=actor,
                reason=reason or "register_model",
            )

        return {
            "ok": True,
            "task": task,
            "model_name": model_name,
            "version": version,
            "package_path": str(dst),
            "hashes": hashes,
            "alias_set": set_alias,
        }

    def resolve_model(self, task: str, model_ref: str, verify: bool = True) -> Path:
        ref = parse_model_ref(model_ref)
        self._ensure_model_dirs(task, ref.model_name)

        alias_file = L.alias_path(self.root, task, ref.model_name, ref.selector)
        if alias_file.exists():
            data = json.loads(alias_file.read_text())
            version = data["version"]
        else:
            version = ref.selector

        vdir = L.version_dir(self.root, task, ref.model_name, version)
        if not vdir.exists():
            raise ModelNotFound(
                f"Model not found: task={task}, ref={model_ref} (resolved version={version})"
            )

        if verify or self.cfg.strict_verify:
            validate_package_files(vdir)
            verify_sha256_manifest(vdir, vdir / "sha256sums.txt")

        return vdir

    def promote_model(
        self,
        task: str,
        model_name: str,
        alias: str,
        version: str,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:

        self._ensure_model_dirs(task, model_name)

        vdir = L.version_dir(self.root, task, model_name, version)
        if not vdir.exists():
            raise ModelNotFound(
                f"Cannot promote missing version: {task}/{model_name}/{version}"
            )

        payload = {
            "task": task,
            "model_name": model_name,
            "alias": alias,
            "version": version,
            "updated_at": now_utc_iso(),
            "actor": actor,
            "reason": reason,
        }

        self.backend.atomic_write_text(
            L.alias_path(self.root, task, model_name, alias),
            json.dumps(payload, indent=2) + "\n",
        )

        ev = PromotionEvent(
            task=task,
            model_name=model_name,
            alias=alias,
            version=version,
            actor=actor,
            reason=reason,
            ts_utc=payload["updated_at"],
        )

        append_promotion_event(
            L.promotions_log_path(self.root, task, model_name), ev
        )

    def verify_model_ref(self, task: str, model_ref: str) -> None:
        _ = self.resolve_model(task, model_ref, verify=True)


# -------------------------
# Convenience functions
# -------------------------

def _default_registry() -> ModelRegistry:
    return ModelRegistry.from_env()


def register_model(*args, **kwargs) -> Dict[str, Any]:
    return _default_registry().register_model(*args, **kwargs)


def resolve_model(task: str, model_ref: str, verify: bool = True) -> str:
    return str(
        _default_registry().resolve_model(task=task, model_ref=model_ref, verify=verify)
    )


def promote_model(*args, **kwargs) -> None:
    return _default_registry().promote_model(*args, **kwargs)


def verify_model_ref(task: str, model_ref: str) -> None:
    return _default_registry().verify_model_ref(task=task, model_ref=model_ref)