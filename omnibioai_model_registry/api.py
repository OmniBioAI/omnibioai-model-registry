# File: omnibioai_model_registry/api.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .audit.audit_log import PromotionEvent, append_promotion_event, now_utc_iso
from .config import RegistryConfig, load_config
from .errors import ModelNotFound, ValidationError, VersionAlreadyExists
from .ownership import check_model_ownership, ensure_model_ownership
from .package import layout as L
from .package.manifest import verify_sha256_manifest, write_sha256_manifest
from .package.validate import validate_package_files
from .path_safety import assert_contained
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

    def _enforce_ownership_or_not_found(
        self,
        task: str,
        model_name: str,
        *,
        enforce_ownership: bool,
        requesting_org_id: Optional[str],
    ) -> None:
        """Phase 2B: the single call site every storage-layer method
        below uses before touching a model that must already exist.
        Raises ModelNotFound (the exact same exception/shape already
        raised for a genuinely nonexistent model) when the caller's
        organization does not own it -- anti-enumerating by construction,
        not by a second, separately-maintained response path. A no-op
        when enforce_ownership=False (the default -- CLI/direct Python
        API callers with no IAM identity at all are unaffected, matching
        Phase 2A's existing --org-id/CLI trust-boundary precedent)."""
        if not enforce_ownership:
            return
        check = check_model_ownership(
            self.root, task, model_name, requesting_org_id=requesting_org_id
        )
        if not check.allowed:
            raise ModelNotFound(f"Model not found: task={task}, model_name={model_name}")

    def _validate_artifacts_dir(self, artifacts_dir: Path) -> None:
        """Security PR (filesystem path hardening): `artifacts_dir` is a
        server-local directory path, not an identifier subject to
        safe_component()'s allowlist -- it legitimately needs to be
        anywhere a training job wrote its output on this server's
        filesystem, so a fixed character allowlist doesn't apply here
        the way it does to task/model_name/version/alias.

        Two checks, deliberately different in nature:

        1. ALWAYS enforced, no configuration involved: `artifacts_dir`
           may never resolve inside this registry's own root. Without
           this, any authenticated model.use holder could point
           artifacts_dir at an *already-registered* version directory
           -- including one belonging to a different organization -- and
           have copy_tree() ingest those files into a brand-new
           registration they own, then read them back via their own,
           legitimately-authorized /v1/artifacts or /v1/resolve. That
           would be a complete, silent bypass of every Phase 2A-2D
           ownership check via a path no ownership check ever
           inspects -- never a legitimate use case under any deployment,
           so this is a structural safety rule, not a policy choice.

        2. OPTIONAL, operator-configured: if
           OMNIBIOAI_MODEL_REGISTRY_ARTIFACTS_ALLOWED_ROOTS is set,
           artifacts_dir must resolve under one of the listed roots.
           Unset (the default) is intentionally unrestricted --
           preserves every existing deployment's behavior unchanged.
           This service has no established convention for where
           training-output directories live on a given server, so a
           mandatory default allowlist would be inventing a policy this
           codebase has no basis to pick; operators who want this
           narrowed can opt in.
        """
        try:
            artifacts_dir.relative_to(self.root)
        except ValueError:
            pass
        else:
            raise ValidationError(
                "artifacts_dir must not be inside the model registry root"
            )

        allowed_roots = self.cfg.artifacts_allowed_roots
        if allowed_roots:
            for allowed in allowed_roots:
                try:
                    artifacts_dir.relative_to(Path(allowed).expanduser().resolve())
                    return
                except ValueError:
                    continue
            raise ValidationError(
                "artifacts_dir is not under an allowed root "
                "(OMNIBIOAI_MODEL_REGISTRY_ARTIFACTS_ALLOWED_ROOTS)"
            )

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
        organization_id: Optional[str] = None,
        *,
        enforce_ownership: bool = False,
    ) -> Dict[str, Any]:
        """
        organization_id: Phase 2A ownership. Must be server-derived from
        an already-verified IAM identity (UserContext.org_id) by the
        caller -- this method never reads it from `metadata` or anywhere
        else, so a client stuffing an "organization_id" key into the free
        -form `metadata` dict has no effect on ownership. Defaults to
        None for non-HTTP callers with no IAM identity at all (CLI,
        direct Python API usage), which is recorded as status="unowned",
        not guessed. See ownership.py for the full design rationale.

        enforce_ownership (Phase 2B): when True, registering a version
        under an EXISTING model requires organization_id to match that
        model's recorded owner -- denied (ModelNotFound, no filesystem
        mutation at all) otherwise. Registering a genuinely NEW model is
        never blocked by this (there is nothing to match against yet);
        organization_id is simply recorded as its owner, same as Phase
        2A. Defaults to False so CLI/direct Python API callers (no IAM
        identity) are unaffected, matching the enforce_ownership default
        already used by resolve_model/promote_model/verify_model_ref.
        """
        artifacts_dir = Path(artifacts_dir).resolve()
        if not artifacts_dir.exists():
            raise ValidationError(f"artifacts_dir does not exist: {artifacts_dir}")
        self._validate_artifacts_dir(artifacts_dir)

        # Snapshot BEFORE _ensure_model_dirs() creates the model's
        # directories -- this is the only reliable signal for whether
        # this is a genuinely new model (ownership derived from the
        # caller) or a pre-existing one (ownership never silently
        # reassigned to the current caller; see ensure_model_ownership).
        #
        # Deliberately NOT "does the model root directory exist": both
        # resolve_model() and promote_model() also call
        # _ensure_model_dirs() as a side effect (e.g. a GET /v1/resolve
        # for a not-yet-registered model creates the empty
        # versions/aliases/audit skeleton before raising ModelNotFound).
        # An empty skeleton from a read/probe must not be mistaken for a
        # real prior registration -- the only trustworthy signal is
        # whether at least one version directory actually exists.
        versions_root_path = L.versions_root(self.root, task, model_name)
        model_pre_existing = versions_root_path.exists() and any(
            versions_root_path.iterdir()
        )

        # Phase 2B: adding a version to an EXISTING model requires
        # belonging to its owning org -- checked before any directory
        # creation, artifact copy, or ownership read/write below, so a
        # denial leaves the filesystem completely untouched (matches
        # register_model's own promise: ownership is only ever
        # established/confirmed after a successful copy+validate). A
        # brand-new model (model_pre_existing=False) is never blocked
        # here -- there is no prior owner to conflict with.
        self._enforce_ownership_or_not_found(
            task, model_name,
            enforce_ownership=enforce_ownership and model_pre_existing,
            requesting_org_id=organization_id,
        )

        self._ensure_model_dirs(task, model_name)

        dst = L.version_dir(self.root, task, model_name, version)
        # Defense-in-depth: dst is already lexically incapable of
        # escaping self.root (task/model_name/version were validated by
        # safe_component() inside version_dir() above), but this
        # re-verifies against the real, symlink-resolved filesystem
        # location before any write -- see path_safety.py.
        assert_contained(dst, self.root, field_name="version")
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

        # sha256sums.txt is itself a REQUIRED_FILE, so it must be written
        # before validate_package_files() runs, or validation will always
        # fail claiming the manifest is missing.
        hashes = write_sha256_manifest(
            dst, dst / "sha256sums.txt", include_files=L.REQUIRED_FILES
        )

        validate_package_files(dst)

        # Phase 2A: establish (never overwrite) the model-level ownership
        # record. Must happen only after the copy/validate above succeed,
        # so a failed registration (bad artifacts, duplicate version) has
        # no ownership side effect.
        ownership = ensure_model_ownership(
            self.backend,
            self.root,
            task,
            model_name,
            organization_id=organization_id,
            actor=actor,
            model_pre_existing=model_pre_existing,
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
            "organization_id": ownership.organization_id,
            "ownership_status": ownership.status,
        }

    def resolve_model(
        self,
        task: str,
        model_ref: str,
        verify: bool = True,
        *,
        enforce_ownership: bool = False,
        requesting_org_id: Optional[str] = None,
    ) -> Path:
        ref = parse_model_ref(model_ref)
        self._ensure_model_dirs(task, ref.model_name)

        # Phase 2B: checked before any alias/version resolution, so a
        # denial reveals nothing about the specific alias or version
        # requested either -- just "not found", identical to a genuinely
        # nonexistent model at every subsequent step.
        self._enforce_ownership_or_not_found(
            task, ref.model_name,
            enforce_ownership=enforce_ownership, requesting_org_id=requesting_org_id,
        )

        alias_file = L.alias_path(self.root, task, ref.model_name, ref.selector)
        if alias_file.exists():
            data = json.loads(alias_file.read_text())
            version = data["version"]
        else:
            version = ref.selector

        vdir = L.version_dir(self.root, task, ref.model_name, version)
        # Defense-in-depth (see register_model's identical comment) --
        # `version` here may have come from an alias file's own content
        # rather than directly from the caller, but version_dir() already
        # validated it as a path component either way.
        assert_contained(vdir, self.root, field_name="version")
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
        *,
        enforce_ownership: bool = False,
        requesting_org_id: Optional[str] = None,
    ) -> None:

        self._ensure_model_dirs(task, model_name)

        # Phase 2B: promote always operates on an existing model (there
        # is no "create" case here, unlike register_model) -- checked
        # before the version-existence check below, so a denial doesn't
        # even confirm whether the requested version exists.
        self._enforce_ownership_or_not_found(
            task, model_name,
            enforce_ownership=enforce_ownership, requesting_org_id=requesting_org_id,
        )

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

        alias_file = L.alias_path(self.root, task, model_name, alias)
        # Defense-in-depth (see register_model's identical comment).
        assert_contained(alias_file, self.root, field_name="alias")
        self.backend.atomic_write_text(
            alias_file,
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

    def verify_model_ref(
        self,
        task: str,
        model_ref: str,
        *,
        enforce_ownership: bool = False,
        requesting_org_id: Optional[str] = None,
    ) -> None:
        _ = self.resolve_model(
            task, model_ref, verify=True,
            enforce_ownership=enforce_ownership, requesting_org_id=requesting_org_id,
        )


# -------------------------
# Convenience functions
# -------------------------

def _default_registry() -> ModelRegistry:
    return ModelRegistry.from_env()


def register_model(*args, **kwargs) -> Dict[str, Any]:
    return _default_registry().register_model(*args, **kwargs)


def resolve_model(task: str, model_ref: str, verify: bool = True, **kwargs) -> str:
    return str(
        _default_registry().resolve_model(
            task=task, model_ref=model_ref, verify=verify, **kwargs
        )
    )


def promote_model(*args, **kwargs) -> None:
    return _default_registry().promote_model(*args, **kwargs)


def verify_model_ref(task: str, model_ref: str, **kwargs) -> None:
    return _default_registry().verify_model_ref(task=task, model_ref=model_ref, **kwargs)