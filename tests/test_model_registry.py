"""
tests/test_model_registry.py

Comprehensive tests for omnibioai_model_registry — target 95%+ coverage.
Covers: api, config, refs, errors, package/*, storage/*, audit/*, cli/main
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from omnibioai_model_registry.api import ModelRegistry
from omnibioai_model_registry.errors import (
    IntegrityError,
    InvalidModelRef,
    ModelNotFound,
    RegistryNotConfigured,
    ValidationError,
    VersionAlreadyExists,
)
from omnibioai_model_registry.package.layout import REQUIRED_FILES

# ============================================================
# Shared helpers
# ============================================================


def _make_minimal_package(dir_path: Path, *, meta: dict | None = None) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "model.pt").write_bytes(b"fake model weights")
    (dir_path / "model_genes.txt").write_text("GeneA\nGeneB\n", encoding="utf-8")
    (dir_path / "label_map.json").write_text(
        json.dumps({"0": "A", "1": "B"}, indent=2) + "\n", encoding="utf-8"
    )
    (dir_path / "metrics.json").write_text(
        json.dumps({"acc": 0.9}, indent=2) + "\n", encoding="utf-8"
    )
    (dir_path / "feature_schema.json").write_text(
        json.dumps({"features": ["GeneA", "GeneB"]}, indent=2) + "\n", encoding="utf-8"
    )
    (dir_path / "model_meta.json").write_text(
        json.dumps(meta or {}, indent=2) + "\n", encoding="utf-8"
    )
    (dir_path / "sha256sums.txt").write_text("", encoding="utf-8")


@pytest.fixture
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "registry_root"
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_BACKEND", "localfs")
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "1")
    return root


@pytest.fixture
def reg(env_root: Path) -> ModelRegistry:
    return ModelRegistry.from_env()


@pytest.fixture
def registered_reg(env_root: Path, tmp_path: Path) -> tuple[ModelRegistry, Path]:
    src = tmp_path / "pkg_src"
    _make_minimal_package(src)
    r = ModelRegistry.from_env()
    r.register_model(
        task="t",
        model_name="m",
        version="v1",
        artifacts_dir=src,
        metadata={},
        set_alias="latest",
        actor="manish",
        reason="test",
    )
    return r, env_root


# ============================================================
# Original regression tests
# ============================================================


def test_register_creates_version_dir_and_metadata(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src, meta={"framework": "test"})
    reg = ModelRegistry.from_env()
    out = reg.register_model(
        task="celltype_sc",
        model_name="human_pbmc",
        version="2026-02-14_001",
        artifacts_dir=src,
        metadata={"framework": "sklearn", "model_type": "lr"},
        set_alias=None,
    )
    assert out["ok"] is True
    vdir = Path(out["package_path"])
    assert vdir.exists()
    for f in REQUIRED_FILES:
        assert (vdir / f).exists()
    meta = json.loads((vdir / "model_meta.json").read_text())
    assert meta["task"] == "celltype_sc"
    assert meta["model_name"] == "human_pbmc"
    assert meta["version"] == "2026-02-14_001"
    assert "created_at" in meta
    assert meta["framework"] == "sklearn"


def test_register_is_immutable(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src)
    reg = ModelRegistry.from_env()
    reg.register_model(
        task="t",
        model_name="m",
        version="v1",
        artifacts_dir=src,
        metadata={},
        set_alias=None,
    )
    with pytest.raises(VersionAlreadyExists):
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )


def test_resolve_by_version_and_alias(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src)
    reg = ModelRegistry.from_env()
    reg.register_model(
        task="t",
        model_name="m",
        version="v1",
        artifacts_dir=src,
        metadata={},
        set_alias="latest",
        actor="manish",
        reason="unit test",
    )
    vdir1 = reg.resolve_model(task="t", model_ref="m@v1", verify=True)
    vdir2 = reg.resolve_model(task="t", model_ref="m@latest", verify=True)
    assert str(vdir2) == str(vdir1)
    alias_file = env_root / "tasks" / "t" / "models" / "m" / "aliases" / "latest.json"
    assert alias_file.exists()
    alias = json.loads(alias_file.read_text())
    assert alias["version"] == "v1"
    assert alias["actor"] == "manish"


def test_promote_writes_audit_log(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src)
    reg = ModelRegistry.from_env()
    reg.register_model(
        task="t",
        model_name="m",
        version="v1",
        artifacts_dir=src,
        metadata={},
        set_alias=None,
    )
    reg.promote_model(
        task="t",
        model_name="m",
        alias="production",
        version="v1",
        actor="x",
        reason="y",
    )
    log_path = env_root / "tasks" / "t" / "models" / "m" / "audit" / "promotions.jsonl"
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 1
    ev = json.loads(lines[-1])
    assert ev["alias"] == "production"
    assert ev["version"] == "v1"
    assert ev["actor"] == "x"


def test_resolve_missing_raises(env_root: Path):
    reg = ModelRegistry.from_env()
    with pytest.raises(ModelNotFound):
        reg.resolve_model(task="t", model_ref="m@latest", verify=False)


def test_register_fails_if_artifacts_dir_missing(env_root: Path, tmp_path: Path):
    reg = ModelRegistry.from_env()
    with pytest.raises(ValidationError):
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=tmp_path / "does_not_exist",
            metadata={},
        )


# ============================================================
# config.py
# ============================================================


class TestConfig:

    def test_load_config_from_env(self, monkeypatch):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_BACKEND", "localfs")
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "1")
        from omnibioai_model_registry.config import load_config

        cfg = load_config()
        assert cfg.root == "/tmp/reg"
        assert cfg.backend == "localfs"
        assert cfg.strict_verify is True

    def test_load_config_strict_verify_false(self, monkeypatch):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        from omnibioai_model_registry.config import load_config

        cfg = load_config()
        assert cfg.strict_verify is False

    def test_load_config_strict_verify_false_word(self, monkeypatch):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "false")
        from omnibioai_model_registry.config import load_config

        cfg = load_config()
        assert cfg.strict_verify is False

    def test_load_config_missing_root_raises(self, monkeypatch):
        monkeypatch.delenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", raising=False)
        monkeypatch.delenv("REGISTRY_ROOT", raising=False)
        from omnibioai_model_registry.config import load_config

        with pytest.raises(RegistryNotConfigured):
            load_config()

    def test_load_config_fallback_registry_root(self, monkeypatch):
        """Covers line 17: REGISTRY_ROOT fallback."""
        monkeypatch.delenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", raising=False)
        monkeypatch.setenv("REGISTRY_ROOT", "/tmp/fallback")
        from omnibioai_model_registry.config import load_config

        cfg = load_config()
        assert cfg.root == "/tmp/fallback"

    def test_load_config_default_backend(self, monkeypatch):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.delenv("OMNIBIOAI_MODEL_REGISTRY_BACKEND", raising=False)
        from omnibioai_model_registry.config import load_config

        cfg = load_config()
        assert cfg.backend == "localfs"


# ============================================================
# refs.py
# ============================================================


class TestRefs:

    def test_parse_valid_ref(self):
        from omnibioai_model_registry.refs import parse_model_ref

        ref = parse_model_ref("human_pbmc@production")
        assert ref.model_name == "human_pbmc"
        assert ref.selector == "production"

    def test_parse_version_ref(self):
        from omnibioai_model_registry.refs import parse_model_ref

        ref = parse_model_ref("human_pbmc@2026-02-13_001")
        assert ref.model_name == "human_pbmc"
        assert ref.selector == "2026-02-13_001"

    def test_parse_missing_at_raises(self):
        """Covers line 19: no '@' in model_ref."""
        from omnibioai_model_registry.refs import parse_model_ref

        with pytest.raises(InvalidModelRef):
            parse_model_ref("human_pbmc_no_at")

    def test_parse_empty_ref_raises(self):
        from omnibioai_model_registry.refs import parse_model_ref

        with pytest.raises(InvalidModelRef):
            parse_model_ref("")

    def test_parse_empty_model_name_raises(self):
        """Covers line 24: empty model_name after split."""
        from omnibioai_model_registry.refs import parse_model_ref

        with pytest.raises(InvalidModelRef):
            parse_model_ref("@production")

    def test_parse_empty_selector_raises(self):
        """Covers line 24: empty selector after split."""
        from omnibioai_model_registry.refs import parse_model_ref

        with pytest.raises(InvalidModelRef):
            parse_model_ref("human_pbmc@")


# ============================================================
# errors.py
# ============================================================


class TestErrors:

    def test_all_error_types(self):
        from omnibioai_model_registry.errors import (
            IntegrityError,
            InvalidModelRef,
            ModelNotFound,
            RegistryNotConfigured,
            ValidationError,
            VersionAlreadyExists,
        )

        for cls in [
            ModelNotFound,
            VersionAlreadyExists,
            ValidationError,
            IntegrityError,
            InvalidModelRef,
            RegistryNotConfigured,
        ]:
            e = cls("test message")
            assert "test message" in str(e)
            assert isinstance(e, Exception)


# ============================================================
# package/layout.py
# ============================================================


class TestLayout:

    def test_required_files_list(self):
        from omnibioai_model_registry.package.layout import REQUIRED_FILES

        assert "model.pt" in REQUIRED_FILES
        assert "sha256sums.txt" in REQUIRED_FILES

    def test_task_root(self, tmp_path):
        from omnibioai_model_registry.package.layout import task_root

        p = task_root(tmp_path, "celltype_sc")
        assert str(p).endswith("tasks/celltype_sc")

    def test_model_root(self, tmp_path):
        from omnibioai_model_registry.package.layout import model_root

        p = model_root(tmp_path, "celltype_sc", "human_pbmc")
        assert "models/human_pbmc" in str(p)

    def test_version_dir(self, tmp_path):
        from omnibioai_model_registry.package.layout import version_dir

        p = version_dir(tmp_path, "t", "m", "v1")
        assert p.name == "v1"

    def test_alias_path(self, tmp_path):
        from omnibioai_model_registry.package.layout import alias_path

        p = alias_path(tmp_path, "t", "m", "latest")
        assert p.name == "latest.json"

    def test_audit_root(self, tmp_path):
        from omnibioai_model_registry.package.layout import audit_root

        p = audit_root(tmp_path, "t", "m")
        assert p.name == "audit"

    def test_promotions_log_path(self, tmp_path):
        from omnibioai_model_registry.package.layout import promotions_log_path

        p = promotions_log_path(tmp_path, "t", "m")
        assert p.name == "promotions.jsonl"

    def test_package_paths_meta(self, tmp_path):
        """Covers lines 31, 35: PackagePaths properties."""
        from omnibioai_model_registry.package.layout import PackagePaths

        pp = PackagePaths(version_dir=tmp_path)
        assert pp.meta_path == tmp_path / "model_meta.json"
        assert pp.manifest_path == tmp_path / "sha256sums.txt"

    def test_versions_root(self, tmp_path):
        from omnibioai_model_registry.package.layout import versions_root

        p = versions_root(tmp_path, "t", "m")
        assert p.name == "versions"

    def test_aliases_root(self, tmp_path):
        from omnibioai_model_registry.package.layout import aliases_root

        p = aliases_root(tmp_path, "t", "m")
        assert p.name == "aliases"


# ============================================================
# package/validate.py
# ============================================================


class TestValidate:

    def test_validate_passes_with_all_files(self, tmp_path):
        from omnibioai_model_registry.package.validate import validate_package_files

        for f in REQUIRED_FILES:
            (tmp_path / f).write_text("x")
        validate_package_files(tmp_path)

    def test_validate_fails_with_missing_file(self, tmp_path):
        """Covers line 11: raises ValidationError."""
        from omnibioai_model_registry.package.validate import validate_package_files

        for f in REQUIRED_FILES[:-1]:
            (tmp_path / f).write_text("x")
        with pytest.raises(ValidationError) as exc_info:
            validate_package_files(tmp_path)
        assert "missing required files" in str(exc_info.value)


# ============================================================
# package/manifest.py
# ============================================================


class TestManifest:

    def test_sha256_file(self, tmp_path):
        from omnibioai_model_registry.package.manifest import sha256_file

        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        digest = sha256_file(f)
        assert len(digest) == 64
        assert digest == sha256_file(f)

    def test_write_and_read_manifest(self, tmp_path):
        from omnibioai_model_registry.package.manifest import (
            read_sha256_manifest,
            write_sha256_manifest,
        )

        (tmp_path / "model.pt").write_bytes(b"weights")
        (tmp_path / "model_meta.json").write_text("{}")
        manifest_path = tmp_path / "sha256sums.txt"
        hashes = write_sha256_manifest(
            tmp_path, manifest_path, include_files=["model.pt", "model_meta.json"]
        )
        assert "model.pt" in hashes
        assert "model_meta.json" in hashes
        read_back = read_sha256_manifest(manifest_path)
        assert read_back["model.pt"] == hashes["model.pt"]

    def test_write_manifest_skips_manifest_itself(self, tmp_path):
        """Covers line 38: skips sha256sums.txt from being hashed."""
        from omnibioai_model_registry.package.manifest import write_sha256_manifest

        (tmp_path / "model.pt").write_bytes(b"weights")
        manifest_path = tmp_path / "sha256sums.txt"
        hashes = write_sha256_manifest(
            tmp_path, manifest_path, include_files=["model.pt", "sha256sums.txt"]
        )
        assert "sha256sums.txt" not in hashes
        assert "model.pt" in hashes

    def test_write_manifest_skips_missing_files(self, tmp_path):
        """Covers line 50: skips files that don't exist."""
        from omnibioai_model_registry.package.manifest import write_sha256_manifest

        (tmp_path / "model.pt").write_bytes(b"weights")
        manifest_path = tmp_path / "sha256sums.txt"
        hashes = write_sha256_manifest(
            tmp_path, manifest_path, include_files=["model.pt", "nonexistent.bin"]
        )
        assert "nonexistent.bin" not in hashes
        assert "model.pt" in hashes

    def test_write_manifest_empty_files(self, tmp_path):
        """Covers line 59: empty lines list → empty manifest."""
        from omnibioai_model_registry.package.manifest import write_sha256_manifest

        manifest_path = tmp_path / "sha256sums.txt"
        hashes = write_sha256_manifest(tmp_path, manifest_path, include_files=[])
        assert hashes == {}
        assert manifest_path.read_text() == ""

    def test_read_manifest_missing_raises(self, tmp_path):
        """Covers line 55: missing manifest raises IntegrityError."""
        from omnibioai_model_registry.package.manifest import read_sha256_manifest

        with pytest.raises(IntegrityError):
            read_sha256_manifest(tmp_path / "nonexistent.txt")

    def test_read_manifest_skips_short_lines(self, tmp_path):
        """Covers line 71: skips lines with < 2 parts."""
        from omnibioai_model_registry.package.manifest import read_sha256_manifest

        manifest_path = tmp_path / "sha256sums.txt"
        manifest_path.write_text("justoneword\nabc123  file.txt\n")
        result = read_sha256_manifest(manifest_path)
        assert "file.txt" in result
        assert len(result) == 1

    def test_read_manifest_skips_empty_lines(self, tmp_path):
        """Covers line 55: empty lines skipped in read_sha256_manifest."""
        from omnibioai_model_registry.package.manifest import read_sha256_manifest

        manifest_path = tmp_path / "sha256sums.txt"
        manifest_path.write_text(
            "\n"
            "abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1  model.pt\n"
            "\n"
            "def456def456def456def456def456def456def456def456def456def456def4  model_meta.json\n"
            "\n"
        )
        result = read_sha256_manifest(manifest_path)
        assert "model.pt" in result
        assert "model_meta.json" in result
        assert len(result) == 2

    def test_verify_manifest_passes(self, tmp_path):
        from omnibioai_model_registry.package.manifest import (
            verify_sha256_manifest,
            write_sha256_manifest,
        )

        (tmp_path / "model.pt").write_bytes(b"weights")
        manifest_path = tmp_path / "sha256sums.txt"
        write_sha256_manifest(tmp_path, manifest_path, include_files=["model.pt"])
        verify_sha256_manifest(tmp_path, manifest_path)

    def test_verify_manifest_missing_file_raises(self, tmp_path):
        """Covers line 74: file expected by manifest is missing."""
        from omnibioai_model_registry.package.manifest import verify_sha256_manifest

        manifest_path = tmp_path / "sha256sums.txt"
        manifest_path.write_text("abc123  model.pt\n")
        with pytest.raises(IntegrityError) as exc_info:
            verify_sha256_manifest(tmp_path, manifest_path)
        assert "missing" in str(exc_info.value)

    def test_verify_manifest_hash_mismatch_raises(self, tmp_path):
        from omnibioai_model_registry.package.manifest import verify_sha256_manifest

        (tmp_path / "model.pt").write_bytes(b"different content")
        manifest_path = tmp_path / "sha256sums.txt"
        manifest_path.write_text(
            "deadbeef00000000000000000000000000000000000000000000000000000000  model.pt\n"
        )
        with pytest.raises(IntegrityError) as exc_info:
            verify_sha256_manifest(tmp_path, manifest_path)
        assert "mismatch" in str(exc_info.value)


# ============================================================
# storage/localfs.py
# ============================================================


class TestLocalFS:

    def test_ensure_dirs(self, tmp_path):
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        new_dir = tmp_path / "a" / "b" / "c"
        fs.ensure_dirs(new_dir)
        assert new_dir.exists()

    def test_exists_true_false(self, tmp_path):
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        assert fs.exists(tmp_path) is True
        assert fs.exists(tmp_path / "nonexistent") is False

    def test_copy_tree(self, tmp_path):
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.txt").write_text("hello")
        dst = tmp_path / "dst"
        fs.copy_tree(src, dst)
        assert (dst / "file.txt").read_text() == "hello"

    def test_atomic_write_text(self, tmp_path):
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        target = tmp_path / "output.txt"
        fs.atomic_write_text(target, "hello world")
        assert target.read_text() == "hello world"

    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        """Covers lines 31-33: parent.mkdir inside atomic_write_text."""
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        target = tmp_path / "deep" / "nested" / "file.txt"
        fs.atomic_write_text(target, "content")
        assert target.read_text() == "content"

    def test_atomic_write_overwrites(self, tmp_path):
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        target = tmp_path / "file.txt"
        fs.atomic_write_text(target, "first")
        fs.atomic_write_text(target, "second")
        assert target.read_text() == "second"

    def test_atomic_write_cleanup_when_replace_fails(self, tmp_path, monkeypatch):
        """Covers localfs.py line 32: os.unlink(tmp) called when replace fails."""
        import omnibioai_model_registry.storage.localfs as lfs_mod
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        target = tmp_path / "file.txt"
        monkeypatch.setattr(
            lfs_mod.os,
            "replace",
            lambda src, dst: (_ for _ in ()).throw(OSError("fail")),
        )
        with pytest.raises(OSError):
            fs.atomic_write_text(target, "content")
        leftover = list(tmp_path.glob("file.txt.*"))
        assert leftover == []

    def test_atomic_write_cleanup_unlink_exception_suppressed(
        self, tmp_path, monkeypatch
    ):
        """Covers localfs.py line 33: pass — exception in os.unlink is suppressed."""
        import omnibioai_model_registry.storage.localfs as lfs_mod
        from omnibioai_model_registry.storage.localfs import LocalFS

        fs = LocalFS()
        target = tmp_path / "file.txt"
        monkeypatch.setattr(
            lfs_mod.os,
            "replace",
            lambda src, dst: (_ for _ in ()).throw(OSError("replace fail")),
        )
        monkeypatch.setattr(
            lfs_mod.os,
            "unlink",
            lambda p: (_ for _ in ()).throw(OSError("unlink fail")),
        )
        with pytest.raises(OSError, match="replace fail"):
            fs.atomic_write_text(target, "content")


# ============================================================
# audit/audit_log.py
# ============================================================


class TestAuditLog:

    def test_append_and_read_promotion_event(self, tmp_path):
        from omnibioai_model_registry.audit.audit_log import (
            PromotionEvent,
            append_promotion_event,
            now_utc_iso,
        )

        log_path = tmp_path / "promotions.jsonl"
        ev = PromotionEvent(
            task="t",
            model_name="m",
            alias="production",
            version="v1",
            actor="x",
            reason="y",
            ts_utc=now_utc_iso(),
        )
        append_promotion_event(log_path, ev)
        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["alias"] == "production"
        assert data["actor"] == "x"

    def test_append_multiple_events(self, tmp_path):
        from omnibioai_model_registry.audit.audit_log import (
            PromotionEvent,
            append_promotion_event,
            now_utc_iso,
        )

        log_path = tmp_path / "promotions.jsonl"
        for i in range(3):
            ev = PromotionEvent(
                task="t",
                model_name="m",
                alias=f"alias_{i}",
                version=f"v{i}",
                actor="x",
                reason="y",
                ts_utc=now_utc_iso(),
            )
            append_promotion_event(log_path, ev)
        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_now_utc_iso_format(self):
        from omnibioai_model_registry.audit.audit_log import now_utc_iso

        ts = now_utc_iso()
        assert "T" in ts
        assert ts.endswith("Z") or "+00:00" in ts or "UTC" in ts


# ============================================================
# api.py — additional coverage
# ============================================================


class TestAPIAdditional:

    def test_from_env_unsupported_backend_raises(self, monkeypatch, tmp_path):
        """Covers line 28: unsupported backend raises ValueError."""
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(tmp_path))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_BACKEND", "s3")
        with pytest.raises(ValueError, match="Unsupported backend"):
            ModelRegistry.from_env()

    def test_root_property(self, env_root):
        reg = ModelRegistry.from_env()
        assert reg.root.is_absolute()

    def test_promote_missing_version_raises(self, env_root):
        """Covers line 143: promote missing version raises ModelNotFound."""
        reg = ModelRegistry.from_env()
        with pytest.raises(ModelNotFound):
            reg.promote_model(
                task="t",
                model_name="m",
                alias="prod",
                version="nonexistent",
                actor="x",
                reason="y",
            )

    def test_verify_model_ref(self, env_root, tmp_path):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        reg.verify_model_ref(task="t", model_ref="m@v1")

    def test_resolve_model_no_verify(self, env_root, tmp_path):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        vdir = reg.resolve_model(task="t", model_ref="m@v1", verify=False)
        assert vdir.exists()

    def test_register_with_actor_and_reason(self, env_root, tmp_path):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        out = reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias="latest",
            actor="manish",
            reason="ci",
        )
        assert out["alias_set"] == "latest"

    def test_register_no_alias(self, env_root, tmp_path):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        out = reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        assert out["alias_set"] is None

    def test_module_level_register_model(self, env_root, tmp_path):
        """Covers module-level register_model function."""
        import omnibioai_model_registry.api as api_mod

        src = tmp_path / "src"
        _make_minimal_package(src)
        out = api_mod.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        assert out["ok"] is True

    def test_module_level_resolve_model(self, env_root, tmp_path):
        """Covers module-level resolve_model function."""
        import omnibioai_model_registry.api as api_mod

        src = tmp_path / "src"
        _make_minimal_package(src)
        api_mod.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        path_str = api_mod.resolve_model(task="t", model_ref="m@v1", verify=True)
        assert isinstance(path_str, str)
        assert Path(path_str).exists()

    def test_module_level_promote_model(self, env_root, tmp_path):
        """Covers module-level promote_model function."""
        import omnibioai_model_registry.api as api_mod

        src = tmp_path / "src"
        _make_minimal_package(src)
        api_mod.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        api_mod.promote_model(
            task="t",
            model_name="m",
            alias="staging",
            version="v1",
            actor="x",
            reason="y",
        )

    def test_module_level_verify_model_ref(self, env_root, tmp_path):
        """Covers module-level verify_model_ref function."""
        import omnibioai_model_registry.api as api_mod

        src = tmp_path / "src"
        _make_minimal_package(src)
        api_mod.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        api_mod.verify_model_ref(task="t", model_ref="m@v1")

    def test_hashes_returned_in_register(self, env_root, tmp_path):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        out = reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        assert isinstance(out["hashes"], dict)
        assert len(out["hashes"]) > 0

    def test_promote_multiple_aliases(self, env_root, tmp_path):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        reg.promote_model(task="t", model_name="m", alias="staging", version="v1")
        reg.promote_model(task="t", model_name="m", alias="production", version="v1")
        staging = json.loads(
            (env_root / "tasks/t/models/m/aliases/staging.json").read_text()
        )
        production = json.loads(
            (env_root / "tasks/t/models/m/aliases/production.json").read_text()
        )
        assert staging["version"] == "v1"
        assert production["version"] == "v1"


# ============================================================
# cli/main.py — covered via runpy
# ============================================================


class TestCLIMain:

    def _get_cli_path(self):
        import omnibioai_model_registry.cli.main as cli_mod

        return cli_mod.__file__

    def _run_cli(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", argv)
        runpy.run_path(self._get_cli_path(), run_name="__main__")

    def _register_via_api(self, env_root, tmp_path, model="m", version="v1"):
        src = tmp_path / f"src_{model}_{version}"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name=model,
            version=version,
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        return src

    def test_cli_register_plain(self, env_root, tmp_path, monkeypatch, capsys):
        src = tmp_path / "src"
        _make_minimal_package(src)
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "register",
                "--task",
                "t",
                "--model",
                "m",
                "--version",
                "v1",
                "--artifacts",
                str(src),
                "--set-alias",
                "latest",
            ],
        )
        out = capsys.readouterr()
        assert "Registered" in out.out or "v1" in out.out

    def test_cli_register_json_output(self, env_root, tmp_path, monkeypatch, capsys):
        src = tmp_path / "src"
        _make_minimal_package(src)
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "register",
                "--task",
                "t",
                "--model",
                "m",
                "--version",
                "v1",
                "--artifacts",
                str(src),
                "--set-alias",
                "",
                "--json",
            ],
        )
        out = capsys.readouterr()
        data = json.loads(out.out)
        assert data["ok"] is True

    def test_cli_register_with_metadata_inline(
        self, env_root, tmp_path, monkeypatch, capsys
    ):
        src = tmp_path / "src"
        _make_minimal_package(src)
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "register",
                "--task",
                "t",
                "--model",
                "m2",
                "--version",
                "v1",
                "--artifacts",
                str(src),
                "--metadata-inline",
                '{"framework": "sklearn"}',
            ],
        )
        out = capsys.readouterr()
        assert "Registered" in out.out or "m2" in out.out

    def test_cli_register_with_metadata_json_file(
        self, env_root, tmp_path, monkeypatch, capsys
    ):
        src = tmp_path / "src"
        _make_minimal_package(src)
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps({"framework": "pytorch"}))
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "register",
                "--task",
                "t",
                "--model",
                "m3",
                "--version",
                "v1",
                "--artifacts",
                str(src),
                "--metadata-json",
                str(meta_file),
            ],
        )
        out = capsys.readouterr()
        assert "Registered" in out.out or "m3" in out.out

    def test_cli_register_metadata_json_missing_raises(
        self, env_root, tmp_path, monkeypatch
    ):
        src = tmp_path / "src"
        _make_minimal_package(src)
        with pytest.raises(SystemExit) as exc_info:
            self._run_cli(
                monkeypatch,
                [
                    "omr",
                    "register",
                    "--task",
                    "t",
                    "--model",
                    "m4",
                    "--version",
                    "v1",
                    "--artifacts",
                    str(src),
                    "--metadata-json",
                    "/nonexistent/meta.json",
                ],
            )
        assert exc_info.value.code in (1, 2)

    def test_cli_resolve(self, env_root, tmp_path, monkeypatch, capsys):
        self._register_via_api(env_root, tmp_path)
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "resolve",
                "--task",
                "t",
                "--ref",
                "m@v1",
            ],
        )
        out = capsys.readouterr()
        assert "v1" in out.out or str(env_root) in out.out

    def test_cli_promote(self, env_root, tmp_path, monkeypatch, capsys):
        self._register_via_api(env_root, tmp_path)
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "promote",
                "--task",
                "t",
                "--model",
                "m",
                "--alias",
                "production",
                "--version",
                "v1",
                "--actor",
                "manish",
                "--reason",
                "release",
            ],
        )
        out = capsys.readouterr()
        assert "production" in out.out or "Promoted" in out.out

    def test_cli_verify(self, env_root, tmp_path, monkeypatch, capsys):
        self._register_via_api(env_root, tmp_path)
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "verify",
                "--task",
                "t",
                "--ref",
                "m@v1",
            ],
        )
        out = capsys.readouterr()
        assert "passed" in out.out or out.out.strip() != ""

    def test_cli_list(self, env_root, tmp_path, monkeypatch, capsys):
        self._register_via_api(env_root, tmp_path)
        self._run_cli(monkeypatch, ["omr", "list", "--task", "t"])
        out = capsys.readouterr()
        assert "m" in out.out

    def test_cli_list_no_models(self, env_root, monkeypatch, capsys):
        self._run_cli(monkeypatch, ["omr", "list", "--task", "nonexistent_task"])
        out = capsys.readouterr()
        assert "No models" in out.out

    def test_cli_show_pretty(self, env_root, tmp_path, monkeypatch, capsys):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={
                "framework": "sklearn",
                "model_type": "lr",
                "provenance": {
                    "git_commit": "abc123",
                    "training_data_ref": "gs://bucket/data",
                    "trainer_version": "1.0",
                },
            },
            set_alias=None,
        )
        self._run_cli(monkeypatch, ["omr", "show", "--task", "t", "--ref", "m@v1"])
        out = capsys.readouterr()
        assert "Task" in out.out or "Model" in out.out

    def test_cli_show_json(self, env_root, tmp_path, monkeypatch, capsys):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={"framework": "sklearn"},
            set_alias=None,
        )
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "show",
                "--task",
                "t",
                "--ref",
                "m@v1",
                "--json",
            ],
        )
        out = capsys.readouterr()
        data = json.loads(out.out)
        assert data["framework"] == "sklearn"

    def test_cli_show_raw(self, env_root, tmp_path, monkeypatch, capsys):
        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={"framework": "sklearn"},
            set_alias=None,
        )
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "show",
                "--task",
                "t",
                "--ref",
                "m@v1",
                "--raw",
            ],
        )
        out = capsys.readouterr()
        assert "sklearn" in out.out

    def test_cli_no_args_exits(self, env_root, monkeypatch):
        monkeypatch.setattr("sys.argv", ["omr"])
        with pytest.raises(SystemExit):
            runpy.run_path(self._get_cli_path(), run_name="__main__")

    def test_cli_registry_error_exits_1(self, env_root, monkeypatch):
        with pytest.raises(SystemExit) as exc_info:
            self._run_cli(
                monkeypatch,
                [
                    "omr",
                    "resolve",
                    "--task",
                    "t",
                    "--ref",
                    "nonexistent@v1",
                ],
            )
        assert exc_info.value.code == 1

    def test_cli_set_alias_empty_string_becomes_none(
        self, env_root, tmp_path, monkeypatch, capsys
    ):
        """Covers main(): set_alias == '' → None branch."""
        src = tmp_path / "src"
        _make_minimal_package(src)
        self._run_cli(
            monkeypatch,
            [
                "omr",
                "register",
                "--task",
                "t",
                "--model",
                "m_noalias",
                "--version",
                "v1",
                "--artifacts",
                str(src),
                "--set-alias",
                "",
            ],
        )
        out = capsys.readouterr()
        assert "Registered" in out.out or "m_noalias" in out.out

    def test_cli_show_missing_meta_exits_1(
        self, env_root, tmp_path, monkeypatch, capsys
    ):
        """Covers cli/main.py lines 92-93: model_meta.json missing → sys.exit(1)."""
        from omnibioai_model_registry.package import layout as L

        src = tmp_path / "src"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )
        vdir = L.version_dir(reg.root, "t", "m", "v1")

        monkeypatch.setattr(
            "omnibioai_model_registry.api.validate_package_files", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            "omnibioai_model_registry.api.verify_sha256_manifest", lambda *a, **kw: None
        )
        (vdir / "model_meta.json").unlink()

        monkeypatch.setattr("sys.argv", ["omr", "show", "--task", "t", "--ref", "m@v1"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(self._get_cli_path(), run_name="__main__")
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "model_meta.json not found" in err


# ============================================================
# package/layout.py — run path helpers (Phase 1)
# ============================================================


class TestLayoutRunPaths:

    def test_runs_root(self, tmp_path):
        from omnibioai_model_registry.package.layout import runs_root

        p = runs_root(tmp_path, "celltype_sc", "human_pbmc")
        assert p.parent.name == "human_pbmc"
        assert p.name == "runs"
        assert str(p).endswith("models/human_pbmc/runs")

    def test_run_dir(self, tmp_path):
        from omnibioai_model_registry.package.layout import run_dir

        p = run_dir(tmp_path, "t", "m", "run_abc123")
        assert p.name == "run_abc123"
        assert p.parent.name == "runs"

    def test_run_params_path(self, tmp_path):
        from omnibioai_model_registry.package.layout import run_params_path

        p = run_params_path(tmp_path, "t", "m", "r1")
        assert p.name == "params.json"
        assert p.parent.name == "r1"

    def test_run_tags_path(self, tmp_path):
        from omnibioai_model_registry.package.layout import run_tags_path

        p = run_tags_path(tmp_path, "t", "m", "r1")
        assert p.name == "tags.json"
        assert p.parent.name == "r1"

    def test_run_metric_log_path(self, tmp_path):
        from omnibioai_model_registry.package.layout import run_metric_log_path

        p = run_metric_log_path(tmp_path, "t", "m", "r1", "accuracy")
        assert p.name == "accuracy.jsonl"
        assert p.parent.name == "metrics"
        assert p.parent.parent.name == "r1"

    def test_version_tags_path(self, tmp_path):
        from omnibioai_model_registry.package.layout import version_tags_path

        p = version_tags_path(tmp_path, "t", "m", "v1")
        assert p.name == "tags.json"
        assert p.parent.name == "v1"

    def test_run_paths_share_consistent_prefix(self, tmp_path):
        """All run paths for the same run_id share the same parent run_dir."""
        from omnibioai_model_registry.package.layout import (
            run_dir,
            run_metric_log_path,
            run_params_path,
            run_tags_path,
        )

        base = run_dir(tmp_path, "t", "m", "rx")
        assert run_params_path(tmp_path, "t", "m", "rx").parent == base
        assert run_tags_path(tmp_path, "t", "m", "rx").parent == base
        assert run_metric_log_path(tmp_path, "t", "m", "rx", "loss").parent == base / "metrics"


# ============================================================
# run.py — RunLogger (Phase 1)
# ============================================================


class TestRunLogger:

    def test_run_id_auto_generated(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        assert isinstance(r.run_id, str)
        assert len(r.run_id) > 0

    def test_run_id_explicit(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", run_id="fixed_id", registry_root=tmp_path)
        assert r.run_id == "fixed_id"

    def test_run_id_stable_across_calls(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        rid = r.run_id
        r.log_param("x", 1)
        r.log_metric("acc", 0.9, step=0)
        r.set_tag("k", "v")
        assert r.run_id == rid

    def test_log_param_writes_params_json(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_param("lr", 0.001)

        params_file = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r.run_id / "params.json"
        )
        assert params_file.exists()
        data = json.loads(params_file.read_text())
        assert data["lr"] == 0.001

    def test_log_params_merges_all_keys(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_params({"lr": 0.001, "epochs": 50, "batch_size": 32})

        params_file = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r.run_id / "params.json"
        )
        data = json.loads(params_file.read_text())
        assert data["lr"] == 0.001
        assert data["epochs"] == 50
        assert data["batch_size"] == 32

    def test_log_param_then_log_params_cumulates(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_param("lr", 0.001)
        r.log_params({"epochs": 50})

        params_file = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r.run_id / "params.json"
        )
        data = json.loads(params_file.read_text())
        assert data["lr"] == 0.001
        assert data["epochs"] == 50

    def test_log_metric_creates_jsonl_file(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_metric("accuracy", 0.95, step=0)

        metric_file = (
            tmp_path
            / "tasks" / "t" / "models" / "m" / "runs" / r.run_id
            / "metrics" / "accuracy.jsonl"
        )
        assert metric_file.exists()
        lines = [ln for ln in metric_file.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["key"] == "accuracy"
        assert entry["value"] == 0.95
        assert entry["step"] == 0
        assert "ts_utc" in entry

    def test_log_metric_multiple_steps_appends(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_metric("loss", 1.0, step=0)
        r.log_metric("loss", 0.5, step=1)
        r.log_metric("loss", 0.2, step=2)

        metric_file = (
            tmp_path
            / "tasks" / "t" / "models" / "m" / "runs" / r.run_id
            / "metrics" / "loss.jsonl"
        )
        lines = [ln for ln in metric_file.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3
        steps = [json.loads(ln)["step"] for ln in lines]
        assert steps == [0, 1, 2]
        values = [json.loads(ln)["value"] for ln in lines]
        assert values == [1.0, 0.5, 0.2]

    def test_log_metrics_dict_creates_separate_files(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_metrics({"acc": 0.9, "f1": 0.85, "auc": 0.92}, step=0)

        metrics_dir = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r.run_id / "metrics"
        )
        assert (metrics_dir / "acc.jsonl").exists()
        assert (metrics_dir / "f1.jsonl").exists()
        assert (metrics_dir / "auc.jsonl").exists()
        assert json.loads((metrics_dir / "f1.jsonl").read_text().strip())["value"] == 0.85

    def test_set_tag_writes_tags_json(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.set_tag("team", "bioml")

        tags_file = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r.run_id / "tags.json"
        )
        assert tags_file.exists()
        data = json.loads(tags_file.read_text())
        assert data["team"] == "bioml"

    def test_set_tags_merges_all_keys(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.set_tags({"team": "bioml", "env": "training", "version": "v3"})

        tags_file = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r.run_id / "tags.json"
        )
        data = json.loads(tags_file.read_text())
        assert data["team"] == "bioml"
        assert data["env"] == "training"
        assert data["version"] == "v3"

    def test_set_tag_then_set_tags_cumulates(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.set_tag("team", "bioml")
        r.set_tags({"env": "training"})

        tags_file = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r.run_id / "tags.json"
        )
        data = json.loads(tags_file.read_text())
        assert data["team"] == "bioml"
        assert data["env"] == "training"

    def test_finish_returns_run_id(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_param("x", 1)
        result = r.finish()
        assert result == r.run_id

    def test_finish_flushes_params_and_tags(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r.log_param("lr", 0.01)
        r.set_tag("k", "v")
        rid = r.finish()

        base = tmp_path / "tasks" / "t" / "models" / "m" / "runs" / rid
        assert json.loads((base / "params.json").read_text())["lr"] == 0.01
        assert json.loads((base / "tags.json").read_text())["k"] == "v"

    def test_context_manager_calls_finish(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        with RunLogger(task="t", model_name="m", registry_root=tmp_path) as r:
            r.log_param("lr", 0.01)
            run_id = r.run_id

        params_file = (
            tmp_path / "tasks" / "t" / "models" / "m" / "runs" / run_id / "params.json"
        )
        assert params_file.exists()
        assert json.loads(params_file.read_text())["lr"] == 0.01

    def test_context_manager_with_metrics_and_tags(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        with RunLogger(task="t", model_name="m", registry_root=tmp_path) as r:
            r.log_params({"n_estimators": 100})
            r.log_metric("accuracy", 0.95, step=0)
            r.log_metric("accuracy", 0.97, step=1)
            r.set_tag("team", "bioml")
            run_id = r.run_id

        base = tmp_path / "tasks" / "t" / "models" / "m" / "runs" / run_id
        assert json.loads((base / "params.json").read_text())["n_estimators"] == 100
        assert json.loads((base / "tags.json").read_text())["team"] == "bioml"
        lines = [
            ln for ln in (base / "metrics" / "accuracy.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) == 2

    def test_two_instances_produce_separate_dirs(self, tmp_path):
        from omnibioai_model_registry.run import RunLogger

        r1 = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        r2 = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        assert r1.run_id != r2.run_id

        r1.log_param("x", 1)
        r2.log_param("x", 2)

        p1 = tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r1.run_id / "params.json"
        p2 = tmp_path / "tasks" / "t" / "models" / "m" / "runs" / r2.run_id / "params.json"
        assert json.loads(p1.read_text())["x"] == 1
        assert json.loads(p2.read_text())["x"] == 2

    def test_runlogger_exported_from_package(self):
        """Covers __init__.py export."""
        import omnibioai_model_registry as pkg

        assert hasattr(pkg, "RunLogger")
        assert pkg.RunLogger is not None

    def test_invalid_stage_transition_error_exists(self):
        """Covers errors.py: InvalidStageTransition is exported and inherits correctly."""
        from omnibioai_model_registry.errors import InvalidStageTransition, ModelRegistryError

        e = InvalidStageTransition("cannot go from production to none")
        assert isinstance(e, ModelRegistryError)
        assert "cannot go" in str(e)
