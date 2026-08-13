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


def _write_unowned_ownership(root: Path, task: str, model_name: str) -> None:
    """Phase 2B test helper: several older tests construct model/version
    directories directly on the filesystem (bypassing register_model) to
    isolate route/parsing logic under test, independent of registration.
    Since Phase 2B, every read/write route requires an ownership.json to
    exist at all -- a model with no ownership record is denied for every
    caller, the same as legacy_unowned (see ownership.py's
    check_model_ownership) -- so these fixtures now need one. This writes
    exactly the record register_model() itself would have produced for
    the same auth-disabled ("system", org_id=None) scenario these tests
    already run under."""
    from omnibioai_model_registry.ownership import ensure_model_ownership
    from omnibioai_model_registry.storage.localfs import LocalFS

    ensure_model_ownership(
        LocalFS(), root, task, model_name,
        organization_id=None, actor="test", model_pre_existing=False,
    )


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


def test_register_then_resolve_end_to_end_without_precomputed_manifest(
    env_root: Path, tmp_path: Path
):
    """
    Real regression guard for the v0.1.2 bug: register_model() called
    validate_package_files() (which requires sha256sums.txt to already
    exist, since it's in REQUIRED_FILES) BEFORE write_sha256_manifest()
    ever created that file. That made every registration fail unless the
    caller had already pre-supplied a sha256sums.txt of their own — which
    defeats the point of the registry generating it.

    This test intentionally does NOT put a sha256sums.txt in the
    artifacts_dir (unlike _make_minimal_package/registered_reg above,
    which pre-seed an empty one and would silently mask this exact bug).
    It exercises register_model() and resolve_model() end-to-end against
    a real temp filesystem registry root, with no mocking of write/copy/
    validate/verify logic, and asserts:
      - registration succeeds and actually writes a real sha256sums.txt
      - the manifest contains real hashes for the real files on disk
      - resolve_model() (which re-validates + re-verifies the manifest)
        succeeds for both the explicit version and the alias
    """
    src = tmp_path / "pkg_src_no_manifest"
    src.mkdir()
    (src / "model.pt").write_bytes(b"real weights bytes")
    (src / "model_genes.txt").write_text("GeneA\nGeneB\nGeneC\n", encoding="utf-8")
    (src / "label_map.json").write_text(
        json.dumps({"0": "A", "1": "B"}) + "\n", encoding="utf-8"
    )
    (src / "metrics.json").write_text(
        json.dumps({"acc": 0.95}) + "\n", encoding="utf-8"
    )
    (src / "feature_schema.json").write_text(
        json.dumps({"features": ["GeneA", "GeneB", "GeneC"]}) + "\n", encoding="utf-8"
    )
    (src / "model_meta.json").write_text(json.dumps({}) + "\n", encoding="utf-8")
    assert not (src / "sha256sums.txt").exists()

    reg = ModelRegistry.from_env()
    out = reg.register_model(
        task="t",
        model_name="e2e_model",
        version="v1",
        artifacts_dir=src,
        metadata={"framework": "pytorch"},
        set_alias="latest",
        actor="tester",
        reason="end-to-end regression test",
    )

    assert out["ok"] is True
    vdir = Path(out["package_path"])
    manifest_path = vdir / "sha256sums.txt"
    assert manifest_path.exists(), (
        "register_model() must generate sha256sums.txt itself; "
        "it must not require the caller to supply one"
    )

    from omnibioai_model_registry.package.manifest import (
        read_sha256_manifest,
        sha256_file,
    )

    on_disk_hashes = read_sha256_manifest(manifest_path)
    assert on_disk_hashes["model.pt"] == sha256_file(vdir / "model.pt")
    assert on_disk_hashes["model_meta.json"] == sha256_file(vdir / "model_meta.json")
    assert out["hashes"] == on_disk_hashes

    # resolve_model re-runs validate_package_files + verify_sha256_manifest
    # for real (no mocking) — this is the actual consumer path plugins use.
    resolved_by_version = reg.resolve_model(task="t", model_ref="e2e_model@v1", verify=True)
    resolved_by_alias = reg.resolve_model(task="t", model_ref="e2e_model@latest", verify=True)
    assert resolved_by_version.exists()
    assert str(resolved_by_version) == str(resolved_by_alias)


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


# ============================================================
# cli/main.py — Phase 3 new commands
# ============================================================


class TestCLINewCommands:

    def _get_cli_path(self):
        import omnibioai_model_registry.cli.main as cli_mod
        return cli_mod.__file__

    def _run_cli(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", argv)
        runpy.run_path(self._get_cli_path(), run_name="__main__")

    def _register(self, tmp_path, task="t", model="m", version="v1"):
        src = tmp_path / f"src_{task}_{model}_{version}"
        _make_minimal_package(src)
        reg = ModelRegistry.from_env()
        reg.register_model(
            task=task, model_name=model, version=version,
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        return reg

    # omr metrics -------------------------------------------------------

    def test_cli_metrics_prints_version_metrics(self, env_root, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        self._register(tmp_path)
        self._run_cli(monkeypatch, ["omr", "metrics", "--task", "t", "--ref", "m@v1"])
        out = capsys.readouterr().out
        assert "acc" in out

    def test_cli_metrics_json_output(self, env_root, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        self._register(tmp_path)
        self._run_cli(monkeypatch, ["omr", "metrics", "--task", "t", "--ref", "m@v1", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "acc" in data

    # omr aliases -------------------------------------------------------

    def test_cli_aliases_empty(self, env_root, tmp_path, monkeypatch, capsys):
        self._register(tmp_path)  # register with set_alias=None → no aliases
        self._run_cli(monkeypatch, ["omr", "aliases", "--task", "t", "--model", "m"])
        out = capsys.readouterr().out
        assert "No aliases" in out

    def test_cli_aliases_shows_promoted(self, env_root, tmp_path, monkeypatch, capsys):
        reg = self._register(tmp_path)
        reg.promote_model(task="t", model_name="m", alias="latest", version="v1")
        self._run_cli(monkeypatch, ["omr", "aliases", "--task", "t", "--model", "m"])
        out = capsys.readouterr().out
        assert "latest" in out

    def test_cli_aliases_json(self, env_root, tmp_path, monkeypatch, capsys):
        reg = self._register(tmp_path)
        reg.promote_model(task="t", model_name="m", alias="prod", version="v1")
        self._run_cli(monkeypatch, ["omr", "aliases", "--task", "t", "--model", "m", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert any(e.get("alias") == "prod" for e in data)

    # omr tag -----------------------------------------------------------

    def test_cli_tag_writes_to_meta(self, env_root, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        self._register(tmp_path)
        self._run_cli(monkeypatch, [
            "omr", "tag", "--task", "t", "--ref", "m@v1", "--key", "team", "--value", "bioml",
        ])
        out = capsys.readouterr().out
        assert "Tagged" in out
        from omnibioai_model_registry.package.layout import version_dir
        vdir = version_dir(env_root, "t", "m", "v1")
        meta = json.loads((vdir / "model_meta.json").read_text())
        assert meta["tags"]["team"] == "bioml"

    def test_cli_tag_idempotent_merge(self, env_root, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        self._register(tmp_path)
        self._run_cli(monkeypatch, [
            "omr", "tag", "--task", "t", "--ref", "m@v1", "--key", "k1", "--value", "v1",
        ])
        self._run_cli(monkeypatch, [
            "omr", "tag", "--task", "t", "--ref", "m@v1", "--key", "k2", "--value", "v2",
        ])
        from omnibioai_model_registry.package.layout import version_dir
        vdir = version_dir(env_root, "t", "m", "v1")
        meta = json.loads((vdir / "model_meta.json").read_text())
        assert meta["tags"]["k1"] == "v1"
        assert meta["tags"]["k2"] == "v2"

    # omr stage ---------------------------------------------------------

    def test_cli_stage_sets_meta(self, env_root, tmp_path, monkeypatch, capsys):
        self._register(tmp_path)
        self._run_cli(monkeypatch, [
            "omr", "stage", "--task", "t", "--model", "m", "--version", "v1", "--stage", "staging",
        ])
        out = capsys.readouterr().out
        assert "staging" in out
        from omnibioai_model_registry.package.layout import version_dir
        vdir = version_dir(env_root, "t", "m", "v1")
        meta = json.loads((vdir / "model_meta.json").read_text())
        assert meta["stage"] == "staging"

    def test_cli_stage_production_creates_alias(self, env_root, tmp_path, monkeypatch, capsys):
        self._register(tmp_path)
        self._run_cli(monkeypatch, [
            "omr", "stage", "--task", "t", "--model", "m", "--version", "v1",
            "--stage", "production", "--actor", "ci",
        ])
        alias_file = env_root / "tasks" / "t" / "models" / "m" / "aliases" / "production.json"
        assert alias_file.exists()
        data = json.loads(alias_file.read_text())
        assert data["version"] == "v1"

    def test_cli_stage_rejects_invalid(self, env_root, tmp_path, monkeypatch):
        self._register(tmp_path)
        with pytest.raises(SystemExit) as exc:
            self._run_cli(monkeypatch, [
                "omr", "stage", "--task", "t", "--model", "m",
                "--version", "v1", "--stage", "deployed",
            ])
        assert exc.value.code == 1

    def test_cli_stage_none_does_not_create_alias(self, env_root, tmp_path, monkeypatch, capsys):
        self._register(tmp_path)
        self._run_cli(monkeypatch, [
            "omr", "stage", "--task", "t", "--model", "m", "--version", "v1", "--stage", "none",
        ])
        none_alias = env_root / "tasks" / "t" / "models" / "m" / "aliases" / "none.json"
        assert not none_alias.exists()

    # omr compare -------------------------------------------------------

    def test_cli_compare_prints_table(self, env_root, tmp_path, monkeypatch, capsys):
        self._register(tmp_path, version="v1")
        self._register(tmp_path, version="v2")
        self._run_cli(monkeypatch, [
            "omr", "compare", "--task", "t", "--model", "m", "--versions", "v1", "v2",
        ])
        out = capsys.readouterr().out
        assert "acc" in out
        assert "v1" in out
        assert "v2" in out

    def test_cli_compare_json(self, env_root, tmp_path, monkeypatch, capsys):
        self._register(tmp_path, version="v1")
        self._register(tmp_path, version="v2")
        self._run_cli(monkeypatch, [
            "omr", "compare", "--task", "t", "--model", "m",
            "--versions", "v1", "v2", "--json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "v1" in data["versions"]
        assert "v2" in data["versions"]


# ============================================================
# service/app/main.py — Phase 3 routes
# ============================================================


@pytest.fixture
def svc_client(tmp_path, monkeypatch):
    """TestClient with module-level registry patched to a tmp root."""
    root = tmp_path / "registry"
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")

    import omnibioai_model_registry.service.app.main as _svc
    from fastapi.testclient import TestClient

    new_reg = _svc.ModelRegistry.from_env()
    monkeypatch.setattr(_svc, "registry", new_reg)
    return TestClient(_svc.app, raise_server_exceptions=False), new_reg.root


class TestServicePhase3Routes:

    # GET /v1/aliases ---------------------------------------------------

    def test_get_aliases_returns_entries(self, svc_client):
        client, root = svc_client
        _write_unowned_ownership(root, "t", "m")
        aliases_dir = root / "tasks" / "t" / "models" / "m" / "aliases"
        aliases_dir.mkdir(parents=True)
        (aliases_dir / "latest.json").write_text(json.dumps({
            "alias": "latest", "version": "v1", "task": "t", "model_name": "m",
            "updated_at": "2026-01-01T00:00:00Z", "actor": "test", "reason": "ci",
        }))
        r = client.get("/v1/aliases", params={"task": "t", "model": "m"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert len(data["aliases"]) == 1
        assert data["aliases"][0]["alias"] == "latest"
        assert data["aliases"][0]["version"] == "v1"

    def test_get_aliases_empty_model(self, svc_client):
        """Phase 2B: a model with no ownership record at all (never
        registered) is denied the same anti-enumerating way an
        other-org-owned model would be -- 404, not a soft empty list."""
        client, root = svc_client
        r = client.get("/v1/aliases", params={"task": "t", "model": "nomodel"})
        assert r.status_code == 404

    # POST /v1/stage ----------------------------------------------------

    def test_post_stage_updates_meta_and_returns_stage(self, svc_client, tmp_path):
        client, root = svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        r = client.post("/v1/stage", json={
            "task": "t", "model_name": "m", "version": "v1",
            "stage": "staging", "actor": "ci",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["stage"] == "staging"
        assert data["version"] == "v1"
        meta_path = root / "tasks" / "t" / "models" / "m" / "versions" / "v1" / "model_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["stage"] == "staging"

    def test_post_stage_production_creates_alias(self, svc_client, tmp_path):
        client, root = svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        r = client.post("/v1/stage", json={
            "task": "t", "model_name": "m", "version": "v1", "stage": "production",
        })
        assert r.status_code == 200
        alias_file = root / "tasks" / "t" / "models" / "m" / "aliases" / "production.json"
        assert alias_file.exists()

    def test_post_stage_rejects_invalid(self, svc_client):
        client, _ = svc_client
        r = client.post("/v1/stage", json={
            "task": "t", "model_name": "m", "version": "v1", "stage": "deployed",
        })
        assert r.status_code == 400

    # GET /v1/compare ---------------------------------------------------

    def test_get_compare_returns_metrics_for_both_versions(self, svc_client):
        client, root = svc_client
        _write_unowned_ownership(root, "t", "m")
        for ver, acc in [("v1", 0.9), ("v2", 0.85)]:
            vdir = root / "tasks" / "t" / "models" / "m" / "versions" / ver
            vdir.mkdir(parents=True)
            (vdir / "metrics.json").write_text(json.dumps({"accuracy": acc}))
            (vdir / "model_meta.json").write_text(json.dumps({
                "task": "t", "model_name": "m", "version": ver, "stage": "none",
            }))
        r = client.get("/v1/compare", params={"task": "t", "model": "m", "versions": ["v1", "v2"]})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["versions"]["v1"]["metrics"]["accuracy"] == 0.9
        assert data["versions"]["v2"]["metrics"]["accuracy"] == 0.85

    def test_get_compare_requires_at_least_two_versions(self, svc_client):
        client, _ = svc_client
        r = client.get("/v1/compare", params={"task": "t", "model": "m", "versions": ["v1"]})
        assert r.status_code == 400

    def test_get_compare_zero_versions_returns_400(self, svc_client):
        client, _ = svc_client
        r = client.get("/v1/compare", params={"task": "t", "model": "m"})
        assert r.status_code == 400

    # GET /v1/models with metric_gte ------------------------------------

    def test_get_models_metric_gte_filters_correctly(self, svc_client):
        client, root = svc_client
        _write_unowned_ownership(root, "t", "m")
        for ver, acc in [("v1", 0.95), ("v2", 0.80)]:
            vdir = root / "tasks" / "t" / "models" / "m" / "versions" / ver
            vdir.mkdir(parents=True)
            (vdir / "model_meta.json").write_text(json.dumps({
                "task": "t", "model_name": "m", "version": ver,
            }))
            (vdir / "metrics.json").write_text(json.dumps({"accuracy": acc}))
        r = client.get("/v1/models", params={"metric_gte": "accuracy:0.9"})
        assert r.status_code == 200
        data = r.json()
        versions = [m["version"] for m in data]
        assert "v1" in versions
        assert "v2" not in versions

    def test_get_models_without_metric_gte_returns_all(self, svc_client):
        client, root = svc_client
        _write_unowned_ownership(root, "t", "m")
        for ver in ["v1", "v2"]:
            vdir = root / "tasks" / "t" / "models" / "m" / "versions" / ver
            vdir.mkdir(parents=True)
            (vdir / "model_meta.json").write_text(json.dumps({
                "task": "t", "model_name": "m", "version": ver,
            }))
        r = client.get("/v1/models")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_get_models_metric_gte_invalid_format_returns_400(self, svc_client):
        client, _ = svc_client
        r = client.get("/v1/models", params={"metric_gte": "no_colon_here"})
        assert r.status_code == 400


# ============================================================
# db.py + tracking.py — TestDBAndTracking
# ============================================================


class TestDBAndTracking:

    @pytest.fixture
    def mock_conn(self):
        from unittest.mock import MagicMock

        cursor = MagicMock()
        cursor.rowcount = 1

        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False
        return conn, cursor

    # ── db.py ──────────────────────────────────────────────────────────────

    def test_get_connection_raises_without_db_host(self, monkeypatch):
        monkeypatch.delenv("DB_HOST", raising=False)
        from omnibioai_model_registry.db import get_connection
        from omnibioai_model_registry.errors import RegistryNotConfigured

        with pytest.raises(RegistryNotConfigured):
            get_connection()

    def test_get_connection_calls_pymysql_connect(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_USER", "root")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("DB_NAME", "testdb")

        fake_conn = MagicMock()
        with patch("pymysql.connect", return_value=fake_conn):
            from omnibioai_model_registry.db import get_connection

            conn = get_connection()
        assert conn is fake_conn

    def test_init_tables_executes_all_ddl(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.db import _ALTER_DDL, _DDL, init_tables

        init_tables(conn)
        # Phase 2C: init_tables() now also runs the additive-only
        # omr_runs column migration after the CREATE TABLE loop.
        assert cursor.execute.call_count == len(_DDL) + len(_ALTER_DDL)

    # ── db.py — Phase 2C additive-only omr_runs migration ───────────────────

    def test_alter_ddl_idempotent_swallows_duplicate_column(self, mock_conn):
        """A duplicate-column error (MySQL 1060) on either ALTER
        statement -- the expected outcome on every startup after the
        first, or immediately on a freshly created table since _DDL
        already includes both columns -- must be silently swallowed, not
        raised."""
        conn, cursor = mock_conn
        from omnibioai_model_registry.db import _run_alter_ddl_idempotent

        dup_col_error = Exception()
        dup_col_error.args = (1060, "Duplicate column name 'organization_id'")
        cursor.execute.side_effect = [dup_col_error, dup_col_error]
        # Should not raise.
        _run_alter_ddl_idempotent(conn)

    def test_alter_ddl_idempotent_reraises_other_errors(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.db import _run_alter_ddl_idempotent

        other_error = Exception()
        other_error.args = (1146, "Table 'omr_runs' doesn't exist")
        cursor.execute.side_effect = other_error
        with pytest.raises(Exception) as excinfo:
            _run_alter_ddl_idempotent(conn)
        assert excinfo.value.args[0] == 1146

    def test_alter_ddl_applies_cleanly_on_first_run(self, mock_conn):
        """No error at all (e.g. a genuinely pre-2C table): both ALTER
        statements execute without being swallowed."""
        conn, cursor = mock_conn
        from omnibioai_model_registry.db import _ALTER_DDL, _run_alter_ddl_idempotent

        _run_alter_ddl_idempotent(conn)
        assert cursor.execute.call_count == len(_ALTER_DDL)

    # ── tracking.py — create / finish ──────────────────────────────────────

    def test_create_run_returns_run_dict(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchone.return_value = {
            "run_id": "r1",
            "task": "t",
            "model_name": "m",
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "actor": None,
        }
        cursor.fetchall.return_value = []

        from omnibioai_model_registry.tracking import create_run

        result = create_run(conn, task="t", model_name="m", run_id="r1")
        assert result["run_id"] == "r1"
        assert result["status"] == "running"

    def test_finish_run_updates_status(self, mock_conn):
        conn, cursor = mock_conn
        cursor.rowcount = 1
        from omnibioai_model_registry.tracking import finish_run

        finish_run(conn, "r1")
        sqls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        assert any("UPDATE" in s for s in sqls)

    def test_finish_run_raises_if_not_found(self, mock_conn):
        conn, cursor = mock_conn
        cursor.rowcount = 0
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import finish_run

        with pytest.raises(ModelNotFound):
            finish_run(conn, "nonexistent")

    # ── tracking.py — params ───────────────────────────────────────────────

    def test_log_param_executes_insert(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import log_param

        log_param(conn, "r1", "lr", 0.001, task="t", model_name="m")
        sqls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        assert any("INSERT" in s for s in sqls)

    def test_log_params_calls_log_param_for_each(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import log_params

        log_params(conn, "r1", {"lr": 0.001, "epochs": 50}, task="t", model_name="m")
        assert cursor.execute.call_count >= 4  # 2 params × 2 executes each

    # ── tracking.py — metrics ──────────────────────────────────────────────

    def test_log_metric_executes_insert(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import log_metric

        log_metric(conn, "r1", "accuracy", 0.95, step=1, task="t", model_name="m")
        sqls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        assert any("INSERT" in s for s in sqls)

    def test_log_metrics_iterates_dict(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import log_metrics

        log_metrics(conn, "r1", {"acc": 0.9, "f1": 0.85}, step=0, task="t", model_name="m")
        assert cursor.execute.call_count >= 4  # 2 metrics × 2 executes each

    # ── tracking.py — tags ─────────────────────────────────────────────────

    def test_set_tag_executes_insert(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import set_tag

        set_tag(conn, "r1", "team", "bioml", task="t", model_name="m")
        sqls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        assert any("INSERT" in s for s in sqls)

    def test_set_tags_iterates_dict(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import set_tags

        set_tags(conn, "r1", {"team": "bioml", "env": "prod"}, task="t", model_name="m")
        assert cursor.execute.call_count >= 4

    # ── tracking.py — get_run ──────────────────────────────────────────────

    def test_get_run_returns_full_snapshot(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchone.return_value = {
            "run_id": "r1",
            "task": "t",
            "model_name": "m",
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "actor": "manish",
        }
        cursor.fetchall.return_value = []

        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1")
        assert result["run_id"] == "r1"
        assert result["task"] == "t"
        assert result["actor"] == "manish"
        assert isinstance(result["params"], dict)
        assert isinstance(result["tags"], dict)
        assert isinstance(result["metrics_summary"], dict)
        assert result["finished_at"] is None
        assert "T" in result["started_at"]

    def test_get_run_with_finished_datetime(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchone.return_value = {
            "run_id": "r1",
            "task": "t",
            "model_name": "m",
            "status": "finished",
            "started_at": now,
            "finished_at": now,
            "actor": None,
        }
        cursor.fetchall.return_value = []

        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1")
        assert result["finished_at"] is not None
        assert "T" in result["finished_at"]

    def test_get_run_with_none_started_at(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {
            "run_id": "r1",
            "task": "t",
            "model_name": "m",
            "status": "running",
            "started_at": None,
            "finished_at": None,
            "actor": None,
        }
        cursor.fetchall.return_value = []

        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1")
        assert result["started_at"] is None

    def test_get_run_raises_if_not_found(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None

        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import get_run

        with pytest.raises(ModelNotFound):
            get_run(conn, "nonexistent")

    def test_get_run_parses_params_from_json(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchone.return_value = {
            "run_id": "r1",
            "task": "t",
            "model_name": "m",
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "actor": None,
        }
        cursor.fetchall.side_effect = [
            [{"key_name": "lr", "value_text": "0.001"}],
            [],
            [],
        ]

        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1")
        assert result["params"]["lr"] == 0.001

    def test_get_run_handles_invalid_json_param(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchone.return_value = {
            "run_id": "r1",
            "task": "t",
            "model_name": "m",
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "actor": None,
        }
        cursor.fetchall.side_effect = [
            [{"key_name": "mode", "value_text": "not_json_value"}],
            [],
            [],
        ]

        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1")
        assert result["params"]["mode"] == "not_json_value"

    def test_get_run_metrics_summary_deduplicates_keys(self, mock_conn):
        """Covers the 'key already in metrics_summary' branch."""
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchone.return_value = {
            "run_id": "r1",
            "task": "t",
            "model_name": "m",
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "actor": None,
        }
        cursor.fetchall.side_effect = [
            [],
            [],
            [
                {"key_name": "acc", "value": 0.97, "step": 1},
                {"key_name": "acc", "value": 0.95, "step": 0},
            ],
        ]

        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1")
        assert result["metrics_summary"]["acc"] == 0.97

    # ── tracking.py — get_metric_history ───────────────────────────────────

    def test_get_metric_history_returns_list(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchall.return_value = [
            {"value": 0.9, "step": 0, "ts_utc": now},
            {"value": 0.95, "step": 1, "ts_utc": now},
        ]

        from omnibioai_model_registry.tracking import get_metric_history

        result = get_metric_history(conn, "r1", "accuracy")
        assert len(result) == 2
        assert result[0]["value"] == 0.9
        assert result[1]["step"] == 1
        assert "T" in result[0]["ts_utc"]

    def test_get_metric_history_handles_none_ts_utc(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = [
            {"value": 0.9, "step": 0, "ts_utc": None},
        ]

        from omnibioai_model_registry.tracking import get_metric_history

        result = get_metric_history(conn, "r1", "accuracy")
        assert result[0]["ts_utc"] is None

    # ── tracking.py — list_runs ────────────────────────────────────────────

    def test_list_runs_returns_list(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchall.return_value = [
            {"run_id": "r1", "status": "running", "started_at": now, "finished_at": None},
        ]

        from omnibioai_model_registry.tracking import list_runs

        result = list_runs(conn, "t", "m")
        assert len(result) == 1
        assert result[0]["run_id"] == "r1"
        assert result[0]["finished_at"] is None

    def test_list_runs_with_finished_at(self, mock_conn):
        from datetime import datetime, timezone

        conn, cursor = mock_conn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.fetchall.return_value = [
            {"run_id": "r1", "status": "finished", "started_at": now, "finished_at": now},
        ]

        from omnibioai_model_registry.tracking import list_runs

        result = list_runs(conn, "t", "m")
        assert result[0]["finished_at"] is not None

    def test_list_runs_empty(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []

        from omnibioai_model_registry.tracking import list_runs

        assert list_runs(conn, "t", "m") == []

    # ── tracking.py — version tags ─────────────────────────────────────────

    def test_set_version_tag_executes_insert(self, mock_conn):
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import set_version_tag

        set_version_tag(conn, "t", "m", "v1", "team", "bioml")
        sqls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        assert any("INSERT" in s for s in sqls)

    def test_get_version_tags_returns_dict(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = [
            {"key_name": "team", "value_text": "bioml"},
            {"key_name": "env", "value_text": "prod"},
        ]

        from omnibioai_model_registry.tracking import get_version_tags

        result = get_version_tags(conn, "t", "m", "v1")
        assert result["team"] == "bioml"
        assert result["env"] == "prod"

    def test_get_version_tags_empty(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []

        from omnibioai_model_registry.tracking import get_version_tags

        assert get_version_tags(conn, "t", "m", "v1") == {}

# ============================================================
# Phase 2C — tracking-layer organization-scoped isolation
# ============================================================
#
# Pure/mocked-cursor unit coverage of tracking.py's new run-ownership
# decision logic (mirrors TestCheckModelOwnership's structure for
# ownership.py) and the version-tag ownership.json integration. HTTP-
# level route coverage is in TestPhase2CHTTPRunTracking further below.


class TestTrackingPhase2COwnership:
    """Unit-level coverage of _evaluate_run_ownership()/
    check_run_ownership() and the enforce_ownership gate threaded
    through every tracking.py read/write function -- independent of the
    HTTP layer."""

    @pytest.fixture
    def mock_conn(self):
        from unittest.mock import MagicMock

        cursor = MagicMock()
        cursor.rowcount = 1
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False
        return conn, cursor

    # ── _evaluate_run_ownership (pure decision, mirrors ownership.py) ──────

    def test_owned_matching_org_allowed(self):
        from omnibioai_model_registry.tracking import _evaluate_run_ownership

        result = _evaluate_run_ownership("org-A", "owned", requesting_org_id="org-A")
        assert result.allowed is True
        assert result.reason == "owned_by_caller"

    def test_owned_different_org_denied(self):
        from omnibioai_model_registry.tracking import _evaluate_run_ownership

        result = _evaluate_run_ownership("org-A", "owned", requesting_org_id="org-B")
        assert result.allowed is False
        assert result.reason == "owned_by_other_org"

    def test_unowned_open_mode_caller_allowed(self):
        from omnibioai_model_registry.tracking import _evaluate_run_ownership

        result = _evaluate_run_ownership(None, "unowned", requesting_org_id=None)
        assert result.allowed is True
        assert result.reason == "open_mode_match"

    def test_unowned_real_org_caller_denied(self):
        """A real org_id reaching into a None-org run is NOT 'everyone's'."""
        from omnibioai_model_registry.tracking import _evaluate_run_ownership

        result = _evaluate_run_ownership(None, "unowned", requesting_org_id="org-A")
        assert result.allowed is False
        assert result.reason == "owned_by_other_org"

    def test_legacy_unowned_denied_for_every_caller(self):
        from omnibioai_model_registry.tracking import _evaluate_run_ownership

        for requesting_org_id in ("org-A", None):
            result = _evaluate_run_ownership("org-A", "legacy_unowned", requesting_org_id=requesting_org_id)
            assert result.allowed is False
            assert result.reason == "legacy_unowned"

    # ── check_run_ownership (DB-touching wrapper) ───────────────────────────

    def test_check_run_ownership_not_found(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None
        from omnibioai_model_registry.tracking import check_run_ownership

        result = check_run_ownership(conn, "nonexistent", requesting_org_id="org-A")
        assert result.allowed is False
        assert result.exists is False
        assert result.reason == "run_not_found"

    def test_check_run_ownership_owned_by_caller(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.tracking import check_run_ownership

        result = check_run_ownership(conn, "r1", requesting_org_id="org-A")
        assert result.allowed is True
        assert result.exists is True

    # ── _ensure_run (write-path pre-check) ──────────────────────────────────

    def test_ensure_run_creates_new_run_attributed_to_caller(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None  # not found -> creatable
        from omnibioai_model_registry.tracking import _ensure_run

        _ensure_run(
            conn, "r1", "t", "m", "alice",
            requesting_org_id="org-A", enforce_ownership=True,
        )
        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT IGNORE" in str(c[0][0])]
        assert len(insert_calls) == 1
        params = insert_calls[0][0][1]
        assert params[-2:] == ("org-A", "owned")

    def test_ensure_run_new_run_no_org_is_unowned(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None
        from omnibioai_model_registry.tracking import _ensure_run

        _ensure_run(conn, "r1", "t", "m", enforce_ownership=True, requesting_org_id=None)
        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT IGNORE" in str(c[0][0])]
        params = insert_calls[0][0][1]
        assert params[-2:] == (None, "unowned")

    def test_ensure_run_existing_run_other_org_denied_no_insert(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import _ensure_run

        with pytest.raises(ModelNotFound):
            _ensure_run(
                conn, "r1", "t", "m",
                requesting_org_id="org-B", enforce_ownership=True,
            )
        # Security requirement: cross-org denial causes no mutation --
        # the INSERT IGNORE must never have been attempted.
        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT IGNORE" in str(c[0][0])]
        assert insert_calls == []

    def test_ensure_run_existing_run_same_org_allowed(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.tracking import _ensure_run

        _ensure_run(
            conn, "r1", "t", "m",
            requesting_org_id="org-A", enforce_ownership=True,
        )
        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT IGNORE" in str(c[0][0])]
        assert len(insert_calls) == 1

    def test_ensure_run_legacy_unowned_denied(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": None, "ownership_status": "legacy_unowned"}
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import _ensure_run

        with pytest.raises(ModelNotFound):
            _ensure_run(conn, "r1", "t", "m", requesting_org_id="org-A", enforce_ownership=True)
        with pytest.raises(ModelNotFound):
            _ensure_run(conn, "r1", "t", "m", requesting_org_id=None, enforce_ownership=True)

    def test_ensure_run_enforce_ownership_false_skips_check_entirely(self, mock_conn):
        """Backward-compat regression guard: CLI/direct Python API callers
        (enforce_ownership defaults to False) never issue the ownership
        SELECT at all -- existing behavior fully unchanged."""
        conn, cursor = mock_conn
        from omnibioai_model_registry.tracking import _ensure_run

        _ensure_run(conn, "r1", "t", "m", "alice")
        select_calls = [c for c in cursor.execute.call_args_list if "SELECT organization_id" in str(c[0][0])]
        assert select_calls == []
        cursor.fetchone.assert_not_called()

    # ── log_metric / log_param / set_tag thread enforcement through ────────

    def test_log_metric_cross_org_denied_no_metric_insert(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import log_metric

        with pytest.raises(ModelNotFound):
            log_metric(
                conn, "r1", "acc", 0.9, task="t", model_name="m",
                requesting_org_id="org-B", enforce_ownership=True,
            )
        metric_inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_metrics" in str(c[0][0])]
        assert metric_inserts == []

    def test_log_param_cross_org_denied_no_param_insert(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import log_param

        with pytest.raises(ModelNotFound):
            log_param(
                conn, "r1", "lr", 0.01, task="t", model_name="m",
                requesting_org_id="org-B", enforce_ownership=True,
            )
        param_inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_params" in str(c[0][0])]
        assert param_inserts == []

    def test_set_tag_cross_org_denied_no_tag_insert(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import set_tag

        with pytest.raises(ModelNotFound):
            set_tag(
                conn, "r1", "team", "attacker", task="t", model_name="m",
                requesting_org_id="org-B", enforce_ownership=True,
            )
        tag_inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_tags" in str(c[0][0])]
        assert tag_inserts == []

    def test_log_metric_same_org_allowed(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.tracking import log_metric

        log_metric(
            conn, "r1", "acc", 0.9, task="t", model_name="m",
            requesting_org_id="org-A", enforce_ownership=True,
        )
        metric_inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_metrics" in str(c[0][0])]
        assert len(metric_inserts) == 1

    # ── get_run enforcement (reuses the row it already fetched) ────────────

    def _run_row(self, **overrides):
        from datetime import datetime, timezone

        row = {
            "run_id": "r1", "task": "t", "model_name": "m", "status": "running",
            "started_at": datetime.now(timezone.utc).replace(tzinfo=None), "finished_at": None,
            "actor": None, "organization_id": "org-A", "ownership_status": "owned",
        }
        row.update(overrides)
        return row

    def test_get_run_same_org_allowed(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = self._run_row()
        cursor.fetchall.return_value = []
        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1", requesting_org_id="org-A", enforce_ownership=True)
        assert result["run_id"] == "r1"

    def test_get_run_other_org_denied_same_shape_as_not_found(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = self._run_row()
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import get_run

        with pytest.raises(ModelNotFound, match="Run not found: r1"):
            get_run(conn, "r1", requesting_org_id="org-B", enforce_ownership=True)

    def test_get_run_legacy_unowned_denied(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = self._run_row(organization_id=None, ownership_status="legacy_unowned")
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import get_run

        with pytest.raises(ModelNotFound):
            get_run(conn, "r1", requesting_org_id="org-A", enforce_ownership=True)

    def test_get_run_enforce_ownership_false_ignores_mismatch(self, mock_conn):
        """Backward-compat regression guard: a library/CLI caller that
        never passes enforce_ownership gets the pre-Phase-2C behavior
        exactly -- org mismatch present in the row is simply not
        consulted."""
        conn, cursor = mock_conn
        cursor.fetchone.return_value = self._run_row(organization_id="org-A")
        cursor.fetchall.return_value = []
        from omnibioai_model_registry.tracking import get_run

        result = get_run(conn, "r1", requesting_org_id="org-B")  # enforce_ownership defaults False
        assert result["run_id"] == "r1"

    # ── list_runs query-layer filtering ──────────────────────────────────────

    def test_list_runs_enforce_ownership_adds_org_filter(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []
        from omnibioai_model_registry.tracking import list_runs

        list_runs(conn, "t", "m", requesting_org_id="org-A", enforce_ownership=True)
        sql, params = cursor.execute.call_args[0]
        assert "organization_id" in sql
        assert "ownership_status" in sql
        assert "org-A" in params

    def test_list_runs_enforce_ownership_false_is_unfiltered_original_query(self, mock_conn):
        """Regression guard: the exact pre-Phase-2C query/behavior for
        library/CLI callers that don't opt in."""
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []
        from omnibioai_model_registry.tracking import list_runs

        list_runs(conn, "t", "m")
        sql, params = cursor.execute.call_args[0]
        assert "organization_id" not in sql
        assert params == ("t", "m")

    # ── get_metric_history enforcement ──────────────────────────────────────

    def test_get_metric_history_cross_org_denied_no_metrics_query(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import get_metric_history

        with pytest.raises(ModelNotFound):
            get_metric_history(conn, "r1", "acc", requesting_org_id="org-B", enforce_ownership=True)
        metric_selects = [c for c in cursor.execute.call_args_list if "FROM omr_metrics" in str(c[0][0])]
        assert metric_selects == []

    def test_get_metric_history_same_org_allowed(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        cursor.fetchall.return_value = []
        from omnibioai_model_registry.tracking import get_metric_history

        result = get_metric_history(conn, "r1", "acc", requesting_org_id="org-A", enforce_ownership=True)
        assert result == []

    # ── finish_run / create_run passthrough ─────────────────────────────────

    def test_finish_run_cross_org_denied_no_update(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import finish_run

        with pytest.raises(ModelNotFound):
            finish_run(conn, "r1", requesting_org_id="org-B", enforce_ownership=True)
        update_calls = [c for c in cursor.execute.call_args_list if "UPDATE omr_runs" in str(c[0][0])]
        assert update_calls == []

    def test_create_run_threads_ownership_through(self, mock_conn):
        conn, cursor = mock_conn
        cursor.fetchone.return_value = None  # brand new
        cursor.fetchall.return_value = []
        from omnibioai_model_registry.tracking import create_run

        # After _ensure_run's INSERT IGNORE, get_run's own SELECT * must
        # return something -- simulate the freshly created row.
        cursor.fetchone.side_effect = [None, self._run_row(organization_id="org-A")]
        result = create_run(conn, "t", "m", "r1", requesting_org_id="org-A", enforce_ownership=True)
        assert result["run_id"] == "r1"


class TestVersionTagOwnership:
    """set_version_tag()/get_version_tags() -- Phase 2C: no independent
    organization_id column, boundary derived from the SAME
    ownership.json Phase 2A/2B already established (real filesystem via
    tmp_path/env_root, mocked DB cursor)."""

    @pytest.fixture
    def mock_conn(self):
        from unittest.mock import MagicMock

        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False
        return conn, cursor

    def _own(self, env_root, task, model_name, org_id, *, legacy=False):
        from omnibioai_model_registry.ownership import ensure_model_ownership
        from omnibioai_model_registry.storage.localfs import LocalFS

        ensure_model_ownership(
            LocalFS(), env_root, task, model_name,
            organization_id=None if legacy else org_id,
            actor="alice", model_pre_existing=legacy,
        )

    def test_set_version_tag_same_org_allowed(self, env_root, mock_conn):
        conn, cursor = mock_conn
        self._own(env_root, "t", "m", "org-A")
        from omnibioai_model_registry.tracking import set_version_tag

        set_version_tag(
            conn, "t", "m", "v1", "team", "bioml",
            registry_root=env_root, requesting_org_id="org-A", enforce_ownership=True,
        )
        inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_version_tags" in str(c[0][0])]
        assert len(inserts) == 1

    def test_set_version_tag_other_org_denied_no_insert(self, env_root, mock_conn):
        conn, cursor = mock_conn
        self._own(env_root, "t", "m", "org-A")
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import set_version_tag

        with pytest.raises(ModelNotFound):
            set_version_tag(
                conn, "t", "m", "v1", "team", "attacker",
                registry_root=env_root, requesting_org_id="org-B", enforce_ownership=True,
            )
        inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_version_tags" in str(c[0][0])]
        assert inserts == []

    def test_set_version_tag_legacy_unowned_denied(self, env_root, mock_conn):
        conn, cursor = mock_conn
        self._own(env_root, "t", "old_model", None, legacy=True)
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import set_version_tag

        with pytest.raises(ModelNotFound):
            set_version_tag(
                conn, "t", "old_model", "v1", "team", "x",
                registry_root=env_root, requesting_org_id="org-A", enforce_ownership=True,
            )

    def test_get_version_tags_other_org_denied(self, env_root, mock_conn):
        conn, cursor = mock_conn
        self._own(env_root, "t", "m", "org-A")
        from omnibioai_model_registry.errors import ModelNotFound
        from omnibioai_model_registry.tracking import get_version_tags

        with pytest.raises(ModelNotFound):
            get_version_tags(
                conn, "t", "m", "v1",
                registry_root=env_root, requesting_org_id="org-B", enforce_ownership=True,
            )
        selects = [c for c in cursor.execute.call_args_list if "FROM omr_version_tags" in str(c[0][0])]
        assert selects == []

    def test_set_version_tag_enforce_ownership_false_unchanged_behavior(self, env_root, mock_conn):
        """Backward-compat regression guard: the CLI's `omr tag` command
        (cli/main.py) calls set_version_tag() with no IAM identity at
        all -- must remain completely unaffected."""
        conn, cursor = mock_conn
        self._own(env_root, "t", "m", "org-A")
        from omnibioai_model_registry.tracking import set_version_tag

        set_version_tag(conn, "t", "m", "v1", "team", "bioml")
        inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_version_tags" in str(c[0][0])]
        assert len(inserts) == 1

# ============================================================
# plugin_client.py — TestPluginClient
# ============================================================


class TestPluginClient:

    def _fake_urlopen(self, posted: list, body: bytes = b'{"ok": true}'):
        """Return a fake urlopen side_effect that records the posted payloads."""
        from unittest.mock import MagicMock

        def side_effect(req, **kwargs):
            posted.append(json.loads(req.data))
            mock_resp = MagicMock()
            mock_resp.read.return_value = body
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        return side_effect

    # ── __init__ ───────────────────────────────────────────────────────────

    def test_init_with_explicit_url(self):
        from omnibioai_model_registry.plugin_client import PluginClient

        client = PluginClient(registry_url="http://example.com:9000")
        assert client._url == "http://example.com:9000"

    def test_init_strips_trailing_slash(self):
        from omnibioai_model_registry.plugin_client import PluginClient

        client = PluginClient(registry_url="http://example.com/")
        assert client._url == "http://example.com"

    def test_init_without_url_uses_default(self, monkeypatch):
        monkeypatch.delenv("OMNIBIOAI_REGISTRY_URL", raising=False)
        from omnibioai_model_registry.plugin_client import PluginClient

        client = PluginClient()
        assert "localhost" in client._url or "8000" in client._url

    def test_init_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("OMNIBIOAI_REGISTRY_URL", "http://myregistry:7000")
        from omnibioai_model_registry.plugin_client import PluginClient

        client = PluginClient()
        assert client._url == "http://myregistry:7000"

    # ── start ──────────────────────────────────────────────────────────────

    def test_start_creates_run_id(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            run_id = client.start(task="t", model_name="m")

        assert isinstance(run_id, str)
        assert len(run_id) > 0
        assert client._run_id == run_id

    def test_start_posts_to_runs_start(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            client.start(task="t", model_name="m")

        assert len(posted) == 1
        assert posted[0]["task"] == "t"
        assert posted[0]["model_name"] == "m"

    # ── log_param / log_params ─────────────────────────────────────────────

    def test_log_param_posts_correct_payload(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            client.log_param("lr", 0.001)

        assert len(posted) == 1
        assert posted[0]["key"] == "lr"
        assert posted[0]["value"] == 0.001
        assert posted[0]["run_id"] == "run123"

    def test_log_params_posts_each_param(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            client.log_params({"lr": 0.001, "epochs": 50})

        assert len(posted) == 2
        keys = {p["key"] for p in posted}
        assert keys == {"lr", "epochs"}

    # ── log_metric / log_metrics ───────────────────────────────────────────

    def test_log_metric_posts_correct_payload(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            client.log_metric("accuracy", 0.95, step=1)

        assert len(posted) == 1
        assert posted[0]["key"] == "accuracy"
        assert posted[0]["value"] == 0.95
        assert posted[0]["step"] == 1

    def test_log_metrics_posts_each_metric(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            client.log_metrics({"acc": 0.9, "f1": 0.85}, step=0)

        assert len(posted) == 2
        keys = {p["key"] for p in posted}
        assert keys == {"acc", "f1"}
        for p in posted:
            assert p["step"] == 0

    # ── set_tag ────────────────────────────────────────────────────────────

    def test_set_tag_posts_correct_payload(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            client.set_tag("team", "bioml")

        assert len(posted) == 1
        assert posted[0]["key"] == "team"
        assert posted[0]["value"] == "bioml"

    # ── finish ─────────────────────────────────────────────────────────────

    def test_finish_returns_run_id(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            result = client.finish()

        assert result == "run123"

    def test_finish_with_no_run_id_returns_none(self):
        from omnibioai_model_registry.plugin_client import PluginClient

        client = PluginClient(registry_url="http://test")
        assert client.finish() is None

    # ── context manager ────────────────────────────────────────────────────

    def test_context_manager_calls_finish_on_exit(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        posted: list = []
        with patch("urllib.request.urlopen", side_effect=self._fake_urlopen(posted)):
            with PluginClient(registry_url="http://test") as client:
                client._run_id = "run123"
                client.set_tag("k", "v")

        run_ids = [p.get("run_id") for p in posted]
        assert "run123" in run_ids

    # ── _post error handling ───────────────────────────────────────────────

    def test_post_handles_http_error_gracefully(self):
        import urllib.error
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        def bad_urlopen(req, **kwargs):
            raise urllib.error.HTTPError(
                url="http://test/v1/runs/log_param",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=None,
            )

        with patch("urllib.request.urlopen", side_effect=bad_urlopen):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            result = client._post("/v1/runs/log_param", {"run_id": "run123"})

        assert result is not None
        assert result.get("code") == 500

    def test_post_handles_connection_error_gracefully(self):
        from unittest.mock import patch

        from omnibioai_model_registry.plugin_client import PluginClient

        def bad_urlopen(req, **kwargs):
            raise ConnectionError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=bad_urlopen):
            client = PluginClient(registry_url="http://test")
            client._run_id = "run123"
            result = client._post("/v1/runs/log_param", {"run_id": "run123"})

        assert result is None


# ============================================================
# auth.py — TestAuth
# ============================================================


class TestAuth:

    def test_require_auth_returns_system_when_disabled(self, monkeypatch):
        import asyncio
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        from omnibioai_model_registry.auth import require_auth
        actor = asyncio.run(require_auth(authorization=None))
        assert actor == "system"

    def test_extract_token_raises_on_missing_header(self):
        from omnibioai_model_registry.auth import AuthError, extract_token
        with pytest.raises(AuthError) as exc_info:
            extract_token(None)
        assert exc_info.value.status_code == 401

    def test_extract_token_raises_on_non_bearer_header(self):
        from omnibioai_model_registry.auth import AuthError, extract_token
        with pytest.raises(AuthError) as exc_info:
            extract_token("Basic abc123")
        assert exc_info.value.status_code == 401

    def test_extract_token_returns_token_on_valid_bearer(self):
        from omnibioai_model_registry.auth import extract_token
        token = extract_token("Bearer mytoken123")
        assert token == "mytoken123"

    # validate_token()/get_actor() (local HS256-only decode + raw-payload
    # field extraction) were removed by the Model Registry IAM Integration
    # PR -- JWT verification is now centralized via AsyncIAMClient
    # (see TestVerifyAndAuthorize/TestActorIdentifier below), and
    # _actor_identifier() reads a verified UserContext, not a raw dict.

    def test_actor_identifier_prefers_email(self):
        from iam_client.models import UserContext
        from omnibioai_model_registry.auth import _actor_identifier
        user = UserContext(
            user_id="42", email="user@example.com", roles=[], permissions=["model.use"], valid=True,
        )
        assert _actor_identifier(user) == "user@example.com"

    def test_actor_identifier_falls_back_to_user_id(self):
        from iam_client.models import UserContext
        from omnibioai_model_registry.auth import _actor_identifier
        user = UserContext(user_id="42", email="", roles=[], permissions=[], valid=True)
        assert _actor_identifier(user) == "42"


class TestVerifyAndAuthorize:
    """Model Registry IAM Integration: verify_and_authorize() -- centralized
    IAM authentication (via AsyncIAMClient, no local JWT decoding) plus
    model.use permission enforcement and model_access audit events."""

    @pytest.fixture(autouse=True)
    def _registry_root(self, monkeypatch):
        # load_config() (called inside verify_and_authorize) requires this
        # regardless of auth -- matches every other test class's setup.
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    def _mock_audit(self, monkeypatch):
        from unittest.mock import MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_audit = MagicMock()
        monkeypatch.setattr(auth_mod, "AuditClient", MagicMock(return_value=mock_audit))
        return mock_audit

    def test_valid_jwt_with_model_use_is_allowed(self, monkeypatch):
        import asyncio
        from iam_client.models import UserContext
        from omnibioai_model_registry.auth import verify_and_authorize

        user = UserContext(
            user_id="1", email="user@test.com", roles=["member"],
            permissions=["model.use"], valid=True, org_id="org-1",
        )
        self._mock_iam_client(monkeypatch, user)
        audit = self._mock_audit(monkeypatch)

        result = asyncio.run(verify_and_authorize("sometoken", action="model_access"))

        assert result is user
        audit.log_event.assert_called_once()
        call = audit.log_event.call_args
        assert call.args[0] == "model_access_success"
        assert call.kwargs["metadata"]["organization_id"] == "org-1"

    def test_valid_jwt_without_model_use_is_denied(self, monkeypatch):
        import asyncio
        from fastapi import HTTPException
        from iam_client.models import UserContext
        from omnibioai_model_registry.auth import AuthError, verify_and_authorize

        user = UserContext(
            user_id="2", email="nomodel@test.com", roles=["member"],
            permissions=["dataset.read"], valid=True, org_id="org-2",
        )
        self._mock_iam_client(monkeypatch, user)
        audit = self._mock_audit(monkeypatch)

        with pytest.raises(AuthError) as exc_info:
            asyncio.run(verify_and_authorize("sometoken", action="model_access"))
        assert exc_info.value.status_code == 403

        audit.log_event.assert_called_once()
        call = audit.log_event.call_args
        assert call.args[0] == "model_access_denied"
        assert call.kwargs["metadata"]["reason"] == "missing_permission"
        assert call.kwargs["metadata"]["organization_id"] == "org-2"

    def test_invalid_jwt_is_denied(self, monkeypatch):
        from omnibioai_model_registry.auth import AuthError, verify_and_authorize
        import asyncio

        self._mock_iam_client(monkeypatch, None)  # get_user() returns None: unverifiable token
        audit = self._mock_audit(monkeypatch)

        with pytest.raises(AuthError) as exc_info:
            asyncio.run(verify_and_authorize("badtoken", action="model_access"))
        assert exc_info.value.status_code == 401
        audit.log_event.assert_called_once_with(
            "model_access_denied", "unknown", "model_access", metadata={"reason": "invalid_token"},
        )

    def test_expired_jwt_is_denied(self, monkeypatch):
        """AsyncIAMClient.get_user() returns None for an expired token (its
        own local-decode step raises ExpiredSignatureError, caught
        internally) -- indistinguishable from any other invalid token at
        this layer, which is the correct fail-closed behavior: this
        service never re-derives *why* a token failed from a raw payload,
        it only ever trusts the IAM client's verified verdict."""
        from omnibioai_model_registry.auth import AuthError, verify_and_authorize
        import asyncio

        self._mock_iam_client(monkeypatch, None)

        with pytest.raises(AuthError) as exc_info:
            asyncio.run(verify_and_authorize("expiredtoken", action="model_access"))
        assert exc_info.value.status_code == 401

    def test_iam_client_exception_is_denied_not_raised(self, monkeypatch):
        """A network/connection failure talking to the IAM/auth service
        must fail closed (401), never propagate as an unhandled 500."""
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod
        from omnibioai_model_registry.auth import AuthError, verify_and_authorize
        import asyncio

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(side_effect=ConnectionError("auth service unreachable"))
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        self._mock_audit(monkeypatch)

        with pytest.raises(AuthError) as exc_info:
            asyncio.run(verify_and_authorize("sometoken", action="model_access"))
        assert exc_info.value.status_code == 401

    def test_organization_id_is_captured_from_identity(self, monkeypatch):
        """Organization isolation (this PR's confirmed scope): org_id is
        extracted from the verified identity and reaches the audit event
        -- it is not used to filter registry data (no tenant concept
        exists in the data model yet; see auth.py's module docstring)."""
        import asyncio
        from iam_client.models import UserContext
        from omnibioai_model_registry.auth import verify_and_authorize

        user_org_a = UserContext(
            user_id="1", email="a@test.com", roles=[], permissions=["model.use"],
            valid=True, org_id="org-a",
        )
        self._mock_iam_client(monkeypatch, user_org_a)
        audit = self._mock_audit(monkeypatch)

        result = asyncio.run(verify_and_authorize("token-a", action="model_access"))
        assert result.org_id == "org-a"
        assert audit.log_event.call_args.kwargs["metadata"]["organization_id"] == "org-a"

    def test_organization_id_none_for_identity_with_no_org(self, monkeypatch):
        """A valid identity with no org membership yet (org_id=None) is a
        legitimate state, not an error -- must still be granted access
        (model.use is a global concept here, not org-scoped in this
        service) and logged with organization_id=None, not crash."""
        import asyncio
        from iam_client.models import UserContext
        from omnibioai_model_registry.auth import verify_and_authorize

        user_no_org = UserContext(
            user_id="3", email="noorg@test.com", roles=[], permissions=["model.use"],
            valid=True, org_id=None,
        )
        self._mock_iam_client(monkeypatch, user_no_org)
        audit = self._mock_audit(monkeypatch)

        result = asyncio.run(verify_and_authorize("token-noorg", action="model_access"))
        assert result.org_id is None
        assert audit.log_event.call_args.kwargs["metadata"]["organization_id"] is None


# ============================================================
# audit_client.py — TestAuditClient
# ============================================================


class TestAuditClient:

    def test_log_event_with_empty_audit_url_does_nothing(self):
        from omnibioai_model_registry.audit_client import AuditClient
        client = AuditClient("")
        # Should complete silently without raising or spawning a thread
        client.log_event("register_model", "system", "t/m@v1")

    def test_log_event_fires_http_post_in_background_thread(self):
        from unittest.mock import MagicMock, patch
        from omnibioai_model_registry.audit_client import AuditClient

        client = AuditClient("http://test-audit:8004")
        mock_thread = MagicMock()

        with patch("omnibioai_model_registry.audit_client.threading.Thread", return_value=mock_thread) as mock_cls:
            client.log_event("register_model", "actor@test", "t/m@v1",
                             task="t", model_name="m", version="v1")
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs.get("daemon") is True
            mock_thread.start.assert_called_once()

    def test_send_swallows_connection_errors_silently(self):
        from unittest.mock import patch
        from omnibioai_model_registry.audit_client import AuditClient

        def bad_urlopen(req, timeout=None):
            raise ConnectionError("Connection refused")

        client = AuditClient("http://test-audit:8004")
        with patch("urllib.request.urlopen", side_effect=bad_urlopen):
            # Must not raise
            client._send({"action": "test", "actor": "system"})

    def test_payload_shape_matches_spec(self):
        from unittest.mock import MagicMock, patch
        from omnibioai_model_registry.audit_client import AuditClient

        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(json.loads(req.data))
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        client = AuditClient("http://test-audit:8004")
        payload = {
            "service": "model-registry",
            "action": "register_model",
            "actor": "user@test",
            "resource": "t/m@v1",
            "task": "t",
            "model_name": "m",
            "version": "v1",
            "ts_utc": "2026-06-15T00:00:00+00:00",
            "metadata": {"env": "prod"},
        }
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client._send(payload)

        assert len(captured) == 1
        p = captured[0]
        for key in ("service", "action", "actor", "resource", "ts_utc"):
            assert key in p, f"missing key: {key}"
        assert p["service"] == "model-registry"
        assert p["action"] == "register_model"
        assert p["actor"] == "user@test"
        assert p["resource"] == "t/m@v1"
        assert p["metadata"] == {"env": "prod"}


# ============================================================
# run.py — coverage gaps
# ============================================================


class TestRunLoggerCoverageGaps:

    def test_run_logger_without_registry_root_uses_env(self, env_root):
        """Covers run.py _resolve_registry_root(None) → load_config() path."""
        from omnibioai_model_registry.run import RunLogger
        r = RunLogger(task="t", model_name="m")
        assert isinstance(r.run_id, str)
        r.log_param("lr", 0.001)

    def test_run_logger_write_json_cleanup_on_replace_failure(self, tmp_path, monkeypatch):
        """Covers run.py lines 33-35: os.unlink(tmp) called when replace fails."""
        import omnibioai_model_registry.run as run_mod
        from omnibioai_model_registry.run import RunLogger

        replace_calls = []

        orig_replace = run_mod.os.replace

        def bad_replace(src, dst):
            replace_calls.append(src)
            raise OSError("simulated replace failure")

        monkeypatch.setattr(run_mod.os, "replace", bad_replace)
        r = RunLogger(task="t", model_name="m", registry_root=tmp_path)
        with pytest.raises(OSError):
            r.log_param("x", 1)


# ============================================================
# service/app/main.py — comprehensive route coverage
# ============================================================


@pytest.fixture
def full_svc_client(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    import omnibioai_model_registry.service.app.main as _svc
    from fastapi.testclient import TestClient
    new_reg = _svc.ModelRegistry.from_env()
    monkeypatch.setattr(_svc, "registry", new_reg)
    return TestClient(_svc.app, raise_server_exceptions=False), new_reg.root


class TestServiceAllRoutes:
    """Exercise previously untested routes for coverage."""

    def _register(self, root, tmp_path, task="t", model="m", version="v1"):
        src = tmp_path / f"src_{task}_{model}_{version}"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task=task, model_name=model, version=version,
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        return src

    def test_health_endpoint(self, full_svc_client):
        client, _ = full_svc_client
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert "version" in r.json()

    def test_register_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        r = client.post("/v1/register", json={
            "task": "t", "model_name": "m", "version": "v1",
            "artifacts_dir": str(src), "metadata": {},
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["version"] == "v1"

    def test_register_endpoint_emits_model_registered(self, full_svc_client, tmp_path, monkeypatch):
        """PR14.2B-3: auth_enabled=False (this fixture's default) means
        require_write_auth_with_context returns the synthetic 'system'
        UserContext with org_id=None -- emission is skipped, not called
        with a fabricated org. Covered separately below with a real org_id."""
        from unittest.mock import patch
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_emit_usage_safe") as mock_emit_safe:
            r = client.post("/v1/register", json={
                "task": "t", "model_name": "m", "version": "v1",
                "artifacts_dir": str(src), "metadata": {},
            })
        assert r.status_code == 200
        mock_emit_safe.assert_called_once()
        args, kwargs = mock_emit_safe.call_args
        assert args[0] is _svc.emit_model_registered
        assert kwargs["organization_id"] is None

    def test_register_endpoint_emits_with_real_org_id_when_auth_enabled(self, full_svc_client, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch
        from iam_client.models import UserContext
        import omnibioai_model_registry.auth as auth_mod
        import omnibioai_model_registry.service.app.main as _svc

        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=UserContext(
            user_id="9", email="u@test.com", roles=[], permissions=["model.use"],
            valid=True, org_id="123",
        ))
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))

        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        with patch.object(_svc, "_emit_usage_safe") as mock_emit_safe:
            r = client.post(
                "/v1/register",
                json={"task": "t", "model_name": "m", "version": "v1",
                      "artifacts_dir": str(src), "metadata": {}},
                headers={"Authorization": "Bearer sometoken"},
            )
        assert r.status_code == 200
        mock_emit_safe.assert_called_once_with(
            _svc.emit_model_registered, organization_id="123", user_id="9",
        )

    def test_register_endpoint_emission_exception_does_not_affect_response(self, full_svc_client, tmp_path):
        """Fail-open: an unexpected exception from emit_model_registered
        must not affect the /register response -- _emit_usage_safe is
        the mechanism, tested for real here (not mocked away)."""
        from unittest.mock import patch
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "emit_model_registered", side_effect=RuntimeError("boom")):
            r = client.post("/v1/register", json={
                "task": "t", "model_name": "m", "version": "v1",
                "artifacts_dir": str(src), "metadata": {},
            })
        assert r.status_code == 200

    def test_other_write_routes_unaffected_by_context_change(self, full_svc_client, tmp_path):
        """promote/set_tag/patch_version/set_stage still depend on
        require_write_auth (bare str), not require_write_auth_with_context
        -- unaffected by this PR."""
        client, root = full_svc_client
        src = self._register(root, tmp_path)
        r = client.post("/v1/promote", json={
            "task": "t", "model_name": "m", "version": "v1", "alias": "prod",
        })
        assert r.status_code == 200

    def test_register_endpoint_error(self, full_svc_client):
        client, _ = full_svc_client
        r = client.post("/v1/register", json={
            "task": "t", "model_name": "m", "version": "v1",
            "artifacts_dir": "/nonexistent/path", "metadata": {},
        })
        assert r.status_code == 400

    def test_promote_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.post("/v1/promote", json={
            "task": "t", "model_name": "m", "alias": "staging", "version": "v1",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_promote_endpoint_error(self, full_svc_client):
        client, _ = full_svc_client
        r = client.post("/v1/promote", json={
            "task": "t", "model_name": "m", "alias": "prod", "version": "nonexistent",
        })
        assert r.status_code == 400

    def test_resolve_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.get("/v1/resolve", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert "path" in r.json()

    def test_resolve_endpoint_missing(self, full_svc_client):
        client, _ = full_svc_client
        r = client.get("/v1/resolve", params={"task": "t", "ref": "m@missing"})
        assert r.status_code == 400

    def test_verify_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.post("/v1/verify", json={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_show_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.get("/v1/show", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "meta" in data
        assert "package_dir" in data

    def test_show_endpoint_missing_meta(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        meta_path = root / "tasks" / "t" / "models" / "m" / "versions" / "v1" / "model_meta.json"
        meta_path.unlink()
        r = client.get("/v1/show", params={"task": "t", "ref": "m@v1"})
        # HTTPException(404) raised inside try-except gets caught and re-wrapped as 500
        assert r.status_code == 500

    def test_metrics_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.get("/v1/metrics", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "version_metrics" in data
        assert "run_history" in data
        assert data["version_metrics"]["acc"] == 0.9

    def test_artifacts_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.get("/v1/artifacts", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert isinstance(data["files"], list)
        assert len(data["files"]) > 0

    def test_runs_log_metric_returns_503_without_db(self, full_svc_client):
        client, _ = full_svc_client
        r = client.post("/v1/runs/log-metric", json={
            "task": "t", "model_name": "m", "run_id": "r1",
            "key": "accuracy", "value": 0.95, "step": 0,
        })
        assert r.status_code == 503

    def test_runs_log_param_returns_503_without_db(self, full_svc_client):
        client, _ = full_svc_client
        r = client.post("/v1/runs/log-param", json={
            "task": "t", "model_name": "m", "run_id": "r1",
            "key": "lr", "value": 0.001,
        })
        assert r.status_code == 503

    def test_runs_log_batch_returns_503_without_db(self, full_svc_client):
        client, _ = full_svc_client
        r = client.post("/v1/runs/log-batch", json={
            "task": "t", "model_name": "m", "run_id": "r1",
            "metrics": [], "params": {}, "tags": {},
        })
        assert r.status_code == 503

    def test_runs_get_returns_503_without_db(self, full_svc_client):
        client, _ = full_svc_client
        r = client.get("/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"})
        assert r.status_code == 503

    def test_runs_list_returns_503_without_db(self, full_svc_client):
        client, _ = full_svc_client
        r = client.get("/v1/runs/list", params={"task": "t", "model": "m"})
        assert r.status_code == 503

    def test_tags_endpoint_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.put("/v1/tags", json={
            "task": "t", "model_name": "m", "version": "v1",
            "key": "team", "value": "bioml",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True
        meta_path = root / "tasks" / "t" / "models" / "m" / "versions" / "v1" / "model_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["tags"]["team"] == "bioml"

    def test_versions_patch_success(self, full_svc_client, tmp_path):
        client, root = full_svc_client
        self._register(root, tmp_path)
        r = client.post("/v1/versions/patch", json={
            "task": "t", "model_name": "m", "version": "v1",
            "description": "Updated description",
            "tags": {"env": "prod"},
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_versions_patch_not_found(self, full_svc_client):
        client, _ = full_svc_client
        r = client.post("/v1/versions/patch", json={
            "task": "t", "model_name": "nonexistent", "version": "v99",
        })
        assert r.status_code == 404

    def test_auth_status_open_mode(self, full_svc_client, monkeypatch):
        client, _ = full_svc_client
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        r = client.get("/v1/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert data["auth_enabled"] is False
        assert data["mode"] == "open"
        assert data["iam_url"] is None

    def test_auth_status_jwt_mode(self, full_svc_client, monkeypatch):
        client, _ = full_svc_client
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("IAM_URL", "http://auth-service:8001")
        r = client.get("/v1/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert data["auth_enabled"] is True
        assert data["mode"] == "jwt"
        assert data["iam_url"] == "http://auth-service:8001"

    def test_register_then_promote_then_resolve_alias(self, full_svc_client, tmp_path):
        """End-to-end: register → promote → resolve alias."""
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        client.post("/v1/register", json={
            "task": "t", "model_name": "m", "version": "v1",
            "artifacts_dir": str(src), "metadata": {},
        })
        client.post("/v1/promote", json={
            "task": "t", "model_name": "m", "alias": "latest", "version": "v1",
        })
        r = client.get("/v1/resolve", params={"task": "t", "ref": "m@latest"})
        assert r.status_code == 200
        assert "v1" in r.json()["path"]

    def test_startup_handler_auth_disabled(self, monkeypatch, tmp_path):
        """Cover startup handler: auth_enabled=False branch."""
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(tmp_path))
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        import omnibioai_model_registry.service.app.main as _svc
        _svc._startup()  # must not raise

    def test_startup_handler_auth_enabled(self, monkeypatch, tmp_path):
        """Cover startup handler: auth_enabled=True branch."""
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(tmp_path))
        monkeypatch.setenv("AUTH_ENABLED", "true")
        import omnibioai_model_registry.service.app.main as _svc
        _svc._startup()  # must not raise

    def test_startup_handler_load_config_exception(self, monkeypatch):
        """Cover startup handler except Exception: pass (line ~72)."""
        monkeypatch.delenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", raising=False)
        monkeypatch.delenv("REGISTRY_ROOT", raising=False)
        import omnibioai_model_registry.service.app.main as _svc
        _svc._startup()  # load_config raises → caught silently

    def test_metrics_endpoint_with_run_id_filesystem_fallback(self, full_svc_client, tmp_path, monkeypatch):
        """Cover metrics endpoint filesystem fallback for run history (lines ~543-579)."""
        monkeypatch.delenv("DB_HOST", raising=False)
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src,
            metadata={"lineage": {"run_id": "run123"}},
            set_alias=None,
        )
        run_dir = root / "tasks" / "t" / "models" / "m" / "runs" / "run123" / "metrics"
        run_dir.mkdir(parents=True)
        (run_dir / "accuracy.jsonl").write_text(
            json.dumps({"key": "accuracy", "value": 0.95, "step": 0, "ts_utc": "2026-01-01T00:00:00Z"}) + "\n"
            + "\n"  # blank line to test skip
            + "invalid json line\n"  # to test exception handling
        )
        r = client.get("/v1/metrics", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "accuracy" in data["run_history"]

    def test_runs_log_metric_with_mocked_db(self, full_svc_client, monkeypatch):
        """Cover runs/log-metric handler body with mocked DB."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        mock_conn = MagicMock()
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.log_metric"):
                r = client.post("/v1/runs/log-metric", json={
                    "task": "t", "model_name": "m", "run_id": "r1",
                    "key": "accuracy", "value": 0.95, "step": 0,
                })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_runs_log_param_with_mocked_db(self, full_svc_client):
        """Cover runs/log-param handler body with mocked DB."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        mock_conn = MagicMock()
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.log_param"):
                r = client.post("/v1/runs/log-param", json={
                    "task": "t", "model_name": "m", "run_id": "r1",
                    "key": "lr", "value": 0.001,
                })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_runs_log_batch_with_mocked_db(self, full_svc_client):
        """Cover runs/log-batch handler body with mocked DB."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        mock_conn = MagicMock()
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.log_metric"):
                with patch("omnibioai_model_registry.tracking.log_params"):
                    with patch("omnibioai_model_registry.tracking.set_tags"):
                        r = client.post("/v1/runs/log-batch", json={
                            "task": "t", "model_name": "m", "run_id": "r1",
                            "metrics": [{"key": "acc", "value": 0.9, "step": 0}],
                            "params": {"lr": 0.001},
                            "tags": {"team": "bioml"},
                        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_runs_get_with_mocked_db(self, full_svc_client):
        """Cover runs/get handler body with mocked DB."""
        from datetime import datetime, timezone
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        mock_conn = MagicMock()
        run_data = {
            "run_id": "r1", "task": "t", "model_name": "m",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "actor": "system",
            "params": {}, "tags": {}, "metrics_summary": {},
        }
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.get_run", return_value=run_data):
                with patch("omnibioai_model_registry.tracking.get_metric_history", return_value=[]):
                    r = client.get("/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_runs_list_with_mocked_db(self, full_svc_client):
        """Cover runs/list handler body with mocked DB."""
        from datetime import datetime, timezone
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        mock_conn = MagicMock()
        run_list = [{"run_id": "r1", "status": "running"}]
        run_detail = {
            "run_id": "r1", "task": "t", "model_name": "m",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "actor": None,
            "params": {}, "tags": {}, "metrics_summary": {},
        }
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.list_runs", return_value=run_list):
                with patch("omnibioai_model_registry.tracking.get_run", return_value=run_detail):
                    r = client.get("/v1/runs/list", params={"task": "t", "model": "m"})
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestUsageEmit:
    """PR14.2B-3: unit tests for usage_emit.py itself, isolated from the
    FastAPI route layer."""

    def test_client_constructs_a_real_usage_client(self):
        from omnibioai_model_registry.usage_emit import _client
        from usage_client import UsageClient

        assert isinstance(_client(), UsageClient)

    def test_emits_correct_fields(self):
        from unittest.mock import MagicMock, patch
        from omnibioai_model_registry.usage_emit import emit_model_registered

        mock_client = MagicMock()
        with patch("omnibioai_model_registry.usage_emit._client", return_value=mock_client):
            emit_model_registered(organization_id="77", user_id="9", trace_id="trace-1")

        mock_client.emit_usage_event.assert_called_once_with(
            organization_id="77",
            service="model.registry",
            resource="model.register",
            action="registered",
            quantity=1,
            unit="count",
            user_id="9",
            trace_id="trace-1",
        )

    def test_skips_emission_when_organization_id_none(self):
        from unittest.mock import MagicMock, patch
        from omnibioai_model_registry.usage_emit import emit_model_registered

        mock_client = MagicMock()
        with patch("omnibioai_model_registry.usage_emit._client", return_value=mock_client):
            emit_model_registered(organization_id=None)

        mock_client.emit_usage_event.assert_not_called()

    def test_client_exception_is_swallowed(self):
        from unittest.mock import MagicMock, patch
        from omnibioai_model_registry.usage_emit import emit_model_registered

        mock_client = MagicMock()
        mock_client.emit_usage_event.side_effect = RuntimeError("boom")
        with patch("omnibioai_model_registry.usage_emit._client", return_value=mock_client):
            emit_model_registered(organization_id="77")  # must not raise

    def test_client_construction_failure_is_swallowed(self):
        from unittest.mock import patch
        from omnibioai_model_registry.usage_emit import emit_model_registered

        with patch("omnibioai_model_registry.usage_emit._client", side_effect=RuntimeError("boom")):
            emit_model_registered(organization_id="77")  # must not raise

    # -----------------------------------------------------------------
    # PR14.3-4: structured-logging observability (no metrics dependency)
    # -----------------------------------------------------------------

    def _states(self, caplog):
        return [
            r.usage_event_state for r in caplog.records
            if hasattr(r, "usage_event_state")
        ]

    def test_success_logs_attempted_then_succeeded(self, caplog):
        from unittest.mock import MagicMock, patch
        from omnibioai_model_registry.usage_emit import emit_model_registered

        mock_client = MagicMock()
        with caplog.at_level("INFO", logger="omnibioai_model_registry.usage_emit"):
            with patch("omnibioai_model_registry.usage_emit._client", return_value=mock_client):
                emit_model_registered(organization_id="77")

        assert self._states(caplog) == ["attempted", "succeeded"]

    def test_missing_org_logs_attempted_then_skipped(self, caplog):
        from omnibioai_model_registry.usage_emit import emit_model_registered

        with caplog.at_level("INFO", logger="omnibioai_model_registry.usage_emit"):
            emit_model_registered(organization_id=None)

        assert self._states(caplog) == ["attempted", "skipped_missing_org"]

    def test_exception_logs_attempted_then_failed(self, caplog):
        from unittest.mock import MagicMock, patch
        from omnibioai_model_registry.usage_emit import emit_model_registered

        mock_client = MagicMock()
        mock_client.emit_usage_event.side_effect = RuntimeError("boom")
        with caplog.at_level("INFO", logger="omnibioai_model_registry.usage_emit"):
            with patch("omnibioai_model_registry.usage_emit._client", return_value=mock_client):
                emit_model_registered(organization_id="77")  # must not raise

        assert self._states(caplog) == ["attempted", "failed_exception"]


class TestAuthRequireWriteAuth:
    """Cover auth.py require_auth/require_write_auth -- centralized IAM
    verification (AsyncIAMClient) + model.use enforcement, not local JWT
    decoding. Mocks AsyncIAMClient so no real network/Redis call is made."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    def test_require_write_auth_disabled_returns_system(self, monkeypatch):
        import asyncio
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        from omnibioai_model_registry.auth import require_write_auth
        actor = asyncio.run(require_write_auth(authorization=None))
        assert actor == "system"

    def test_require_write_auth_enabled_valid_jwt_with_model_use_is_allowed(self, monkeypatch):
        import asyncio
        from iam_client.models import UserContext
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=["model.use"], valid=True,
        ))
        from omnibioai_model_registry.auth import require_write_auth
        actor = asyncio.run(require_write_auth(authorization="Bearer sometoken"))
        assert actor == "user@test.com"

    def test_require_auth_enabled_valid_jwt_without_model_use_raises_403(self, monkeypatch):
        import asyncio
        from fastapi import HTTPException
        from iam_client.models import UserContext
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=[], valid=True,
        ))
        from omnibioai_model_registry.auth import require_auth
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_auth(authorization="Bearer sometoken"))
        assert exc_info.value.status_code == 403

    def test_require_auth_enabled_invalid_token_raises_401(self, monkeypatch):
        import asyncio
        from fastapi import HTTPException
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        self._mock_iam_client(monkeypatch, None)
        from omnibioai_model_registry.auth import require_auth
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_auth(authorization="Bearer invalidtoken"))
        assert exc_info.value.status_code == 401

    def test_require_auth_enabled_expired_token_raises_401(self, monkeypatch):
        """AsyncIAMClient.get_user() returns None for an expired token --
        this layer never re-derives the reason locally, it only trusts
        the IAM client's verdict (see TestVerifyAndAuthorize's identical
        assertion for the underlying implementation)."""
        import asyncio
        from fastapi import HTTPException
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        self._mock_iam_client(monkeypatch, None)
        from omnibioai_model_registry.auth import require_auth
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_auth(authorization="Bearer expiredtoken"))
        assert exc_info.value.status_code == 401

    def test_require_auth_enabled_missing_header_raises_401(self, monkeypatch):
        import asyncio
        from fastapi import HTTPException
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        from omnibioai_model_registry.auth import require_auth
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_auth(authorization=None))
        assert exc_info.value.status_code == 401


class TestRequireWriteAuthWithContext:
    """PR14.2B-3: require_write_auth_with_context -- same enforcement as
    require_write_auth, but returns the full UserContext (for
    organization_id) instead of just the actor string. Additive: does
    not change require_write_auth's own contract, tested separately
    above."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    def test_disabled_returns_synthetic_system_context_with_none_org_id(self, monkeypatch):
        import asyncio
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        from omnibioai_model_registry.auth import require_write_auth_with_context, _actor_identifier
        user = asyncio.run(require_write_auth_with_context(authorization=None))
        assert user.org_id is None
        assert _actor_identifier(user) == "system"

    def test_enabled_valid_jwt_returns_full_context_with_org_id(self, monkeypatch):
        import asyncio
        from iam_client.models import UserContext
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=["model.use"],
            valid=True, org_id="77",
        ))
        from omnibioai_model_registry.auth import require_write_auth_with_context, _actor_identifier
        user = asyncio.run(require_write_auth_with_context(authorization="Bearer sometoken"))
        assert user.org_id == "77"
        assert _actor_identifier(user) == "user@test.com"

    def test_enabled_without_model_use_raises_403(self, monkeypatch):
        import asyncio
        from fastapi import HTTPException
        from iam_client.models import UserContext
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=[], valid=True, org_id="77",
        ))
        from omnibioai_model_registry.auth import require_write_auth_with_context
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_write_auth_with_context(authorization="Bearer sometoken"))
        assert exc_info.value.status_code == 403

    def test_enabled_invalid_token_raises_401(self, monkeypatch):
        import asyncio
        from fastapi import HTTPException
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")
        self._mock_iam_client(monkeypatch, None)
        from omnibioai_model_registry.auth import require_write_auth_with_context
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(require_write_auth_with_context(authorization="Bearer invalidtoken"))
        assert exc_info.value.status_code == 401

    def test_other_write_auth_routes_unaffected(self, monkeypatch):
        """require_write_auth itself (used by promote/set_tag/patch_version/
        set_stage) still returns a bare str -- this dependency's addition
        must not change that contract."""
        import asyncio
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "/tmp/reg")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        from omnibioai_model_registry.auth import require_write_auth
        actor = asyncio.run(require_write_auth(authorization=None))
        assert actor == "system"
        assert isinstance(actor, str)


# ============================================================
# Targeted gap-closure tests for remaining uncovered statements
# ============================================================


class TestServiceCoverageGaps:
    """Close the remaining ~50 uncovered statements in service/app/main.py."""

    # ── _startup() variations ────────────────────────────────────────────────

    def test_startup_db_none_early_return(self, monkeypatch, tmp_path):
        """Cover line 72: early return when _db is None."""
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(tmp_path))
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        import omnibioai_model_registry.service.app.main as _svc
        orig_db = _svc._db
        try:
            _svc._db = None
            _svc._startup()  # hits line 71 (if _db is None: return)
        finally:
            _svc._db = orig_db

    def test_startup_mock_db_success(self, monkeypatch, tmp_path):
        """Cover lines 75-77: successful DB init in startup."""
        from unittest.mock import MagicMock
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(tmp_path))
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        import omnibioai_model_registry.service.app.main as _svc
        mock_conn = MagicMock()
        mock_db = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        orig_db = _svc._db
        try:
            _svc._db = mock_db
            _svc._startup()  # hits init_tables + conn.close + log.info
        finally:
            _svc._db = orig_db

    # ── _get_db_conn() variations ────────────────────────────────────────────

    def test_get_db_conn_when_db_module_is_none(self, full_svc_client, monkeypatch):
        """Cover line 259: HTTPException 503 when _db is None."""
        from unittest.mock import patch
        client, _ = full_svc_client
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_db", None):
            r = client.get("/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"})
        assert r.status_code == 503

    def test_get_db_conn_generic_exception(self, full_svc_client, monkeypatch):
        """Cover lines 264-265: HTTPException 503 on generic DB error."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        import omnibioai_model_registry.service.app.main as _svc
        mock_db = MagicMock()
        mock_db.get_connection.side_effect = RuntimeError("pool exhausted")
        with patch.object(_svc, "_db", mock_db):
            r = client.get("/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"})
        assert r.status_code == 503

    # ── api_verify error path ────────────────────────────────────────────────

    def test_verify_nonexistent_model_returns_error(self, full_svc_client):
        """Cover lines 339-340: api_verify exception path."""
        client, _ = full_svc_client
        r = client.post("/v1/verify", json={"task": "t", "ref": "m@nonexistent_ver"})
        assert r.status_code in (400, 404, 500)

    # ── list_models filter edge cases ─────────────────────────────────────────

    def test_list_models_task_filter_skips_wrong_task(self, full_svc_client):
        """Cover line 396: task filter continue."""
        client, root = full_svc_client
        for task, ver in [("task_a", "v1"), ("task_b", "v1")]:
            vdir = root / "tasks" / task / "models" / "m" / "versions" / ver
            vdir.mkdir(parents=True)
            (vdir / "model_meta.json").write_text(
                json.dumps({"task": task, "model_name": "m", "version": ver})
            )
            _write_unowned_ownership(root, task, "m")
        r = client.get("/v1/models", params={"task": "task_a"})
        assert r.status_code == 200
        tasks = [m["task"] for m in r.json()]
        assert all(t == "task_a" for t in tasks)
        assert len(tasks) == 1

    def test_list_models_model_name_filter_skips_wrong_model(self, full_svc_client):
        """Cover line 398: model_name filter continue."""
        client, root = full_svc_client
        for mn, ver in [("model_a", "v1"), ("model_b", "v1")]:
            vdir = root / "tasks" / "t" / "models" / mn / "versions" / ver
            vdir.mkdir(parents=True)
            (vdir / "model_meta.json").write_text(
                json.dumps({"task": "t", "model_name": mn, "version": ver})
            )
            _write_unowned_ownership(root, "t", mn)
        r = client.get("/v1/models", params={"model_name": "model_a"})
        assert r.status_code == 200
        names = [m["model_name"] for m in r.json()]
        assert names == ["model_a"]

    def test_list_models_metric_gte_skips_model_without_metrics_file(self, full_svc_client):
        """Cover line 403: continue when metrics file absent with metric_gte filter."""
        client, root = full_svc_client
        vdir = root / "tasks" / "t" / "models" / "no_metrics" / "versions" / "v1"
        vdir.mkdir(parents=True)
        (vdir / "model_meta.json").write_text(json.dumps({"task": "t", "model_name": "no_metrics"}))
        _write_unowned_ownership(root, "t", "no_metrics")
        r = client.get("/v1/models", params={"metric_gte": "accuracy:0.5"})
        assert r.status_code == 200
        assert r.json() == []

    def test_list_models_metric_gte_skips_bad_metrics_json(self, full_svc_client):
        """Cover lines 406-407: continue when metrics.json is malformed."""
        client, root = full_svc_client
        vdir = root / "tasks" / "t" / "models" / "bad_m" / "versions" / "v1"
        vdir.mkdir(parents=True)
        (vdir / "model_meta.json").write_text(json.dumps({"task": "t", "model_name": "bad_m"}))
        (vdir / "metrics.json").write_text("{{{ invalid json")
        r = client.get("/v1/models", params={"metric_gte": "accuracy:0.5"})
        assert r.status_code == 200
        assert r.json() == []

    def test_list_models_skips_bad_meta_json(self, full_svc_client):
        """Cover lines 413-414: continue when model_meta.json is malformed."""
        client, root = full_svc_client
        vdir = root / "tasks" / "t" / "models" / "bad2" / "versions" / "v1"
        vdir.mkdir(parents=True)
        (vdir / "model_meta.json").write_text("{{{ not json at all")
        r = client.get("/v1/models")
        assert r.status_code == 200
        assert all(m.get("model_name") != "bad2" for m in r.json())

    # ── DB route error paths (after mock conn) ────────────────────────────────

    def test_log_metric_exception_after_conn(self, full_svc_client):
        """Cover lines 436-437: exception in api_log_metric body after DB conn."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        import omnibioai_model_registry.service.app.main as _svc
        mock_conn = MagicMock()
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.log_metric",
                       side_effect=RuntimeError("tracking failed")):
                r = client.post("/v1/runs/log-metric", json={
                    "task": "t", "model_name": "m", "run_id": "r1",
                    "key": "acc", "value": 0.9, "step": 0,
                })
        assert r.status_code == 500

    def test_log_param_exception_after_conn(self, full_svc_client):
        """Cover lines 451-452: exception in api_log_param body after DB conn."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        import omnibioai_model_registry.service.app.main as _svc
        mock_conn = MagicMock()
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.log_param",
                       side_effect=RuntimeError("tracking failed")):
                r = client.post("/v1/runs/log-param", json={
                    "task": "t", "model_name": "m", "run_id": "r1",
                    "key": "lr", "value": 0.001,
                })
        assert r.status_code == 500

    def test_log_batch_exception_after_conn(self, full_svc_client):
        """Cover lines 471-472: exception in api_log_batch body after DB conn."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        import omnibioai_model_registry.service.app.main as _svc
        mock_conn = MagicMock()
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.log_metric",
                       side_effect=RuntimeError("batch failed")):
                r = client.post("/v1/runs/log-batch", json={
                    "task": "t", "model_name": "m", "run_id": "r1",
                    "metrics": [{"key": "acc", "value": 0.9}],
                    "params": {}, "tags": {},
                })
        assert r.status_code == 500

    def test_run_get_exception_after_conn(self, full_svc_client):
        """Cover lines 484, 487-488: exception in api_run_get after DB conn."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        import omnibioai_model_registry.service.app.main as _svc
        mock_conn = MagicMock()
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.get_run",
                       side_effect=RuntimeError("run not found")):
                r = client.get("/v1/runs/get",
                               params={"task": "t", "model": "m", "run_id": "r1"})
        assert r.status_code == 500

    def test_runs_list_exception_after_conn(self, full_svc_client):
        """Cover lines 503-504, 506-507: exception in api_runs_list after DB conn."""
        from unittest.mock import MagicMock, patch
        client, _ = full_svc_client
        import omnibioai_model_registry.service.app.main as _svc
        mock_conn = MagicMock()
        with patch.object(_svc, "_get_db_conn", return_value=mock_conn):
            with patch("omnibioai_model_registry.tracking.list_runs",
                       side_effect=RuntimeError("list failed")):
                r = client.get("/v1/runs/list", params={"task": "t", "model": "m"})
        assert r.status_code == 500

    # ── api_metrics edge cases ────────────────────────────────────────────────

    def test_metrics_endpoint_error_resolve(self, full_svc_client):
        """Cover lines 520-521: error path when resolve_model fails."""
        client, _ = full_svc_client
        r = client.get("/v1/metrics", params={"task": "t", "ref": "m@does_not_exist"})
        assert r.status_code in (400, 404, 500)

    def test_metrics_endpoint_bad_metrics_json(self, full_svc_client, tmp_path):
        """Cover lines 531-532: except Exception when metrics.json is malformed."""
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        vdir = root / "tasks" / "t" / "models" / "m" / "versions" / "v1"
        (vdir / "metrics.json").write_text("{{{malformed")
        r = client.get("/v1/metrics", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        assert r.json()["version_metrics"] == {}

    def test_metrics_endpoint_bad_meta_json(self, full_svc_client, tmp_path):
        """Cover lines 536-537: except Exception when model_meta.json is malformed."""
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        vdir = root / "tasks" / "t" / "models" / "m" / "versions" / "v1"
        (vdir / "model_meta.json").write_text("{{{malformed")
        r = client.get("/v1/metrics", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200

    def test_metrics_with_run_id_and_mock_db_cursor(self, full_svc_client, tmp_path, monkeypatch):
        """Cover lines 545-555: api_metrics DB cursor path."""
        from unittest.mock import MagicMock, patch
        monkeypatch.delenv("DB_HOST", raising=False)
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src,
            metadata={"lineage": {"run_id": "run_abc"}},
            set_alias=None,
        )
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"key_name": "accuracy"}]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = lambda s, *a: False
        mock_db = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        mock_tracking = MagicMock()
        mock_tracking.get_metric_history.return_value = [
            {"value": 0.9, "step": 0, "ts_utc": "2026-01-01T00:00:00"}
        ]
        with patch.object(_svc, "_db", mock_db):
            with patch.object(_svc, "_tracking", mock_tracking):
                r = client.get("/v1/metrics", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True

    # ── api_aliases bad file ──────────────────────────────────────────────────

    def test_aliases_skips_malformed_alias_file(self, full_svc_client):
        """Cover lines 600-601: continue when alias file has bad JSON."""
        client, root = full_svc_client
        _write_unowned_ownership(root, "t", "m")
        aliases_dir = root / "tasks" / "t" / "models" / "m" / "aliases"
        aliases_dir.mkdir(parents=True)
        (aliases_dir / "bad.json").write_text("{{{ invalid")
        r = client.get("/v1/aliases", params={"task": "t", "model": "m"})
        assert r.status_code == 200
        assert r.json()["aliases"] == []

    # ── api_set_tag DB best-effort path ──────────────────────────────────────

    def test_set_tag_db_best_effort_with_mock(self, full_svc_client, tmp_path):
        """Cover lines 613-616: DB best-effort path in api_set_tag."""
        from unittest.mock import MagicMock, patch
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        mock_conn = MagicMock()
        mock_db = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        with patch.object(_svc, "_db", mock_db):
            with patch.object(_svc, "_tracking", MagicMock()):
                r = client.put("/v1/tags", json={
                    "task": "t", "model_name": "m", "version": "v1",
                    "key": "env", "value": "prod",
                })
        assert r.status_code == 200

    def test_set_tag_bad_meta_json_returns_500(self, full_svc_client, tmp_path):
        """Cover lines 637-638: 500 when model_meta.json is unreadable."""
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        vdir = root / "tasks" / "t" / "models" / "m" / "versions" / "v1"
        (vdir / "model_meta.json").write_text("{{{ bad json")
        r = client.put("/v1/tags", json={
            "task": "t", "model_name": "m", "version": "v1",
            "key": "k", "value": "v",
        })
        assert r.status_code == 500

    # ── api_set_stage edge cases ──────────────────────────────────────────────

    def test_set_stage_not_found_returns_404(self, full_svc_client):
        """Cover line 695: 404 when version dir does not exist."""
        client, _ = full_svc_client
        r = client.post("/v1/stage", json={
            "task": "t", "model_name": "m", "version": "nonexistent", "stage": "staging",
        })
        assert r.status_code == 404

    # ── api_compare bad JSON ──────────────────────────────────────────────────

    def test_compare_bad_metrics_json_uses_empty_dict(self, full_svc_client):
        """Cover lines 751-754: except Exception → entry['metrics'] = {}."""
        client, root = full_svc_client
        _write_unowned_ownership(root, "t", "m")
        for ver in ["v1", "v2"]:
            vdir = root / "tasks" / "t" / "models" / "m" / "versions" / ver
            vdir.mkdir(parents=True)
            (vdir / "model_meta.json").write_text(json.dumps({"task": "t", "version": ver}))
            (vdir / "metrics.json").write_text("{{{ bad")
        r = client.get("/v1/compare", params={"task": "t", "model": "m",
                                               "versions": ["v1", "v2"]})
        assert r.status_code == 200
        data = r.json()
        assert data["versions"]["v1"]["metrics"] == {}
        assert data["versions"]["v2"]["metrics"] == {}

    def test_compare_bad_meta_json_graceful(self, full_svc_client):
        """Cover lines 761-762: except Exception → pass when meta read fails."""
        client, root = full_svc_client
        _write_unowned_ownership(root, "t", "m")
        for ver in ["v1", "v2"]:
            vdir = root / "tasks" / "t" / "models" / "m" / "versions" / ver
            vdir.mkdir(parents=True)
            (vdir / "metrics.json").write_text(json.dumps({"acc": 0.9}))
            (vdir / "model_meta.json").write_text("{{{ bad meta")
        r = client.get("/v1/compare", params={"task": "t", "model": "m",
                                               "versions": ["v1", "v2"]})
        assert r.status_code == 200
        data = r.json()
        assert "v1" in data["versions"]

    # ── api_artifacts edge cases ──────────────────────────────────────────────

    def test_artifacts_nonexistent_model_returns_error(self, full_svc_client):
        """Cover lines 771-772: error path when resolve_model fails."""
        client, _ = full_svc_client
        r = client.get("/v1/artifacts", params={"task": "t", "ref": "m@no_such_version"})
        assert r.status_code in (400, 404, 500)

    def test_artifacts_skips_subdirectory(self, full_svc_client, tmp_path):
        """Cover line 786: continue when entry is a directory (not a file)."""
        client, root = full_svc_client
        src = tmp_path / "src"
        _make_minimal_package(src)
        import omnibioai_model_registry.service.app.main as _svc
        _svc.registry.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias=None,
        )
        vdir = root / "tasks" / "t" / "models" / "m" / "versions" / "v1"
        (vdir / "subdir").mkdir()
        r = client.get("/v1/artifacts", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200
        names = [f["name"] for f in r.json()["files"]]
        assert "subdir" not in names

    # ── api_auth_status exception path ───────────────────────────────────────

    def test_auth_status_load_config_exception(self, full_svc_client, monkeypatch):
        """Cover lines 803-805: except Exception block in api_auth_status."""
        from unittest.mock import patch
        client, _ = full_svc_client
        with patch("omnibioai_model_registry.service.app.main.load_config",
                   side_effect=RuntimeError("config broken")):
            r = client.get("/v1/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert data["auth_enabled"] is False
        assert data["mode"] == "open"


class TestRunCoverageGaps:
    """Cover remaining run.py lines 34-35 (except Exception: pass in cleanup)."""

    def test_atomic_write_json_cleanup_exception_suppressed(self, tmp_path):
        """Cover lines 34-35: os.path.exists raises → except Exception: pass."""
        import omnibioai_model_registry.run as run_mod
        from unittest.mock import patch
        from omnibioai_model_registry.run import _atomic_write_json

        with patch.object(run_mod.os, "replace", side_effect=OSError("replace fail")):
            with patch.object(run_mod.os.path, "exists", side_effect=OSError("exists fail")):
                with pytest.raises(OSError, match="replace fail"):
                    _atomic_write_json(tmp_path / "out.json", {"data": 1})


# ============================================================
# Phase 1 — HIPAA/security hardening: read-path authentication
# ============================================================
#
# Closes the unauthenticated-read-path gap identified in the
# tenant-isolation discovery audit: every non-informational data-bearing
# route (previously only mutations were IAM-gated) must now independently
# authenticate the caller via the same omnibioai-iam-client mechanism,
# rather than relying on the API Gateway / network topology alone.
#
# Deliberately NOT covered here (out of scope for Phase 1, see the PR
# description): organization/team ownership checks, org_id columns,
# model.use redesign, service-to-service tokens, or any change to the
# flat-namespace registration/storage semantics.


class TestPhase1ReadEndpointsRequireAuth:
    """Route-level (real HTTP, via TestClient) coverage proving every
    previously-open GET/data-bearing route now independently requires a
    valid Bearer JWT verified through iam-client, with model.use enforced
    exactly as it already is for writes -- no new/separate permission."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    @pytest.fixture
    def auth_client(self, tmp_path, monkeypatch):
        """AUTH_ENABLED=true TestClient with one registered model version,
        so protected routes have real data to resolve once authenticated."""
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")

        import omnibioai_model_registry.service.app.main as _svc
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)

        src = tmp_path / "src"
        _make_minimal_package(src)
        new_reg.register_model(
            task="t", model_name="m", version="v1",
            artifacts_dir=src, metadata={}, set_alias="latest",
        )
        return TestClient(_svc.app, raise_server_exceptions=False), new_reg.root

    # ── 1-9 explicitly listed routes + discovered extras: 401 when absent ──

    @pytest.mark.parametrize("method,path,params", [
        ("get", "/v1/models", {}),
        ("get", "/v1/show", {"task": "t", "ref": "m@v1"}),
        ("get", "/v1/resolve", {"task": "t", "ref": "m@v1"}),
        ("get", "/v1/runs/get", {"task": "t", "model": "m", "run_id": "r1"}),
        ("get", "/v1/runs/list", {"task": "t", "model": "m"}),
        ("get", "/v1/metrics", {"task": "t", "ref": "m@v1"}),
        ("get", "/v1/aliases", {"task": "t", "model": "m"}),
        ("get", "/v1/compare", {"task": "t", "model": "m", "versions": ["v1", "v1"]}),
        ("get", "/v1/artifacts", {"task": "t", "ref": "m@v1"}),
        ("get", "/v1/hf/push/status/some-job-id", {}),
    ], ids=[
        "models", "show", "resolve", "runs_get", "runs_list",
        "metrics", "aliases", "compare", "artifacts", "hf_push_status",
    ])
    def test_read_endpoint_without_authorization_returns_401(self, auth_client, method, path, params):
        client, _ = auth_client
        r = getattr(client, method)(path, params=params)
        assert r.status_code == 401

    def test_verify_without_authorization_returns_401(self, auth_client):
        """POST /v1/verify is a data-bearing existence/integrity oracle
        discovered during the Phase 1 route-by-route review (not in the
        explicit GET list, but non-informational) -- protected on the
        same basis."""
        client, _ = auth_client
        r = client.post("/v1/verify", json={"task": "t", "ref": "m@v1"})
        assert r.status_code == 401

    # ── malformed / expired / revoked tokens ────────────────────────────────

    def test_malformed_authorization_header_returns_401(self, auth_client):
        client, _ = auth_client
        r = client.get("/v1/models", headers={"Authorization": "NotBearer sometoken"})
        assert r.status_code == 401

    def test_expired_or_revoked_token_returns_401(self, auth_client, monkeypatch):
        """AsyncIAMClient.get_user() returns None for an expired/revoked
        token -- this layer trusts the IAM client's verdict rather than
        re-deriving the reason locally, same contract already asserted
        for require_auth/require_write_auth in TestAuthRequireWriteAuth."""
        self._mock_iam_client(monkeypatch, None)
        client, _ = auth_client
        r = client.get("/v1/models", headers={"Authorization": "Bearer expiredtoken"})
        assert r.status_code == 401

    def test_valid_token_without_model_use_returns_403(self, auth_client, monkeypatch):
        from iam_client.models import UserContext
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=[], valid=True,
        ))
        client, _ = auth_client
        r = client.get("/v1/models", headers={"Authorization": "Bearer sometoken"})
        assert r.status_code == 403

    # ── authenticated + model.use → existing behavior fully preserved ──────

    def test_authenticated_with_model_use_preserves_models_behavior(self, auth_client, monkeypatch):
        from iam_client.models import UserContext
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=["model.use"], valid=True,
        ))
        client, _ = auth_client
        r = client.get("/v1/models", headers={"Authorization": "Bearer sometoken"})
        assert r.status_code == 200
        models = r.json()
        assert isinstance(models, list)
        assert any(m.get("model_name") == "m" for m in models)

    def test_authenticated_with_model_use_preserves_resolve_behavior(self, auth_client, monkeypatch):
        from iam_client.models import UserContext
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=["model.use"], valid=True,
        ))
        client, _ = auth_client
        r = client.get(
            "/v1/resolve", params={"task": "t", "ref": "m@v1"},
            headers={"Authorization": "Bearer sometoken"},
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["path"].endswith("v1")

    # ── 400/404-equivalent behavior preserved after auth succeeds ──────────

    def test_show_missing_ref_still_returns_400_after_auth(self, auth_client, monkeypatch):
        """ModelNotFound -> _handle_registry_error -> 400, unchanged by
        adding auth (matches the pre-existing full_svc_client-based
        assertion for the same scenario without auth enabled)."""
        from iam_client.models import UserContext
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=["model.use"], valid=True,
        ))
        client, _ = auth_client
        r = client.get(
            "/v1/show", params={"task": "t", "ref": "m@does_not_exist"},
            headers={"Authorization": "Bearer sometoken"},
        )
        assert r.status_code == 400

    def test_compare_requires_two_versions_after_auth(self, auth_client, monkeypatch):
        from iam_client.models import UserContext
        self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[], permissions=["model.use"], valid=True,
        ))
        client, _ = auth_client
        r = client.get(
            "/v1/compare", params={"task": "t", "model": "m", "versions": ["v1"]},
            headers={"Authorization": "Bearer sometoken"},
        )
        assert r.status_code == 400

    # ── spoofed gateway identity headers must never authenticate ───────────

    def test_spoofed_gateway_headers_do_not_authenticate(self, auth_client):
        """The registry independently verifies identity via iam-client; it
        must never treat gateway-injected identity headers as a substitute
        for a verified Bearer token, even if a well-formed one is present
        without a token."""
        client, _ = auth_client
        r = client.get(
            "/v1/models",
            headers={
                "X-Organization-ID": "org-attacker",
                "X-Team-ID": "team-attacker",
                "X-User-ID": "9999",
                "X-User-Email": "attacker@example.com",
            },
        )
        assert r.status_code == 401

    def test_spoofed_gateway_headers_alongside_invalid_token_still_401(self, auth_client, monkeypatch):
        """Even with an Authorization header present, spoofed identity
        headers must not influence the outcome -- only IAM's verdict on
        the token itself matters."""
        self._mock_iam_client(monkeypatch, None)
        client, _ = auth_client
        r = client.get(
            "/v1/models",
            headers={
                "Authorization": "Bearer invalidtoken",
                "X-Organization-ID": "org-attacker",
                "X-User-Email": "attacker@example.com",
            },
        )
        assert r.status_code == 401

    # ── AUTH_ENABLED=false: pre-existing dev/test bypass, unchanged ────────

    def test_auth_disabled_still_allows_reads_without_a_token(self, tmp_path, monkeypatch):
        """Phase 1 does not change AUTH_ENABLED semantics. When explicitly
        disabled -- the same pre-existing dev/test switch every write
        route already honors -- require_auth still returns the synthetic
        'system' actor and protected reads remain open. This is not a new
        bypass: it is the existing write-route behavior now applied
        consistently to reads instead of reads having no gate at all. The
        default (AUTH_ENABLED unset) is 'false' both here and in
        production unless an operator explicitly opts in -- unchanged by
        this PR."""
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)

        import omnibioai_model_registry.service.app.main as _svc
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        client = TestClient(_svc.app, raise_server_exceptions=False)

        r = client.get("/v1/models")
        assert r.status_code == 200

    # ── informational endpoints deliberately remain public ─────────────────

    def test_health_remains_public(self, auth_client):
        client, _ = auth_client
        r = client.get("/health")
        assert r.status_code == 200

    def test_auth_status_remains_public(self, auth_client):
        client, _ = auth_client
        r = client.get("/v1/auth/status")
        assert r.status_code == 200

    def test_hf_settings_remains_public(self, auth_client):
        client, _ = auth_client
        r = client.get("/v1/hf/settings")
        assert r.status_code == 200


# ============================================================
# Phase 2A — HIPAA hardening: organization-ownership foundation
# ============================================================
#
# Establishes durable, server-derived organization ownership for models
# (not versions -- versions inherit by construction, see ownership.py).
# Deliberately NOT covered here (out of scope for Phase 2A): query-layer
# cross-org enforcement, HF-push ownership check, model.use redesign,
# public/private semantics. Those belong to later phases.


class TestOwnershipModule:
    """Unit-level coverage of ownership.py: the write-once record, legacy
    detection, and the backfill migration utility -- independent of the
    HTTP layer."""

    def test_new_model_owned_by_given_org(self, env_root, tmp_path):
        from omnibioai_model_registry.ownership import ensure_model_ownership
        from omnibioai_model_registry.storage.localfs import LocalFS

        backend = LocalFS()
        rec = ensure_model_ownership(
            backend, env_root, "t", "m",
            organization_id="org-A", actor="alice@a.com", model_pre_existing=False,
        )
        assert rec.organization_id == "org-A"
        assert rec.status == "owned"
        assert rec.registered_by == "alice@a.com"
        assert rec.registered_at is not None
        assert rec.discovered_at is None

    def test_new_model_with_no_org_is_unowned_not_guessed(self, env_root):
        from omnibioai_model_registry.ownership import ensure_model_ownership
        from omnibioai_model_registry.storage.localfs import LocalFS

        backend = LocalFS()
        rec = ensure_model_ownership(
            backend, env_root, "t", "m",
            organization_id=None, actor="system", model_pre_existing=False,
        )
        assert rec.organization_id is None
        assert rec.status == "unowned"

    def test_pre_existing_model_becomes_legacy_unowned_not_current_org(self, env_root):
        """The caller's org must NEVER be assigned to a model that already
        had versions before this call -- that would let anyone "claim" an
        old model just by touching it."""
        from omnibioai_model_registry.ownership import ensure_model_ownership
        from omnibioai_model_registry.storage.localfs import LocalFS

        backend = LocalFS()
        rec = ensure_model_ownership(
            backend, env_root, "t", "old_model",
            organization_id="org-X", actor="whoever@x.com", model_pre_existing=True,
        )
        assert rec.organization_id is None
        assert rec.status == "legacy_unowned"
        assert rec.registered_by is None
        assert rec.discovered_at is not None

    def test_ownership_is_write_once_second_call_does_not_overwrite(self, env_root):
        from omnibioai_model_registry.ownership import ensure_model_ownership, read_ownership
        from omnibioai_model_registry.storage.localfs import LocalFS

        backend = LocalFS()
        first = ensure_model_ownership(
            backend, env_root, "t", "m",
            organization_id="org-A", actor="alice@a.com", model_pre_existing=False,
        )
        # A second call for the "same" model, now claiming pre_existing
        # (as register_model would compute on a subsequent version) with
        # a DIFFERENT org -- must be a no-op.
        second = ensure_model_ownership(
            backend, env_root, "t", "m",
            organization_id="org-B", actor="mallory@b.com", model_pre_existing=True,
        )
        assert second == first
        assert second.organization_id == "org-A"
        on_disk = read_ownership(env_root, "t", "m")
        assert on_disk == first

    def test_read_ownership_none_when_never_established(self, env_root):
        from omnibioai_model_registry.ownership import read_ownership
        assert read_ownership(env_root, "t", "never_touched") is None

    def test_write_once_text_second_writer_loses_race(self, env_root):
        from omnibioai_model_registry.storage.localfs import LocalFS

        backend = LocalFS()
        target = Path(env_root) / "race.json"
        first = backend.write_once_text(target, "first\n")
        second = backend.write_once_text(target, "second\n")
        assert first is True
        assert second is False
        assert target.read_text() == "first\n"

    def test_backfill_scans_and_migrates_only_unowned_models(self, env_root, tmp_path):
        from omnibioai_model_registry import register_model
        from omnibioai_model_registry.ownership import (
            backfill_legacy_ownership,
            read_ownership,
        )

        src = tmp_path / "src"
        _make_minimal_package(src)

        # A properly-owned Phase-2A model.
        register_model(
            task="t", model_name="owned_model", version="v1",
            artifacts_dir=str(src), metadata={}, set_alias=None,
            organization_id="org-A", actor="alice@a.com",
        )
        # Two genuinely legacy models: real version directories with no
        # ownership.json (simulating pre-Phase-2A data).
        for name in ("legacy_one", "legacy_two"):
            legacy_dir = (
                env_root / "tasks" / "t" / "models" / name / "versions" / "v1"
            )
            legacy_dir.mkdir(parents=True)
            for f in REQUIRED_FILES:
                (legacy_dir / f).write_text("{}" if f.endswith(".json") else "x")

        summary = backfill_legacy_ownership(env_root)
        assert summary == {"scanned": 3, "migrated": 2, "already_had_ownership": 1}

        assert read_ownership(env_root, "t", "owned_model").status == "owned"
        assert read_ownership(env_root, "t", "legacy_one").status == "legacy_unowned"
        assert read_ownership(env_root, "t", "legacy_two").status == "legacy_unowned"

    def test_backfill_skips_task_dir_with_no_models_subdir(self, env_root):
        """A task/ directory that exists but has no models/ subdirectory
        yet (e.g. an empty task namespace) must be skipped, not error."""
        from omnibioai_model_registry.ownership import backfill_legacy_ownership

        (env_root / "tasks" / "empty_task").mkdir(parents=True)
        summary = backfill_legacy_ownership(env_root)
        assert summary == {"scanned": 0, "migrated": 0, "already_had_ownership": 0}

    def test_backfill_is_idempotent_on_rerun(self, env_root):
        from omnibioai_model_registry.ownership import backfill_legacy_ownership

        legacy_dir = env_root / "tasks" / "t" / "models" / "legacy" / "versions" / "v1"
        legacy_dir.mkdir(parents=True)
        for f in REQUIRED_FILES:
            (legacy_dir / f).write_text("{}" if f.endswith(".json") else "x")

        first = backfill_legacy_ownership(env_root)
        second = backfill_legacy_ownership(env_root)
        assert first == {"scanned": 1, "migrated": 1, "already_had_ownership": 0}
        assert second == {"scanned": 1, "migrated": 0, "already_had_ownership": 1}

    def test_backfill_on_empty_registry_is_a_safe_noop(self, tmp_path, monkeypatch):
        """MIGRATION TESTING: empty registry (no tasks/ dir at all yet)."""
        from omnibioai_model_registry.ownership import backfill_legacy_ownership

        empty_root = tmp_path / "brand_new_registry"
        summary = backfill_legacy_ownership(empty_root)
        assert summary == {"scanned": 0, "migrated": 0, "already_had_ownership": 0}
        # Must not have created the tasks/ dir as a side effect.
        assert not empty_root.exists() or not (empty_root / "tasks").exists()

    def test_backfill_does_not_touch_existing_artifacts_or_data(self, env_root):
        """MIGRATION TESTING: no existing model/version/artifact is lost
        or modified -- the migration is additive-only."""
        from omnibioai_model_registry.ownership import backfill_legacy_ownership

        legacy_dir = env_root / "tasks" / "t" / "models" / "legacy" / "versions" / "v1"
        legacy_dir.mkdir(parents=True)
        contents_before = {}
        for f in REQUIRED_FILES:
            text = "{}" if f.endswith(".json") else "original-bytes"
            (legacy_dir / f).write_text(text)
            contents_before[f] = text

        backfill_legacy_ownership(env_root)

        for f, text in contents_before.items():
            assert (legacy_dir / f).read_text() == text, f"{f} was modified by migration"
        # Only the new ownership.json should have appeared alongside the
        # existing versions/aliases/audit structure at the model root.
        model_root_dir = env_root / "tasks" / "t" / "models" / "legacy"
        assert (model_root_dir / "ownership.json").exists()
        assert (model_root_dir / "versions" / "v1" / "model.pt").exists()


class TestRegisterModelOwnership:
    """api.py-level (ModelRegistry.register_model) ownership behavior --
    below the HTTP layer, proving the core write-once/legacy contract the
    HTTP tests below also exercise end-to-end."""

    def test_register_response_includes_server_derived_ownership(self, env_root, tmp_path):
        from omnibioai_model_registry import register_model

        src = tmp_path / "src"
        _make_minimal_package(src)
        out = register_model(
            task="t", model_name="m", version="v1", artifacts_dir=str(src),
            metadata={}, set_alias=None, organization_id="org-A", actor="alice@a.com",
        )
        assert out["organization_id"] == "org-A"
        assert out["ownership_status"] == "owned"

    def test_second_version_under_existing_model_does_not_reassign_ownership(
        self, env_root, tmp_path
    ):
        """Security requirement #6: existing model ownership cannot be
        overwritten during version registration, even by a legitimately
        authenticated different-org caller (Phase 2A does not yet BLOCK
        this write -- that's a later phase -- but ownership attribution
        itself must never drift)."""
        from omnibioai_model_registry import register_model

        src = tmp_path / "src"
        _make_minimal_package(src)
        out1 = register_model(
            task="t", model_name="m", version="v1", artifacts_dir=str(src),
            metadata={}, set_alias=None, organization_id="org-A", actor="alice@a.com",
        )
        out2 = register_model(
            task="t", model_name="m", version="v2", artifacts_dir=str(src),
            metadata={}, set_alias=None, organization_id="org-B", actor="mallory@b.com",
        )
        assert out1["organization_id"] == "org-A"
        assert out2["organization_id"] == "org-A"  # NOT org-B
        assert out2["ownership_status"] == "owned"

    def test_metadata_body_organization_id_key_is_never_consulted(self, env_root, tmp_path):
        """Security requirement #3: a client cannot influence ownership by
        stuffing an organization_id (or any other) key into the free-form
        metadata dict -- register_model() never reads it from there."""
        from omnibioai_model_registry import register_model

        src = tmp_path / "src"
        _make_minimal_package(src)
        out = register_model(
            task="t", model_name="m", version="v1", artifacts_dir=str(src),
            metadata={"organization_id": "org-SPOOFED", "team_id": "team-SPOOFED"},
            set_alias=None, organization_id="org-A", actor="alice@a.com",
        )
        assert out["organization_id"] == "org-A"

    def test_probe_via_resolve_before_registration_does_not_create_false_legacy(
        self, env_root, tmp_path
    ):
        """Regression guard: resolve_model()/promote_model() also call
        _ensure_model_dirs() as a side effect. A GET-style probe against a
        model that was never registered must not cause the real, later
        registration to be mistaken for a pre-existing (legacy) model."""
        from omnibioai_model_registry import register_model, resolve_model
        from omnibioai_model_registry.errors import ModelNotFound

        with pytest.raises(ModelNotFound):
            resolve_model(task="t", model_ref="brandnew@v1", verify=False)

        src = tmp_path / "src"
        _make_minimal_package(src)
        out = register_model(
            task="t", model_name="brandnew", version="v1", artifacts_dir=str(src),
            metadata={}, set_alias=None, organization_id="org-A", actor="alice@a.com",
        )
        assert out["organization_id"] == "org-A"
        assert out["ownership_status"] == "owned"

    def test_organization_id_defaults_to_none_for_direct_api_callers(self, env_root, tmp_path):
        """A caller with no IAM identity at all (direct Python API usage,
        matching CLI behavior without --org-id) gets status=unowned, never
        a guessed organization."""
        from omnibioai_model_registry import register_model

        src = tmp_path / "src"
        _make_minimal_package(src)
        out = register_model(
            task="t", model_name="m", version="v1", artifacts_dir=str(src),
            metadata={}, set_alias=None, actor="alice@a.com",
        )
        assert out["organization_id"] is None
        assert out["ownership_status"] == "unowned"


class TestPhase2AHTTPRegisterOwnership:
    """HTTP/TestClient-level: proves organization_id can only ever come
    from the verified JWT (via require_write_auth_with_context), never
    from a header, body field, or query/path parameter."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    @pytest.fixture
    def auth_client(self, tmp_path, monkeypatch):
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")

        import omnibioai_model_registry.service.app.main as _svc
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        return TestClient(_svc.app, raise_server_exceptions=False), new_reg.root

    def _register_payload(self, tmp_path, **overrides):
        src = tmp_path / "src"
        _make_minimal_package(src)
        payload = {
            "task": "t", "model_name": "m", "version": "v1",
            "artifacts_dir": str(src), "metadata": {}, "set_alias": None,
        }
        payload.update(overrides)
        return payload

    def _as_org(self, monkeypatch, org_id, permissions=("model.use",)):
        from iam_client.models import UserContext
        return self._mock_iam_client(monkeypatch, UserContext(
            user_id="1", email="user@test.com", roles=[],
            permissions=list(permissions), valid=True, org_id=org_id,
        ))

    def test_register_authenticated_as_org_a_owns_model(self, auth_client, tmp_path, monkeypatch):
        self._as_org(monkeypatch, "org-A")
        client, _ = auth_client
        r = client.post(
            "/v1/register", json=self._register_payload(tmp_path),
            headers={"Authorization": "Bearer sometoken"},
        )
        assert r.status_code == 200
        assert r.json()["organization_id"] == "org-A"
        assert r.json()["ownership_status"] == "owned"

    def test_spoofed_x_organization_id_header_does_not_change_ownership(
        self, auth_client, tmp_path, monkeypatch
    ):
        """Security requirement #2."""
        self._as_org(monkeypatch, "org-A")
        client, _ = auth_client
        r = client.post(
            "/v1/register", json=self._register_payload(tmp_path),
            headers={
                "Authorization": "Bearer sometoken",
                "X-Organization-ID": "org-SPOOFED",
            },
        )
        assert r.status_code == 200
        assert r.json()["organization_id"] == "org-A"

    def test_spoofed_x_team_id_header_does_not_change_ownership(
        self, auth_client, tmp_path, monkeypatch
    ):
        """Security requirement #3."""
        self._as_org(monkeypatch, "org-A")
        client, _ = auth_client
        r = client.post(
            "/v1/register", json=self._register_payload(tmp_path),
            headers={
                "Authorization": "Bearer sometoken",
                "X-Team-ID": "team-SPOOFED",
            },
        )
        assert r.status_code == 200
        assert r.json()["organization_id"] == "org-A"

    def test_conflicting_metadata_body_field_is_rejected_server_derived_wins(
        self, auth_client, tmp_path, monkeypatch
    ):
        self._as_org(monkeypatch, "org-A")
        client, _ = auth_client
        payload = self._register_payload(
            tmp_path, metadata={"organization_id": "org-SPOOFED"}
        )
        r = client.post(
            "/v1/register", json=payload,
            headers={"Authorization": "Bearer sometoken"},
        )
        assert r.status_code == 200
        assert r.json()["organization_id"] == "org-A"

    def test_ownership_derived_only_after_jwt_verification_not_before(
        self, auth_client, tmp_path, monkeypatch
    ):
        """An invalid/unverified token must never reach ownership logic at
        all -- 401 before any registration or ownership write happens."""
        self._mock_iam_client(monkeypatch, None)
        client, root = auth_client
        r = client.post(
            "/v1/register", json=self._register_payload(tmp_path),
            headers={
                "Authorization": "Bearer invalidtoken",
                "X-Organization-ID": "org-SPOOFED",
            },
        )
        assert r.status_code == 401
        from omnibioai_model_registry.ownership import read_ownership
        assert read_ownership(root, "t", "m") is None

    def test_second_org_registering_new_version_does_not_reassign_ownership_http(
        self, auth_client, tmp_path, monkeypatch
    ):
        """Security requirement #6, end-to-end over HTTP. Phase 2A recorded
        this as "ownership attribution never drifts, but the write is not
        yet BLOCKED"; Phase 2B closes that gap -- org-B's attempt is now
        denied outright (register_model(enforce_ownership=True)), so
        ownership not only stays org-A's, the version is never written at
        all (see TestPhase2BOrgEnforcement for the filesystem-mutation
        assertion)."""
        client, root = auth_client

        self._as_org(monkeypatch, "org-A")
        r1 = client.post(
            "/v1/register", json=self._register_payload(tmp_path),
            headers={"Authorization": "Bearer token-a"},
        )
        assert r1.status_code == 200
        assert r1.json()["organization_id"] == "org-A"

        self._as_org(monkeypatch, "org-B")
        r2 = client.post(
            "/v1/register",
            json=self._register_payload(tmp_path, version="v2"),
            headers={"Authorization": "Bearer token-b"},
        )
        assert r2.status_code == 400  # denied, not silently reattributed
        from omnibioai_model_registry.ownership import read_ownership
        assert read_ownership(root, "t", "m").organization_id == "org-A"

    def test_auth_disabled_register_is_unowned_not_a_new_bypass(self, tmp_path, monkeypatch):
        """AUTH_ENABLED=false is the pre-existing dev/test switch (already
        used by every write route since Phase 1) -- it still runs fully
        open, attributing to the synthetic 'system' actor with no org.
        This is unchanged, documented, existing behavior, not a new
        insecure default introduced by Phase 2A."""
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)

        import omnibioai_model_registry.service.app.main as _svc
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        client = TestClient(_svc.app, raise_server_exceptions=False)

        r = client.post("/v1/register", json=self._register_payload(tmp_path))
        assert r.status_code == 200
        assert r.json()["organization_id"] is None
        assert r.json()["ownership_status"] == "unowned"

    def test_audit_event_carries_ownership_metadata_smallest_compatible_change(
        self, auth_client, tmp_path, monkeypatch
    ):
        """Reuses the existing AuditClient.log_event call site (no new
        audit plumbing) -- the 'register_model' event's metadata now
        includes organization_id/ownership_status."""
        from unittest.mock import patch

        self._as_org(monkeypatch, "org-A")
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        with patch.object(_svc, "_audit") as mock_audit:
            r = client.post(
                "/v1/register", json=self._register_payload(tmp_path),
                headers={"Authorization": "Bearer sometoken"},
            )
        assert r.status_code == 200
        mock_audit.log_event.assert_called_once()
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["metadata"]["organization_id"] == "org-A"
        assert kwargs["metadata"]["ownership_status"] == "owned"

    def test_no_secrets_or_tokens_in_ownership_record(self, auth_client, tmp_path, monkeypatch):
        """Security requirement #9: the JWT/Authorization header value
        itself must never be persisted anywhere in the ownership record."""
        self._as_org(monkeypatch, "org-A")
        client, root = auth_client
        r = client.post(
            "/v1/register", json=self._register_payload(tmp_path),
            headers={"Authorization": "Bearer super-secret-token-value"},
        )
        assert r.status_code == 200
        from omnibioai_model_registry.ownership import read_ownership
        rec = read_ownership(root, "t", "m")
        assert "super-secret-token-value" not in rec.to_json()


class TestMigrateOwnershipCLI:
    """CLI-level coverage: --org-id on `omr register` (admin/operator
    convenience for the out-of-band, non-HTTP path) and the
    `omr migrate-ownership` backfill command."""

    def test_register_with_org_id_flag(self, env_root, tmp_path, capsys):
        from omnibioai_model_registry.cli.main import build_parser

        src = tmp_path / "src"
        _make_minimal_package(src)
        parser = build_parser()
        args = parser.parse_args([
            "register", "--task", "t", "--model", "m", "--version", "v1",
            "--artifacts", str(src), "--org-id", "org-A", "--set-alias", "",
        ])
        if args.set_alias == "":
            args.set_alias = None
        args.func(args)
        out = capsys.readouterr().out
        assert "organization_id=org-A" in out
        assert "status=owned" in out

    def test_migrate_ownership_cli_json_summary(self, env_root, capsys):
        from omnibioai_model_registry.cli.main import build_parser

        legacy_dir = env_root / "tasks" / "t" / "models" / "legacy" / "versions" / "v1"
        legacy_dir.mkdir(parents=True)
        for f in REQUIRED_FILES:
            (legacy_dir / f).write_text("{}" if f.endswith(".json") else "x")

        parser = build_parser()
        args = parser.parse_args(["migrate-ownership", "--json"])
        args.func(args)
        out = capsys.readouterr().out
        summary = json.loads(out)
        assert summary == {"scanned": 1, "migrated": 1, "already_had_ownership": 0}

    def test_migrate_ownership_cli_human_readable(self, env_root, capsys):
        from omnibioai_model_registry.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["migrate-ownership"])
        args.func(args)
        out = capsys.readouterr().out
        assert "Scanned:" in out
        assert "Migrated to legacy:" in out


# ============================================================
# Phase 2B — HIPAA hardening: organization-scoped enforcement
# ============================================================
#
# Builds on Phase 2A's ownership foundation (TestOwnershipModule /
# TestRegisterModelOwnership / TestPhase2AHTTPRegisterOwnership above):
# check_model_ownership() is now wired into every read/write route (see
# ownership.py's "PHASE 2B ADDENDUM" and api.py/main.py/hf_routes.py's
# call sites), turning the write-once record into real cross-org denial.


class TestCheckModelOwnership:
    """Unit-level coverage of check_model_ownership() itself -- the single
    centralized authorization decision every route above goes through."""

    def test_owned_model_matching_org_allowed(self, env_root):
        from omnibioai_model_registry.ownership import (
            check_model_ownership, ensure_model_ownership,
        )
        from omnibioai_model_registry.storage.localfs import LocalFS

        ensure_model_ownership(
            LocalFS(), env_root, "t", "m",
            organization_id="org-A", actor="alice", model_pre_existing=False,
        )
        result = check_model_ownership(env_root, "t", "m", requesting_org_id="org-A")
        assert result.allowed is True
        assert result.reason == "owned_by_caller"

    def test_owned_model_different_org_denied(self, env_root):
        from omnibioai_model_registry.ownership import (
            check_model_ownership, ensure_model_ownership,
        )
        from omnibioai_model_registry.storage.localfs import LocalFS

        ensure_model_ownership(
            LocalFS(), env_root, "t", "m",
            organization_id="org-A", actor="alice", model_pre_existing=False,
        )
        result = check_model_ownership(env_root, "t", "m", requesting_org_id="org-B")
        assert result.allowed is False
        assert result.reason == "owned_by_other_org"
        # Server-side visibility into who actually owns it is preserved
        # for audit -- just never exposed to the denied caller by any
        # route above.
        assert result.ownership.organization_id == "org-A"

    def test_unowned_model_open_mode_caller_allowed(self, env_root):
        """AUTH_ENABLED=false registrant, AUTH_ENABLED=false caller: both
        sides genuinely have no org context -- unchanged pre-Phase-2B dev
        mode."""
        from omnibioai_model_registry.ownership import (
            check_model_ownership, ensure_model_ownership,
        )
        from omnibioai_model_registry.storage.localfs import LocalFS

        ensure_model_ownership(
            LocalFS(), env_root, "t", "m",
            organization_id=None, actor="system", model_pre_existing=False,
        )
        result = check_model_ownership(env_root, "t", "m", requesting_org_id=None)
        assert result.allowed is True
        assert result.reason == "open_mode_match"

    def test_unowned_model_real_org_caller_denied(self, env_root):
        """A real org_id reaching into a None-org model is NOT treated as
        'everyone's' -- denied the same as any other org mismatch."""
        from omnibioai_model_registry.ownership import (
            check_model_ownership, ensure_model_ownership,
        )
        from omnibioai_model_registry.storage.localfs import LocalFS

        ensure_model_ownership(
            LocalFS(), env_root, "t", "m",
            organization_id=None, actor="system", model_pre_existing=False,
        )
        result = check_model_ownership(env_root, "t", "m", requesting_org_id="org-A")
        assert result.allowed is False
        assert result.reason == "owned_by_other_org"

    def test_legacy_unowned_denied_for_every_caller(self, env_root):
        from omnibioai_model_registry.ownership import (
            check_model_ownership, ensure_model_ownership,
        )
        from omnibioai_model_registry.storage.localfs import LocalFS

        ensure_model_ownership(
            LocalFS(), env_root, "t", "old_model",
            organization_id="org-X", actor="whoever", model_pre_existing=True,
        )
        for requesting_org_id in ("org-A", None):
            result = check_model_ownership(
                env_root, "t", "old_model", requesting_org_id=requesting_org_id
            )
            assert result.allowed is False
            assert result.reason == "legacy_unowned"

    def test_nonexistent_model_denied_not_a_crash(self, env_root):
        from omnibioai_model_registry.ownership import check_model_ownership

        result = check_model_ownership(env_root, "t", "never_registered", requesting_org_id="org-A")
        assert result.allowed is False
        assert result.reason == "model_not_found"
        assert result.ownership is None


class TestPhase2BOrgEnforcement:
    """HTTP/TestClient-level: two organizations, one model owned by org-A
    -- every read/write route must independently enforce
    verified_user.org_id == model_owner.organization_id, denying org-B
    the same anti-enumerating way a genuinely nonexistent model would be
    denied, with zero behavior change for org-A itself."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    def _as_org(self, monkeypatch, org_id, *, permissions=("model.use",)):
        from iam_client.models import UserContext
        return self._mock_iam_client(monkeypatch, UserContext(
            user_id=f"user-{org_id}", email=f"user@{org_id}.example",
            roles=[], permissions=list(permissions), valid=True, org_id=org_id,
        ))

    @pytest.fixture
    def auth_client(self, tmp_path, monkeypatch):
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")

        import omnibioai_model_registry.service.app.main as _svc
        import omnibioai_model_registry.hf_routes as _hf
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        # hf_routes.py builds its own ModelRegistry once at import time
        # (module-level _registry, unrelated to _svc.registry) -- keep
        # both pointed at the same per-test root.
        monkeypatch.setattr(_hf, "_registry", new_reg)

        class _SyncThread:
            """Deterministic stand-in for threading.Thread: hf_push's
            background push runs synchronously so tests don't need
            sleeps/polling to observe its effect (or its absence)."""
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target
                self._args = args
                self._kwargs = kwargs or {}

            def start(self):
                self._target(*self._args, **self._kwargs)

        class _FakeThreadingModule:
            Thread = _SyncThread

        # Rebinds only hf_routes.py's own `threading` name -- the real
        # stdlib threading module (used by audit_client.py and everything
        # else) is untouched.
        monkeypatch.setattr(_hf, "threading", _FakeThreadingModule())

        return TestClient(_svc.app, raise_server_exceptions=False), new_reg.root

    def _register(
        self, client, tmp_path, monkeypatch, org_id, *,
        task="t", model_name="m", version="v1", token=None, set_alias="latest",
    ):
        self._as_org(monkeypatch, org_id)
        src = tmp_path / f"src_{task}_{model_name}_{version}_{org_id}"
        _make_minimal_package(src)
        r = client.post(
            "/v1/register",
            json={
                "task": task, "model_name": model_name, "version": version,
                "artifacts_dir": str(src), "metadata": {}, "set_alias": set_alias,
            },
            headers={"Authorization": f"Bearer {token or org_id}"},
        )
        assert r.status_code == 200, r.text
        return r.json()

    @pytest.fixture
    def org_a_model(self, auth_client, tmp_path, monkeypatch):
        """task=t, model_name=m, v1 registered by org-A with alias
        'latest'."""
        client, root = auth_client
        self._register(client, tmp_path, monkeypatch, "org-A")
        return client, root

    # ── reads: org-A allowed, org-B denied, across every read route ────────

    def test_resolve_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.get("/v1/resolve", params={"task": "t", "ref": "m@v1"},
                        headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.get("/v1/resolve", params={"task": "t", "ref": "m@v1"},
                        headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 400  # ModelNotFound -> _handle_registry_error

    def test_show_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.get("/v1/show", params={"task": "t", "ref": "m@v1"},
                        headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.get("/v1/show", params={"task": "t", "ref": "m@v1"},
                        headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 400

    def test_verify_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.post("/v1/verify", json={"task": "t", "ref": "m@v1"},
                         headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.post("/v1/verify", json={"task": "t", "ref": "m@v1"},
                         headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 400

    def test_artifacts_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.get("/v1/artifacts", params={"task": "t", "ref": "m@v1"},
                        headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.get("/v1/artifacts", params={"task": "t", "ref": "m@v1"},
                        headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 400

    def test_metrics_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-B")
        r = client.get("/v1/metrics", params={"task": "t", "ref": "m@v1"},
                        headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 400

    def test_aliases_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.get("/v1/aliases", params={"task": "t", "model": "m"},
                        headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        assert len(r.json()["aliases"]) == 1

        self._as_org(monkeypatch, "org-B")
        r = client.get("/v1/aliases", params={"task": "t", "model": "m"},
                        headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 404

    def test_compare_org_a_allowed_org_b_denied(self, org_a_model, tmp_path, monkeypatch):
        client, _ = org_a_model
        self._register(client, tmp_path, monkeypatch, "org-A", version="v2", set_alias=None)

        self._as_org(monkeypatch, "org-A")
        r = client.get("/v1/compare", params={"task": "t", "model": "m", "versions": ["v1", "v2"]},
                        headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.get("/v1/compare", params={"task": "t", "model": "m", "versions": ["v1", "v2"]},
                        headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 404

    def test_models_list_returns_only_caller_owned(self, org_a_model, tmp_path, monkeypatch):
        """Security requirement #8."""
        client, _ = org_a_model
        self._register(client, tmp_path, monkeypatch, "org-B", model_name="m_b")

        self._as_org(monkeypatch, "org-A")
        r = client.get("/v1/models", headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        names = {m["model_name"] for m in r.json()}
        assert names == {"m"}

        self._as_org(monkeypatch, "org-B")
        r = client.get("/v1/models", headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 200
        names = {m["model_name"] for m in r.json()}
        assert names == {"m_b"}

    def test_models_list_skips_malformed_meta_missing_model_name(self, org_a_model):
        """A model_meta.json with no task/model_name (corrupt/malformed)
        is skipped defensively before it ever reaches
        check_model_ownership -- which would otherwise be asked to build
        an ownership.json path out of a None component."""
        client, root = org_a_model
        bad_dir = root / "tasks" / "t" / "models" / "corrupt" / "versions" / "v1"
        bad_dir.mkdir(parents=True)
        (bad_dir / "model_meta.json").write_text(json.dumps({"version": "v1"}))

        r = client.get("/v1/models", headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        assert "corrupt" not in {m.get("model_name") for m in r.json()}

    # ── writes: org-A allowed, org-B denied ─────────────────────────────────

    def test_register_new_version_org_a_allowed_org_b_denied(self, org_a_model, tmp_path, monkeypatch):
        client, root = org_a_model
        out = self._register(client, tmp_path, monkeypatch, "org-A", version="v2", set_alias=None)
        assert out["organization_id"] == "org-A"

        v3dir = root / "tasks" / "t" / "models" / "m" / "versions" / "v3"
        assert not v3dir.exists()
        self._as_org(monkeypatch, "org-B")
        src = tmp_path / "src_org_b_v3"
        _make_minimal_package(src)
        r = client.post(
            "/v1/register",
            json={"task": "t", "model_name": "m", "version": "v3",
                  "artifacts_dir": str(src), "metadata": {}, "set_alias": None},
            headers={"Authorization": "Bearer org-B"},
        )
        assert r.status_code == 400
        # Security requirement #13: cross-org denial mutates nothing.
        assert not v3dir.exists()
        from omnibioai_model_registry.ownership import read_ownership
        assert read_ownership(root, "t", "m").organization_id == "org-A"

    def test_promote_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.post("/v1/promote", json={
            "task": "t", "model_name": "m", "alias": "staging", "version": "v1",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.post("/v1/promote", json={
            "task": "t", "model_name": "m", "alias": "production", "version": "v1",
        }, headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 400

    def test_tags_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, root = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.put("/v1/tags", json={
            "task": "t", "model_name": "m", "version": "v1", "key": "team", "value": "bioml",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.put("/v1/tags", json={
            "task": "t", "model_name": "m", "version": "v1", "key": "team", "value": "attacker",
        }, headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 404
        meta_path = root / "tasks" / "t" / "models" / "m" / "versions" / "v1" / "model_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta.get("tags", {}).get("team") == "bioml"  # org-B's write never landed

    def test_versions_patch_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, root = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.post("/v1/versions/patch", json={
            "task": "t", "model_name": "m", "version": "v1", "description": "org-A note",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.post("/v1/versions/patch", json={
            "task": "t", "model_name": "m", "version": "v1", "description": "attacker note",
        }, headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 404
        meta_path = root / "tasks" / "t" / "models" / "m" / "versions" / "v1" / "model_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta.get("description") == "org-A note"  # org-B's write never landed

    def test_stage_org_a_allowed_org_b_denied(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.post("/v1/stage", json={
            "task": "t", "model_name": "m", "version": "v1", "stage": "staging",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200

        self._as_org(monkeypatch, "org-B")
        r = client.post("/v1/stage", json={
            "task": "t", "model_name": "m", "version": "v1", "stage": "production",
        }, headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 404

    # ── HF push ──────────────────────────────────────────────────────────────

    def test_hf_push_org_a_allowed(self, org_a_model, monkeypatch):
        import omnibioai_model_registry.hf_routes as hf_mod
        from unittest.mock import MagicMock

        client, _ = org_a_model
        mock_run_push = MagicMock()
        monkeypatch.setattr(hf_mod, "_run_push", mock_run_push)
        self._as_org(monkeypatch, "org-A")
        r = client.post("/v1/hf/push", json={
            "task": "t", "model_name": "m", "version": "v1",
            "repo_id": "org-a-space/model", "token": "hf_faketoken",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        mock_run_push.assert_called_once()

    def test_hf_push_org_b_denied_no_push_started(self, org_a_model, monkeypatch):
        """Security requirements #7/#14: HF push cannot be used to
        exfiltrate another organization's model artifacts, and a denial
        never reaches the HF API at all."""
        import omnibioai_model_registry.hf_routes as hf_mod
        from unittest.mock import MagicMock

        client, _ = org_a_model
        mock_run_push = MagicMock()
        monkeypatch.setattr(hf_mod, "_run_push", mock_run_push)
        self._as_org(monkeypatch, "org-B")
        r = client.post("/v1/hf/push", json={
            "task": "t", "model_name": "m", "version": "v1",
            "repo_id": "attacker-space/exfiltrated", "token": "hf_faketoken",
        }, headers={"Authorization": "Bearer org-B"})
        assert r.status_code == 404
        mock_run_push.assert_not_called()

    def test_hf_push_audit_event_carries_organization_id(self, org_a_model, monkeypatch):
        import omnibioai_model_registry.hf_routes as hf_mod
        from unittest.mock import MagicMock, patch

        client, _ = org_a_model
        monkeypatch.setattr(hf_mod, "_run_push", MagicMock())
        self._as_org(monkeypatch, "org-A")
        with patch.object(hf_mod, "_audit") as mock_audit:
            r = client.post("/v1/hf/push", json={
                "task": "t", "model_name": "m", "version": "v1",
                "repo_id": "org-a-space/model", "token": "hf_faketoken",
            }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        mock_audit.log_event.assert_called_once()
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["metadata"]["organization_id"] == "org-A"

    # ── legacy_unowned: never auto-claimed ──────────────────────────────────

    def _write_legacy_model(self, root, task, model_name):
        from omnibioai_model_registry.ownership import ensure_model_ownership
        from omnibioai_model_registry.storage.localfs import LocalFS

        vdir = root / "tasks" / task / "models" / model_name / "versions" / "v1"
        vdir.mkdir(parents=True)
        for f in REQUIRED_FILES:
            (vdir / f).write_text("{}" if f.endswith(".json") else "x")
        ensure_model_ownership(
            LocalFS(), root, task, model_name,
            organization_id=None, actor=None, model_pre_existing=True,
        )

    def test_legacy_unowned_model_not_auto_claimed_by_authenticated_org(self, auth_client, monkeypatch):
        """Security requirement #9."""
        client, root = auth_client
        self._write_legacy_model(root, "t", "old_model")

        self._as_org(monkeypatch, "org-A")
        r = client.get("/v1/resolve", params={"task": "t", "ref": "old_model@v1"},
                        headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 400  # denied, not silently claimed by org-A

        from omnibioai_model_registry.ownership import read_ownership
        rec = read_ownership(root, "t", "old_model")
        assert rec.status == "legacy_unowned"
        assert rec.organization_id is None  # still unclaimed after the read attempt

    def test_legacy_unowned_model_denied_even_in_open_mode(self, tmp_path, monkeypatch):
        """A None org_id does not match legacy_unowned either -- open-mode
        callers are not a backdoor around the "never auto-claim" rule."""
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)

        import omnibioai_model_registry.service.app.main as _svc
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        self._write_legacy_model(new_reg.root, "t", "old_model")
        client = TestClient(_svc.app, raise_server_exceptions=False)

        r = client.get("/v1/resolve", params={"task": "t", "ref": "old_model@v1"})
        assert r.status_code == 400

    # ── header/query/body spoofing cannot bypass ownership ──────────────────

    def test_spoofed_x_organization_id_header_cannot_bypass_read(self, org_a_model, monkeypatch):
        """Security requirement #10."""
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-B")
        r = client.get(
            "/v1/resolve", params={"task": "t", "ref": "m@v1"},
            headers={"Authorization": "Bearer org-B", "X-Organization-ID": "org-A"},
        )
        assert r.status_code == 400

    def test_query_param_organization_id_cannot_bypass_read(self, org_a_model, monkeypatch):
        """Security requirement #11: no read route has an organization_id
        query parameter at all -- one supplied anyway is simply ignored,
        never consulted for authorization."""
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-B")
        r = client.get(
            "/v1/resolve",
            params={"task": "t", "ref": "m@v1", "organization_id": "org-A"},
            headers={"Authorization": "Bearer org-B"},
        )
        assert r.status_code == 400

    def test_body_organization_id_cannot_bypass_promote(self, org_a_model, monkeypatch):
        """Security requirement #11: PromoteRequest has no organization_id
        field -- FastAPI/pydantic silently drops the unknown key, so it
        can never reach ownership logic."""
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-B")
        r = client.post(
            "/v1/promote",
            json={"task": "t", "model_name": "m", "alias": "production", "version": "v1",
                  "organization_id": "org-A"},
            headers={"Authorization": "Bearer org-B"},
        )
        assert r.status_code == 400

    # ── audit org_id propagation (security requirement #16) ────────────────

    def test_promote_audit_event_carries_organization_id(self, org_a_model, monkeypatch):
        from unittest.mock import patch
        client, _ = org_a_model
        import omnibioai_model_registry.service.app.main as _svc
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_audit") as mock_audit:
            r = client.post("/v1/promote", json={
                "task": "t", "model_name": "m", "alias": "staging", "version": "v1",
            }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["metadata"]["organization_id"] == "org-A"

    def test_tag_audit_event_carries_organization_id(self, org_a_model, monkeypatch):
        from unittest.mock import patch
        client, _ = org_a_model
        import omnibioai_model_registry.service.app.main as _svc
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_audit") as mock_audit:
            r = client.put("/v1/tags", json={
                "task": "t", "model_name": "m", "version": "v1", "key": "k", "value": "v",
            }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["metadata"]["organization_id"] == "org-A"

    def test_patch_version_audit_event_carries_organization_id(self, org_a_model, monkeypatch):
        from unittest.mock import patch
        client, _ = org_a_model
        import omnibioai_model_registry.service.app.main as _svc
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_audit") as mock_audit:
            r = client.post("/v1/versions/patch", json={
                "task": "t", "model_name": "m", "version": "v1", "description": "d",
            }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        mock_audit.log_event.assert_called_once()
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["metadata"]["organization_id"] == "org-A"

    def test_stage_audit_event_carries_organization_id(self, org_a_model, monkeypatch):
        from unittest.mock import patch
        client, _ = org_a_model
        import omnibioai_model_registry.service.app.main as _svc
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_audit") as mock_audit:
            r = client.post("/v1/stage", json={
                "task": "t", "model_name": "m", "version": "v1", "stage": "staging",
            }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["metadata"]["organization_id"] == "org-A"

    # ── same-org / Phase-1 regression guard ─────────────────────────────────

    def test_same_org_full_lifecycle_unchanged(self, org_a_model, monkeypatch):
        """Security requirement #18: owning-org behavior is a pure
        superset of Phase 2A -- register, promote, tag, patch, stage, all
        succeed exactly as before for the org that actually owns the
        model."""
        client, root = org_a_model
        self._as_org(monkeypatch, "org-A")
        r = client.post("/v1/promote", json={
            "task": "t", "model_name": "m", "alias": "staging", "version": "v1",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        r = client.put("/v1/tags", json={
            "task": "t", "model_name": "m", "version": "v1", "key": "k", "value": "v",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        r = client.post("/v1/versions/patch", json={
            "task": "t", "model_name": "m", "version": "v1", "description": "d",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        r = client.post("/v1/stage", json={
            "task": "t", "model_name": "m", "version": "v1", "stage": "production",
        }, headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200


class TestPhase2BConcurrency:
    """Security requirement #19: concurrent registration does not create
    ownership races or duplicate records -- real OS threads, not just the
    single-call write_once_text race already covered in
    TestOwnershipModule."""

    def test_concurrent_brand_new_model_registration_has_one_consistent_winner(
        self, env_root, tmp_path
    ):
        import threading
        from omnibioai_model_registry import register_model
        from omnibioai_model_registry.ownership import read_ownership

        n = 8
        srcs = []
        for i in range(n):
            src = tmp_path / f"src_{i}"
            _make_minimal_package(src)
            srcs.append(src)

        results = [None] * n
        errors = []
        start = threading.Barrier(n)

        def worker(i):
            try:
                start.wait()
                results[i] = register_model(
                    task="t", model_name="m", version=f"v{i}",
                    artifacts_dir=str(srcs[i]), metadata={}, set_alias=None,
                    organization_id=f"org-{i}", actor=f"actor-{i}",
                )
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)

        assert not errors, f"unexpected errors under concurrent registration: {errors}"
        assert all(r is not None for r in results)

        winner_org = read_ownership(env_root, "t", "m").organization_id
        assert winner_org in {f"org-{i}" for i in range(n)}
        # Every thread's own response must agree on who won -- no thread
        # observed a different/duplicate ownership record.
        assert all(r["organization_id"] == winner_org for r in results)
        # All n versions were still written (registration itself is
        # per-version, only the model-level ownership.json is racy/write-
        # once) -- concurrency didn't drop or corrupt any version.
        for i in range(n):
            assert (env_root / "tasks" / "t" / "models" / "m" / "versions" / f"v{i}").exists()


class TestPhase2CHTTPRunTracking:
    """HTTP-level Phase 2C coverage. Unlike the pre-existing
    'cover the handler body' tests in TestServiceAllRoutes (which mock
    tracking.py's functions away entirely), these mock only the DB
    connection/cursor -- the real tracking.py ownership-decision logic
    runs, so cross-org denial is genuinely exercised through the HTTP
    layer end to end, not merely asserted at the unit level."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    def _as_org(self, monkeypatch, org_id, *, permissions=("model.use",)):
        from iam_client.models import UserContext
        return self._mock_iam_client(monkeypatch, UserContext(
            user_id=f"user-{org_id}", email=f"user@{org_id}.example",
            roles=[], permissions=list(permissions), valid=True, org_id=org_id,
        ))

    @pytest.fixture
    def auth_client(self, tmp_path, monkeypatch):
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")

        import omnibioai_model_registry.service.app.main as _svc
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        return TestClient(_svc.app, raise_server_exceptions=False), new_reg.root

    def _fake_cursor(self):
        from unittest.mock import MagicMock

        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False
        return conn, cursor

    def _owned_row(self, org_id="org-A", status="owned"):
        from datetime import datetime, timezone

        return {
            "run_id": "r1", "task": "t", "model_name": "m", "status": "running",
            "started_at": datetime.now(timezone.utc).replace(tzinfo=None), "finished_at": None,
            "actor": None, "organization_id": org_id, "ownership_status": status,
        }

    # ── log-metric: create + cross-org denial ───────────────────────────────

    def test_log_metric_org_a_creates_new_run(self, auth_client, monkeypatch):
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = None  # genuinely new run_id
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.post(
                "/v1/runs/log-metric",
                json={"task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.9, "step": 0},
                headers={"Authorization": "Bearer org-A"},
            )
        assert r.status_code == 200
        inserts = [c for c in cursor.execute.call_args_list if "INSERT IGNORE" in str(c[0][0])]
        assert inserts[0][0][1][-2:] == ("org-A", "owned")

    def test_log_metric_org_b_denied_on_org_a_run(self, auth_client, monkeypatch):
        """Security requirements #1/#6: org B cannot append metrics to
        org A's run."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        self._as_org(monkeypatch, "org-B")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.post(
                "/v1/runs/log-metric",
                json={"task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.1, "step": 0},
                headers={"Authorization": "Bearer org-B"},
            )
        assert r.status_code == 400
        metric_inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_metrics" in str(c[0][0])]
        assert metric_inserts == []

    def test_log_param_org_b_denied_on_org_a_run(self, auth_client, monkeypatch):
        """Security requirement #2: org B cannot append params to org A's
        run."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        self._as_org(monkeypatch, "org-B")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.post(
                "/v1/runs/log-param",
                json={"task": "t", "model_name": "m", "run_id": "r1", "key": "lr", "value": 0.01},
                headers={"Authorization": "Bearer org-B"},
            )
        assert r.status_code == 400
        param_inserts = [c for c in cursor.execute.call_args_list if "INSERT INTO omr_params" in str(c[0][0])]
        assert param_inserts == []

    def test_log_batch_org_b_denied_on_org_a_run_nothing_written(self, auth_client, monkeypatch):
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        self._as_org(monkeypatch, "org-B")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.post(
                "/v1/runs/log-batch",
                json={
                    "task": "t", "model_name": "m", "run_id": "r1",
                    "metrics": [{"key": "acc", "value": 0.1, "step": 0}],
                    "params": {"lr": 0.01}, "tags": {"team": "attacker"},
                },
                headers={"Authorization": "Bearer org-B"},
            )
        assert r.status_code == 400
        mutating = [
            c for c in cursor.execute.call_args_list
            if any(t in str(c[0][0]) for t in ("INSERT INTO omr_metrics", "INSERT INTO omr_params", "INSERT INTO omr_tags"))
        ]
        assert mutating == []

    def test_log_metric_same_org_continues_to_work(self, auth_client, monkeypatch):
        """Security requirement #4: org A can keep reading/writing its own
        run."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.post(
                "/v1/runs/log-metric",
                json={"task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.95, "step": 1},
                headers={"Authorization": "Bearer org-A"},
            )
        assert r.status_code == 200

    # ── runs/get: cross-org denial is an anti-enumeration oracle ───────────

    def test_runs_get_org_b_denied_same_shape_as_missing_run(self, auth_client, monkeypatch):
        """Security requirement #3: a cross-org run_id must not become an
        enumeration oracle -- same status/response shape as a genuinely
        missing run_id."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc

        conn1, cursor1 = self._fake_cursor()
        cursor1.fetchone.return_value = self._owned_row(org_id="org-A")
        self._as_org(monkeypatch, "org-B")
        with patch.object(_svc, "_get_db_conn", return_value=conn1):
            r_other_org = client.get(
                "/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"},
                headers={"Authorization": "Bearer org-B"},
            )

        conn2, cursor2 = self._fake_cursor()
        cursor2.fetchone.return_value = None
        with patch.object(_svc, "_get_db_conn", return_value=conn2):
            r_missing = client.get(
                "/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"},
                headers={"Authorization": "Bearer org-B"},
            )

        assert r_other_org.status_code == r_missing.status_code == 400
        assert r_other_org.json() == r_missing.json()
        # No params/tags/metrics query ever issued once ownership denies.
        extra = [
            c for c in cursor1.execute.call_args_list
            if any(t in str(c[0][0]) for t in ("FROM omr_params", "FROM omr_tags", "FROM omr_metrics"))
        ]
        assert extra == []

    def test_runs_get_legacy_unowned_denied_for_real_org(self, auth_client, monkeypatch):
        """Security requirement #10 (legacy/unowned policy)."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = self._owned_row(org_id=None, status="legacy_unowned")
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.get(
                "/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"},
                headers={"Authorization": "Bearer org-A"},
            )
        assert r.status_code == 400

    def test_runs_get_same_org_allowed(self, auth_client, monkeypatch):
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = self._owned_row(org_id="org-A")
        cursor.fetchall.return_value = []
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.get(
                "/v1/runs/get", params={"task": "t", "model": "m", "run_id": "r1"},
                headers={"Authorization": "Bearer org-A"},
            )
        assert r.status_code == 200
        assert r.json()["run_id"] == "r1"

    # ── runs/list: query-layer org filtering ────────────────────────────────

    def test_runs_list_applies_org_filter_at_query_layer(self, auth_client, monkeypatch):
        """Security requirement: cross-org task/model lookup is blocked --
        the SQL itself is scoped to the caller's org, not filtered after
        fetching everyone's runs."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchall.return_value = []
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.get(
                "/v1/runs/list", params={"task": "t", "model": "m"},
                headers={"Authorization": "Bearer org-A"},
            )
        assert r.status_code == 200
        sql, params = cursor.execute.call_args[0]
        assert "organization_id" in sql
        assert "org-A" in params

    # ── header/query/body spoofing cannot bypass ─────────────────────────────

    def test_spoofed_x_organization_id_header_cannot_bypass_log_metric(self, auth_client, monkeypatch):
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = {"organization_id": "org-A", "ownership_status": "owned"}
        self._as_org(monkeypatch, "org-B")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.post(
                "/v1/runs/log-metric",
                json={"task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.1, "step": 0},
                headers={"Authorization": "Bearer org-B", "X-Organization-ID": "org-A", "X-Team-ID": "team-A"},
            )
        assert r.status_code == 400

    def test_query_param_organization_id_cannot_bypass_runs_get(self, auth_client, monkeypatch):
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = self._owned_row(org_id="org-A")
        self._as_org(monkeypatch, "org-B")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.get(
                "/v1/runs/get",
                params={"task": "t", "model": "m", "run_id": "r1", "organization_id": "org-A"},
                headers={"Authorization": "Bearer org-B"},
            )
        assert r.status_code == 400

    def test_body_organization_id_cannot_bypass_log_metric(self, auth_client, monkeypatch):
        """A spoofed body organization_id must not even influence which
        org a brand-new run gets attributed to."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = None
        self._as_org(monkeypatch, "org-B")
        with patch.object(_svc, "_get_db_conn", return_value=conn):
            r = client.post(
                "/v1/runs/log-metric",
                json={
                    "task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.1, "step": 0,
                    "organization_id": "org-A",
                },
                headers={"Authorization": "Bearer org-B"},
            )
        assert r.status_code == 200
        inserts = [c for c in cursor.execute.call_args_list if "INSERT IGNORE" in str(c[0][0])]
        assert inserts[0][0][1][-2:] == ("org-B", "owned")  # real JWT org, not the spoofed body field

    # ── audit propagation ─────────────────────────────────────────────────

    def test_log_metric_audit_carries_organization_id(self, auth_client, monkeypatch):
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = None
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_audit") as mock_audit:
            with patch.object(_svc, "_get_db_conn", return_value=conn):
                r = client.post(
                    "/v1/runs/log-metric",
                    json={"task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.9, "step": 0},
                    headers={"Authorization": "Bearer org-A"},
                )
        assert r.status_code == 200
        mock_audit.log_event.assert_called_once()
        _, kwargs = mock_audit.log_event.call_args
        assert kwargs["metadata"]["organization_id"] == "org-A"

    def test_log_param_audit_never_includes_value(self, auth_client, monkeypatch):
        """Params are caller-supplied Any and could contain something
        sensitive -- only the key name is ever audited."""
        from unittest.mock import patch
        client, _ = auth_client
        import omnibioai_model_registry.service.app.main as _svc
        conn, cursor = self._fake_cursor()
        cursor.fetchone.return_value = None
        self._as_org(monkeypatch, "org-A")
        with patch.object(_svc, "_audit") as mock_audit:
            with patch.object(_svc, "_get_db_conn", return_value=conn):
                r = client.post(
                    "/v1/runs/log-param",
                    json={"task": "t", "model_name": "m", "run_id": "r1", "key": "api_key", "value": "sk-super-secret"},
                    headers={"Authorization": "Bearer org-A"},
                )
        assert r.status_code == 200
        _, kwargs = mock_audit.log_event.call_args
        assert "sk-super-secret" not in json.dumps(kwargs["metadata"])
        assert kwargs["metadata"]["organization_id"] == "org-A"
        assert kwargs["metadata"]["key"] == "api_key"

    # ── Phase 1 auth still required on the write routes ─────────────────────

    def test_log_metric_without_authorization_returns_401(self, auth_client):
        client, _ = auth_client
        r = client.post("/v1/runs/log-metric", json={
            "task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.1, "step": 0,
        })
        assert r.status_code == 401

    def test_log_param_without_authorization_returns_401(self, auth_client):
        client, _ = auth_client
        r = client.post("/v1/runs/log-param", json={
            "task": "t", "model_name": "m", "run_id": "r1", "key": "lr", "value": 0.1,
        })
        assert r.status_code == 401

    def test_log_batch_without_authorization_returns_401(self, auth_client):
        client, _ = auth_client
        r = client.post("/v1/runs/log-batch", json={"task": "t", "model_name": "m", "run_id": "r1"})
        assert r.status_code == 401

    # ── filesystem-only mode (DB_HOST unset) remains functional ─────────────

    def test_runs_log_metric_returns_503_without_db_configured(self, auth_client, monkeypatch):
        """Security requirement: filesystem-only deployments remain
        completely functional -- a DB-backed route degrades to 503
        (pre-existing behavior), it does not error or bypass auth."""
        monkeypatch.delenv("DB_HOST", raising=False)
        client, _ = auth_client
        self._as_org(monkeypatch, "org-A")
        r = client.post(
            "/v1/runs/log-metric",
            json={"task": "t", "model_name": "m", "run_id": "r1", "key": "acc", "value": 0.1, "step": 0},
            headers={"Authorization": "Bearer org-A"},
        )
        assert r.status_code == 503

    def test_filesystem_only_mode_register_and_resolve_unaffected(self, tmp_path, monkeypatch):
        """The core filesystem model registry (Phase 2A/2B) is completely
        untouched by Phase 2C -- open mode, no DB at all."""
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        monkeypatch.delenv("DB_HOST", raising=False)

        import omnibioai_model_registry.service.app.main as _svc
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        client = TestClient(_svc.app, raise_server_exceptions=False)

        src = tmp_path / "src"
        _make_minimal_package(src)
        r = client.post("/v1/register", json={
            "task": "t", "model_name": "m", "version": "v1",
            "artifacts_dir": str(src), "metadata": {},
        })
        assert r.status_code == 200
        r = client.get("/v1/resolve", params={"task": "t", "ref": "m@v1"})
        assert r.status_code == 200


# ============================================================
# Phase 2D — legacy ownership audit & final tenant-isolation hardening
# ============================================================
#
# Step 3's audit (see PR description) found no IAM signal that fits this
# repo's own "authorize by permission, never by role" precedent for a
# legacy-ownership-resolution endpoint, so none was built -- see the PR
# description for the two candidate designs considered and rejected in
# favor of an explicit product/IAM decision. What Phase 2D actually
# changes in code is narrow: GET /v1/hf/push/status/{job_id} (Phase 1
# authenticated only, explicitly flagged in hf_routes.py since Phase 1 as
# "deferred to the tenant-isolation phase") now enforces the same
# organization boundary every other route already does. This class
# covers that; TestPhase2DFinalConsistencyAudit below is a consolidated,
# single-pass regression guard across every route Phase 2A-2C already
# protect, run once more here as the final tenant-isolation audit this
# phase is about.


class TestPhase2DHFPushStatusOwnership:
    """GET /v1/hf/push/status/{job_id} -- org-scoped as of Phase 2D."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    def _as_org(self, monkeypatch, org_id, *, permissions=("model.use",)):
        from iam_client.models import UserContext
        return self._mock_iam_client(monkeypatch, UserContext(
            user_id=f"user-{org_id}" if org_id else "user-open",
            email=f"user@{org_id}.example" if org_id else "user@open.example",
            roles=[], permissions=list(permissions), valid=True, org_id=org_id,
        ))

    @pytest.fixture
    def auth_client(self, tmp_path, monkeypatch):
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")

        import omnibioai_model_registry.service.app.main as _svc
        import omnibioai_model_registry.hf_routes as _hf
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        monkeypatch.setattr(_hf, "_registry", new_reg)

        class _SyncThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None):
                self._target = target
                self._args = args
                self._kwargs = kwargs or {}

            def start(self):
                self._target(*self._args, **self._kwargs)

        class _FakeThreadingModule:
            Thread = _SyncThread

        monkeypatch.setattr(_hf, "threading", _FakeThreadingModule())

        return TestClient(_svc.app, raise_server_exceptions=False), new_reg.root

    def _push_and_get_job_id(self, client, tmp_path, monkeypatch, org_id, *, task="t", model_name="m", version="v1"):
        import omnibioai_model_registry.hf_routes as hf_mod
        from unittest.mock import MagicMock

        self._as_org(monkeypatch, org_id)
        src = tmp_path / f"src_{org_id}_{task}_{model_name}_{version}"
        _make_minimal_package(src)
        r = client.post(
            "/v1/register",
            json={"task": task, "model_name": model_name, "version": version,
                  "artifacts_dir": str(src), "metadata": {}, "set_alias": None},
            headers={"Authorization": f"Bearer {org_id or 'open'}"},
        )
        assert r.status_code == 200, r.text

        monkeypatch.setattr(hf_mod, "_run_push", MagicMock())
        self._as_org(monkeypatch, org_id)
        r = client.post(
            "/v1/hf/push",
            json={"task": task, "model_name": model_name, "version": version,
                  "repo_id": "some/repo", "token": "hf_faketoken"},
            headers={"Authorization": f"Bearer {org_id or 'open'}"},
        )
        assert r.status_code == 200, r.text
        return r.json()["job_id"]

    def test_org_a_can_poll_its_own_job(self, auth_client, tmp_path, monkeypatch):
        client, _ = auth_client
        job_id = self._push_and_get_job_id(client, tmp_path, monkeypatch, "org-A")
        self._as_org(monkeypatch, "org-A")
        r = client.get(f"/v1/hf/push/status/{job_id}", headers={"Authorization": "Bearer org-A"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_org_b_cannot_poll_org_a_job_same_shape_as_unknown(self, auth_client, tmp_path, monkeypatch):
        """Security requirement: no enumeration oracle -- a cross-org
        job_id and a genuinely unknown job_id must be indistinguishable."""
        client, _ = auth_client
        job_id = self._push_and_get_job_id(client, tmp_path, monkeypatch, "org-A")

        self._as_org(monkeypatch, "org-B")
        r_cross_org = client.get(f"/v1/hf/push/status/{job_id}", headers={"Authorization": "Bearer org-B"})
        r_unknown = client.get(
            "/v1/hf/push/status/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": "Bearer org-B"},
        )
        assert r_cross_org.status_code == r_unknown.status_code == 404
        assert r_cross_org.json() == r_unknown.json()

    def test_spoofed_x_organization_id_header_cannot_bypass_push_status(self, auth_client, tmp_path, monkeypatch):
        client, _ = auth_client
        job_id = self._push_and_get_job_id(client, tmp_path, monkeypatch, "org-A")
        self._as_org(monkeypatch, "org-B")
        r = client.get(
            f"/v1/hf/push/status/{job_id}",
            headers={"Authorization": "Bearer org-B", "X-Organization-ID": "org-A"},
        )
        assert r.status_code == 404

    def test_open_mode_job_pollable_only_in_open_mode(self, auth_client, tmp_path, monkeypatch):
        """A job created with no org context (AUTH_ENABLED technically on
        here, but the identity itself carries org_id=None) is pollable by
        another org_id=None identity -- matches every other resource's
        'both sides have no org context' precedent -- but not by a real
        org."""
        client, _ = auth_client
        job_id = self._push_and_get_job_id(client, tmp_path, monkeypatch, None)

        self._as_org(monkeypatch, None)
        r_open = client.get(f"/v1/hf/push/status/{job_id}", headers={"Authorization": "Bearer open"})
        assert r_open.status_code == 200

        self._as_org(monkeypatch, "org-A")
        r_real_org = client.get(f"/v1/hf/push/status/{job_id}", headers={"Authorization": "Bearer org-A"})
        assert r_real_org.status_code == 404

    def test_push_status_without_authorization_returns_401(self, auth_client):
        """Phase 1 authentication requirement is preserved, not weakened,
        by the Phase 2D org-scoping change."""
        client, _ = auth_client
        r = client.get("/v1/hf/push/status/some-job-id")
        assert r.status_code == 401


class TestPhase2DFinalConsistencyAudit:
    """Step 4's final route/resource consistency pass, run as one
    consolidated regression guard: org-B must be denied on every
    protected route against org-A's model/run in a single pass, using
    exactly the same JWT-derived org_id every other Phase 2A-2C test
    already relies on -- no gateway header, no body/query field, no
    alternate lookup path."""

    def _mock_iam_client(self, monkeypatch, user_context):
        from unittest.mock import AsyncMock, MagicMock
        import omnibioai_model_registry.auth as auth_mod

        mock_client = MagicMock()
        mock_client.get_user = AsyncMock(return_value=user_context)
        mock_client.http.aclose = AsyncMock()
        monkeypatch.setattr(auth_mod, "AsyncIAMClient", MagicMock(return_value=mock_client))
        return mock_client

    def _as_org(self, monkeypatch, org_id, *, permissions=("model.use",)):
        from iam_client.models import UserContext
        return self._mock_iam_client(monkeypatch, UserContext(
            user_id=f"user-{org_id}", email=f"user@{org_id}.example",
            roles=[], permissions=list(permissions), valid=True, org_id=org_id,
        ))

    @pytest.fixture
    def org_a_model(self, tmp_path, monkeypatch):
        root = tmp_path / "registry"
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
        monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "0")
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("JWT_SECRET", "testsecret")

        import omnibioai_model_registry.service.app.main as _svc
        import omnibioai_model_registry.hf_routes as _hf
        from fastapi.testclient import TestClient

        new_reg = _svc.ModelRegistry.from_env()
        monkeypatch.setattr(_svc, "registry", new_reg)
        monkeypatch.setattr(_hf, "_registry", new_reg)
        client = TestClient(_svc.app, raise_server_exceptions=False)

        self._as_org(monkeypatch, "org-A")
        src = tmp_path / "src"
        _make_minimal_package(src)
        r = client.post(
            "/v1/register",
            json={"task": "t", "model_name": "m", "version": "v1",
                  "artifacts_dir": str(src), "metadata": {}, "set_alias": "latest"},
            headers={"Authorization": "Bearer org-A"},
        )
        assert r.status_code == 200
        return client, new_reg.root

    def test_every_model_route_denies_org_b(self, org_a_model, monkeypatch):
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-B")
        headers = {"Authorization": "Bearer org-B"}

        checks = [
            ("resolve", client.get("/v1/resolve", params={"task": "t", "ref": "m@v1"}, headers=headers), 400),
            ("show", client.get("/v1/show", params={"task": "t", "ref": "m@v1"}, headers=headers), 400),
            ("verify", client.post("/v1/verify", json={"task": "t", "ref": "m@v1"}, headers=headers), 400),
            ("artifacts", client.get("/v1/artifacts", params={"task": "t", "ref": "m@v1"}, headers=headers), 400),
            ("metrics", client.get("/v1/metrics", params={"task": "t", "ref": "m@v1"}, headers=headers), 400),
            ("aliases", client.get("/v1/aliases", params={"task": "t", "model": "m"}, headers=headers), 404),
            ("compare", client.get("/v1/compare", params={"task": "t", "model": "m", "versions": ["v1", "v1"]}, headers=headers), 404),
            ("promote", client.post("/v1/promote", json={"task": "t", "model_name": "m", "alias": "staging", "version": "v1"}, headers=headers), 400),
            ("tags", client.put("/v1/tags", json={"task": "t", "model_name": "m", "version": "v1", "key": "k", "value": "v"}, headers=headers), 404),
            ("versions/patch", client.post("/v1/versions/patch", json={"task": "t", "model_name": "m", "version": "v1", "description": "x"}, headers=headers), 404),
            ("stage", client.post("/v1/stage", json={"task": "t", "model_name": "m", "version": "v1", "stage": "staging"}, headers=headers), 404),
        ]
        failures = [(name, resp.status_code, expected) for name, resp, expected in checks if resp.status_code != expected]
        assert failures == [], f"routes that didn't deny org-B as expected: {failures}"

        # /v1/models: org-B's list must not contain org-A's model at all.
        r = client.get("/v1/models", headers=headers)
        assert r.status_code == 200
        assert "m" not in {m.get("model_name") for m in r.json()}

    def test_every_model_route_still_works_for_org_a(self, org_a_model, monkeypatch):
        """The other half of the final audit: none of the above denials
        are a blanket regression -- org-A's own access is fully intact."""
        client, _ = org_a_model
        self._as_org(monkeypatch, "org-A")
        headers = {"Authorization": "Bearer org-A"}

        for name, resp in [
            ("resolve", client.get("/v1/resolve", params={"task": "t", "ref": "m@v1"}, headers=headers)),
            ("show", client.get("/v1/show", params={"task": "t", "ref": "m@v1"}, headers=headers)),
            ("artifacts", client.get("/v1/artifacts", params={"task": "t", "ref": "m@v1"}, headers=headers)),
            ("aliases", client.get("/v1/aliases", params={"task": "t", "model": "m"}, headers=headers)),
            ("models", client.get("/v1/models", headers=headers)),
        ]:
            assert resp.status_code == 200, f"{name} unexpectedly denied for the owning org: {resp.text}"
