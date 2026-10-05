# File: omnibioai_model_registry/package/manifest.py
"""
OmniBioAI omnibioai_model_registry.package.manifest.

Purpose:
    Defines sha256_file, write_sha256_manifest, read_sha256_manifest and verify_sha256_manifest for omnibioai_model_registry.package.manifest.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

from ..errors import IntegrityError


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_sha256_manifest(
    dir_path: Path, manifest_path: Path, include_files: list[str]
) -> Dict[str, str]:
    """
    Writes a manifest like:
      <sha256>  model.pt
    Returns mapping filename -> sha256
    NOTE: We never hash the manifest file itself (sha256sums.txt) to avoid self-referential mismatch.
    """
    hashes: Dict[str, str] = {}
    lines = []
    manifest_name = Path(manifest_path).name  # usually "sha256sums.txt"

    for name in include_files:
        # Never include the manifest itself
        if name == manifest_name:
            continue

        p = dir_path / name
        if not p.exists():
            continue
        digest = sha256_file(p)
        hashes[name] = digest
        lines.append(f"{digest}  {name}")

    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return hashes


def read_sha256_manifest(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        raise IntegrityError(f"Missing manifest: {manifest_path}")
    out: Dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # "<hash>  <filename>"
        parts = line.split()
        if len(parts) < 2:
            continue
        digest = parts[0]
        name = parts[-1]
        out[name] = digest
    return out


def verify_sha256_manifest(dir_path: Path, manifest_path: Path) -> None:
    expected = read_sha256_manifest(manifest_path)
    for name, exp_digest in expected.items():
        p = dir_path / name
        if not p.exists():
            raise IntegrityError(f"Manifest expects file missing: {name}")
        got = sha256_file(p)
        if got != exp_digest:
            raise IntegrityError(
                f"SHA256 mismatch for {name}: expected {exp_digest}, got {got}"
            )
