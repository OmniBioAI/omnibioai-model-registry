# File: omnibioai_model_registry/storage/localfs.py
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .base import StorageBackend


class LocalFS(StorageBackend):
    def ensure_dirs(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def exists(self, path: Path) -> bool:
        return path.exists()

    def copy_tree(self, src_dir: Path, dst_dir: Path) -> None:
        # shutil.copytree requires dst doesn't exist
        shutil.copytree(src_dir, dst_dir)

    def atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass

    def write_once_text(self, path: Path, text: str) -> bool:
        """Exclusive create: write to a temp file in the same directory,
        then os.link() it into place. os.link() is atomic and raises
        FileExistsError if `path` already exists, so two racing callers
        (e.g. two requests registering the same brand-new model at once)
        can never both "win" -- exactly one creates the file, the other
        observes it already exists and its own write is discarded. Unlike
        atomic_write_text's os.replace, this never overwrites."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            try:
                os.link(tmp, path)
                return True
            except FileExistsError:
                return False
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass
