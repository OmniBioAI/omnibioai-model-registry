# File: omnibioai_model_registry/storage/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """
    Minimal storage backend interface for v0.1.0.

    LocalFS implements this today.
    Future backends: S3, Azure Blob, etc.
    """

    @abstractmethod
    def ensure_dirs(self, path: Path) -> None: ...

    @abstractmethod
    def exists(self, path: Path) -> bool: ...

    @abstractmethod
    def copy_tree(self, src_dir: Path, dst_dir: Path) -> None: ...

    @abstractmethod
    def atomic_write_text(self, path: Path, text: str) -> None: ...

    @abstractmethod
    def write_once_text(self, path: Path, text: str) -> bool:
        """Create `path` with `text` only if it does not already exist.

        Unlike atomic_write_text (which always replaces), this never
        overwrites existing content and is safe under concurrent callers.
        Returns True if this call created the file, False if it already
        existed (existing content left untouched). Introduced in Phase 2A
        for write-once ownership records."""
        ...
