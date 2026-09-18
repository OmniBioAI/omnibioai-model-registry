"""Root-level pytest configuration. Installs a MetaPathFinder before any test module is
imported so the Cython .so extensions for manifest, validate and localfs are replaced
with their pure-Python source files: a MetaPathFinder runs the full import machinery
(including setattr(parent, child, module) on the package), which sys.modules pre-seeding
would skip. It execs the real .py source via spec_from_file_location, giving real
coverage on those files while still intercepting the import before the .so can load.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# MetaPathFinder — intercepts the three Cython modules BEFORE the .so can load
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
_PY_SOURCES: dict = {
    "omnibioai_model_registry.package.manifest":
        _ROOT / "omnibioai_model_registry" / "package" / "manifest.py",
    "omnibioai_model_registry.package.validate":
        _ROOT / "omnibioai_model_registry" / "package" / "validate.py",
    "omnibioai_model_registry.storage.localfs":
        _ROOT / "omnibioai_model_registry" / "storage" / "localfs.py",
}


class _MockLoader(importlib.abc.Loader):
    """Loads the real .py source file instead of the compiled .so extension."""

    def __init__(self, fullname: str) -> None:
        self._fullname = fullname

    def create_module(self, spec):  # noqa: ANN001
        return None  # let Python create a standard ModuleType

    def exec_module(self, module) -> None:  # noqa: ANN001
        py_path = _PY_SOURCES[self._fullname]
        inner_spec = importlib.util.spec_from_file_location(
            self._fullname, str(py_path)
        )
        inner_spec.loader.exec_module(module)


class _MockFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that redirects imports of the three Cython modules in _PY_SOURCES
    to _MockLoader; every other import is left alone."""
    def find_spec(self, fullname: str, path, target=None):  # noqa: ANN001
        if fullname in _PY_SOURCES:
            loader = _MockLoader(fullname)
            return importlib.util.spec_from_loader(fullname, loader)
        return None


# Insert at position 0 so we beat the default file finders (which would load
# the .so files) and the .so files in the package directory.
sys.meta_path.insert(0, _MockFinder())
