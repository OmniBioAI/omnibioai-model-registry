"""
conftest.py — root-level pytest configuration.

Must run before any test module is imported so that the Cython .so extensions
(manifest, validate, localfs) are replaced with the pure-Python source files.

WHY MetaPathFinder instead of sys.modules pre-seeding:
  When a module is already in sys.modules, Python's _find_and_load() returns
  immediately, skipping the code that does setattr(parent, child, module).
  This means `import omnibioai_model_registry.storage.localfs as lfs_mod`
  would succeed for the module lookup but then fail on the IMPORT_FROM
  bytecode that does getattr(omnibioai_model_registry, 'storage') because
  `storage` was never set as an attribute on the package object.

  A MetaPathFinder lets Python run the full _find_and_load_unlocked() path,
  which imports parent packages normally, then calls our loader, and finally
  does setattr(parent, child, module) — so all attribute links are wired up.

WHY exec real .py source files instead of hand-crafted mocks:
  Loading the actual manifest.py, validate.py, and localfs.py source files
  via spec_from_file_location gives real code coverage on those files while
  still intercepting the import BEFORE the .so Cython extension can load.
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
    def find_spec(self, fullname: str, path, target=None):  # noqa: ANN001
        if fullname in _PY_SOURCES:
            loader = _MockLoader(fullname)
            return importlib.util.spec_from_loader(fullname, loader)
        return None


# Insert at position 0 so we beat the default file finders (which would load
# the .so files) and the .so files in the package directory.
sys.meta_path.insert(0, _MockFinder())
