# File: omnibioai_model_registry/path_safety.py
from __future__ import annotations

"""
Centralized filesystem-path safety for every caller-supplied identifier
this service turns into a path component: task, model_name, version,
alias, run_id, metric_key.

SECURITY INVARIANT
-------------------
No user-controlled model/task/version/alias/run/metric identifier may
cause any filesystem operation to resolve outside the configured
OMNIBIOAI_MODEL_REGISTRY_ROOT, unless that operation is an explicitly
documented, trusted administrative path (the CLI, which already has
direct filesystem access and is not reachable over HTTP/JWT -- the same
trust boundary Phase 2A's `omr register --org-id` precedent already
established).

DESIGN: allowlist, not blocklist
---------------------------------
Rejecting known-bad patterns one at a time (`..`, absolute paths,
encoded variants, backslashes, repeated separators, ...) is inherently
incomplete -- there is always another variant. Instead, every identifier
must match a strict allowlist of characters that can never form a path
separator, a traversal sequence, or an absolute-path prefix on any
platform this service might run on. A component that matches the
allowlist is *lexically* incapable of escaping the directory it's
joined into, regardless of filesystem/OS/normalization quirks -- a
stronger guarantee than "reject '..' after the fact", and checked
BEFORE the value is ever joined into a Path at all.

WHERE THIS IS APPLIED
-----------------------
Once, centrally, inside package/layout.py's path-builder functions --
the single choke point every filesystem path in this codebase is
already constructed through (task_root/model_root/version_dir/
alias_path/run_dir/... -- confirmed by inspection: no other module
builds a task/model/version/alias/run path by hand). Every consumer
(api.py, ownership.py, service/app/main.py, run.py, cli/main.py)
inherits this automatically, with no per-call-site change needed.

assert_contained() below is a second, independent layer used at the
handful of actual mutating operations in api.py (register_model's
copy_tree destination, promote_model's alias write, resolve_model's
returned version dir) -- defense in depth against a hypothetical future
bug in this module itself, not the primary defense. It is symlink-AWARE
(follows them via Path.resolve(), the same mechanism
config.py/run.py's own `_resolve_registry_root` already uses for
`registry_root` itself), not symlink-HOSTILE: a symlinked registry root
or intermediate directory is followed, not rejected, as long as the
real, final location is still under the real, resolved root. This
service does not itself create any symlinks anywhere in the tree it
manages (confirmed by inspection), so there is no legitimate in-repo
symlink layout this needs to special-case -- but the check still never
assumes symlinks are inherently forbidden.
"""

import re
from pathlib import Path

from .errors import PathTraversalError

# Letters, digits, underscore, hyphen, and interior dots; must start and
# end with an alphanumeric character. That last constraint is what does
# the real work: a lone "." or ".." (or any leading-dot hidden-file
# trick) can never satisfy "starts with alnum, ends with alnum" -- a
# single "." isn't in [A-Za-z0-9] itself, and ".." would need the
# component to start with '.', which the pattern forbids. There is no
# '/', '\\', NUL, or other separator/control character anywhere in the
# allowed set, so absolute paths, traversal sequences, repeated
# separators, and backslash variants are all rejected as a natural
# consequence of the allowlist, not via separate rules for each. Max 128
# characters -- generously above every identifier this system's own
# docs/tests ever use (e.g. "human_pbmc", "2026-02-13_001", "staging").
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")


def safe_component(value: str, field_name: str) -> str:
    """Validates a single path component -- never a full path; this is
    only ever called with one caller-supplied identifier at a time (a
    task name, a model name, a version, an alias, a run_id, or a metric
    key). Returns `value` unchanged if safe.

    Raises PathTraversalError otherwise. Callers let this propagate:
    every HTTP route either already wraps its filesystem calls in the
    existing ModelRegistryError-catching handler (_handle_registry_error
    in service/app/main.py), or that module's global exception handler
    for PathTraversalError catches it directly -- so an invalid
    identifier always surfaces as a safe, generic HTTP 400, never a
    path, a stack trace, or any other internal detail. The error message
    names only the *field*, never the (potentially crafted) value or any
    resolved filesystem path.
    """
    if not isinstance(value, str) or not value:
        raise PathTraversalError(f"Invalid {field_name}")
    if not _SAFE_IDENTIFIER_RE.match(value):
        raise PathTraversalError(f"Invalid {field_name}")
    return value


def assert_contained(path: Path, root: Path, *, field_name: str = "path") -> Path:
    """Defense-in-depth: re-verifies `path` resolves to somewhere under
    `root` on the real, symlink-resolved filesystem, independent of
    safe_component() already having made this lexically impossible for
    any caller-supplied identifier. Works for paths that don't exist yet
    (Path.resolve() with strict=False, the default, resolves as much of
    the path as exists and lexically joins the rest -- correct for both
    a read of an existing version and a write of a brand-new one).

    Returns `path` unchanged (the original, not the resolved form -- the
    resolved form is server-filesystem detail that must never reach an
    HTTP response) if contained. Raises PathTraversalError otherwise.
    """
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise PathTraversalError(f"Invalid {field_name}") from None
    return path
