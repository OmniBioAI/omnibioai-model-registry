# File: omnibioai_model_registry/errors.py
class ModelRegistryError(Exception):
    pass


class RegistryNotConfigured(ModelRegistryError):
    pass


class ModelNotFound(ModelRegistryError):
    pass


class VersionAlreadyExists(ModelRegistryError):
    pass


class OwnershipResolutionNotEligible(ModelRegistryError):
    """Phase 2E: raised by ownership.resolve_legacy_ownership() when the
    target model is not eligible for legacy-ownership resolution --
    either it is already owned by a DIFFERENT organization (ownership is
    never reassigned), or its status is "unowned" (a real, already-
    established state from the open/no-org dev-test mode, not an
    orphaned record -- resolving it into a specific org would silently
    narrow who can already access it, a materially different operation
    from resolving a genuinely orphaned legacy_unowned record, and out
    of scope here). Already owned by the CALLER's own organization is
    NOT an error -- see resolve_legacy_ownership()'s idempotent-success
    return path, which returns normally instead of raising this."""
    pass


class InvalidModelRef(ModelRegistryError):
    pass


class IntegrityError(ModelRegistryError):
    pass


class ValidationError(ModelRegistryError):
    pass


class PathTraversalError(ValidationError):
    """Raised by path_safety.py when a caller-supplied identifier (task,
    model_name, version, alias, run_id, metric_key, ...) is not safe to
    use as a filesystem path component, or when a constructed path fails
    its final containment check against the configured registry root.
    A ValidationError (and therefore a ModelRegistryError) so every
    existing `except Exception: _handle_registry_error(e)` call site
    already maps it to a safe, generic HTTP 400 with no code changes;
    service/app/main.py also registers an explicit FastAPI exception
    handler for it as a second layer, for the handful of routes that
    construct paths outside of any try/except (see that module)."""
    pass


class InvalidStageTransition(ModelRegistryError):
    """Raised when an invalid stage transition is attempted."""
    pass
