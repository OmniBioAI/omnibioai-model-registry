# File: omnibioai_model_registry/errors.py
class ModelRegistryError(Exception):
    pass


class RegistryNotConfigured(ModelRegistryError):
    pass


class ModelNotFound(ModelRegistryError):
    pass


class VersionAlreadyExists(ModelRegistryError):
    pass


class InvalidModelRef(ModelRegistryError):
    pass


class IntegrityError(ModelRegistryError):
    pass


class ValidationError(ModelRegistryError):
    pass


class InvalidStageTransition(ModelRegistryError):
    """Raised when an invalid stage transition is attempted."""
    pass
