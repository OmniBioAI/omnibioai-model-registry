# File: omnibioai_model_registry/refs.py
from dataclasses import dataclass

from .errors import InvalidModelRef, PathTraversalError
from .path_safety import safe_component


@dataclass(frozen=True)
class ModelRef:
    model_name: str
    selector: str  # alias or version


def parse_model_ref(model_ref: str) -> ModelRef:
    """
    Accepts:
      - "human_pbmc@production"
      - "human_pbmc@2026-02-13_001"
    """
    if not model_ref or "@" not in model_ref:
        raise InvalidModelRef(
            f"Invalid model_ref '{model_ref}'. Expected '<model_name>@<alias_or_version>'."
        )
    model_name, selector = model_ref.split("@", 1)
    model_name = model_name.strip()
    selector = selector.strip()
    if not model_name or not selector:
        raise InvalidModelRef(
            f"Invalid model_ref '{model_ref}'. Expected '<model_name>@<alias_or_version>'."
        )
    # Defense-in-depth: every downstream package/layout.py call this
    # feeds into (alias_path/version_dir) already validates model_name/
    # selector independently, so this closes no gap by itself -- but it
    # rejects a hostile model_ref at the earliest possible point, with
    # the same InvalidModelRef-shaped 400 this function already raises
    # for a malformed ref, rather than a generic path-safety error from
    # somewhere deeper in the call stack.
    try:
        safe_component(model_name, "model_name")
        safe_component(selector, "alias_or_version")
    except PathTraversalError:
        raise InvalidModelRef(
            f"Invalid model_ref '{model_ref}'. Expected '<model_name>@<alias_or_version>'."
        ) from None
    return ModelRef(model_name=model_name, selector=selector)
