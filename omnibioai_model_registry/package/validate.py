# File: omnibioai_model_registry/package/validate.py
"""
OmniBioAI omnibioai_model_registry.package.validate.

Purpose:
    Defines validate_package_files for omnibioai_model_registry.package.validate.

Author:
    Manish Kumar <manish@omnibioai.org>
"""

from __future__ import annotations

from pathlib import Path

from ..errors import ValidationError
from .layout import REQUIRED_FILES


def validate_package_files(version_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (version_dir / f).exists()]
    if missing:
        raise ValidationError(
            f"Model package missing required files: {missing}. In: {version_dir}"
        )
