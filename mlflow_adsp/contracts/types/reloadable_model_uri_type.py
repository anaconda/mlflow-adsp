"""
Reloadable Model URI Type Definition
"""

from __future__ import annotations

from enum import Enum


class ReloadableModelUriType(str, Enum):
    """
    Reloadable Model URI Type
    This conforms to the MLflow URI usage for stage and alias specifiers.
    """

    ALIAS = "@"
    STAGE = "/"
