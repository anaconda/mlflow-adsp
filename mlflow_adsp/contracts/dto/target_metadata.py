"""
Target Model Metadata DTO Definition
"""

import re
from typing import Optional

from ..types.reloadable_model_uri_type import ReloadableModelUriType


class TargetMetadata:
    """
    Target Model Metadata DTO
    Contains metadata about the hosted target model.

    Attributes
    ----------
    registry: Optional[str] = None
        The name of the model registry.
    type: Optional[ReloadableModelUriType] = None
        The designation type (stage, or alias)
    name: Optional[str] = None
        The name of the designation (stage or alias)
    reloadable: bool = False
        The reload-ability of the target.
    """

    registry: Optional[str] = None
    type: Optional[ReloadableModelUriType] = None
    name: Optional[str] = None
    reloadable: bool = False

    def __init__(self, model_uri: str):
        model_metadata: Optional[tuple[str, str, str]] = TargetMetadata._parse_model_uri(model_uri=model_uri)
        if model_metadata:
            self.reloadable = True
            (
                self.registry,
                self.type,
                self.name,
            ) = model_metadata

    @staticmethod
    def _parse_model_uri(model_uri: str) -> Optional[tuple[str, str, str]]:
        """
        Model restarts are only supported with URIs of the forms:
        1. models:/model_registry@alias
        2. models:/model_registry/stage

        Parameters
        ----------
        model_uri: str
            The mode URI to check reloadability on and parse.

        Returns
        -------
        A tuple of registry name, delimiter, name if reloadable, and None otherwise.
        """

        regex_expression = re.compile(r"^models:/(.[\w-]+)([/@])(.[\w-]+)$")
        groups: Optional[re.Match] = regex_expression.match(model_uri)

        if groups:
            model_registry_name: str = groups.group(1)
            designation_type: ReloadableModelUriType = ReloadableModelUriType(groups.group(2))
            designation_name = groups.group(3)

            version_regex = re.compile(r"^(\d+)$")
            capture_groups: Optional[re.Match] = version_regex.match(designation_name)
            if designation_type == ReloadableModelUriType.STAGE and capture_groups:
                # Then this is a false positive.  This is a version and not a stage.
                return None

            return model_registry_name, designation_type, designation_name
        return None
