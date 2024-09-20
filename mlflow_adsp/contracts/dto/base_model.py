""" Base Model (Pydantic) Over-Ride """

# pylint: disable=no-name-in-module
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """BaseModel [Pydantic] Over-Ride"""

    # https://pydantic-docs.helpmanual.io/usage/model_config/#options
    class Config:
        """Pydantic Config Over-Ride"""

        arbitrary_types_allowed = True
        use_enum_values = True
