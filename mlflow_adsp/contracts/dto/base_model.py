""" Base Model (Pydantic) Over-Ride """

from typing import List

# pylint: disable=no-name-in-module
from pydantic import BaseModel as PydanticBaseModel


def lower_camel_case(string: str) -> str:
    """
        Alias Generator Definition
        Used for externally based consumption
    .
        Parameters
        ----------
        string: str
            The input string to change casing of.
        Returns
        -------
        new_string: str
            A new string which has been camel cased.
    """

    string_list: List[str] = string.split("_")
    prefix: str = string_list[0]
    suffix: str = "".join(word.capitalize() for word in string_list[1:])
    return prefix + suffix


class BaseModel(PydanticBaseModel):
    """BaseModel [Pydantic] Over-Ride"""

    # Pydantic Config Over-Ride
    class Config:
        alias_generator = lower_camel_case
        populate_by_name = True
        arbitrary_types_allowed = True
        use_enum_values = True
