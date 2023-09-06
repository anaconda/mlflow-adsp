""" This module contains logging functions. """

import logging
from typing import Optional

from ae5_tools import demand_env_var, get_env_var

from ..contracts.types.log_level import LogLevel


def set_log_level(level: Optional[LogLevel] = None) -> None:
    """
    Sets log level.

    Parameters
    ----------
    level: str
        A string name for a log level to set.
    """

    if level is None:
        if get_env_var(name="APP_SERVER_LOG_LEVEL"):
            level = LogLevel(demand_env_var(name="APP_SERVER_LOG_LEVEL"))
        else:
            level = LogLevel.INFO

    if level == LogLevel.NOTSET:
        logging.basicConfig(level=logging.NOTSET)
    elif level == LogLevel.INFO:
        logging.basicConfig(level=logging.INFO)
    elif level == LogLevel.WARN:
        logging.basicConfig(level=logging.WARN)
    elif level == LogLevel.WARNING:
        logging.basicConfig(level=logging.WARNING)
    elif level == LogLevel.DEBUG:
        logging.basicConfig(level=logging.DEBUG)
    elif level == LogLevel.ERROR:
        logging.basicConfig(level=logging.ERROR)
    elif level == LogLevel.CRITICAL:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        message: str = f"LogLevel {level} not implemented"
        raise NotImplementedError(message)
