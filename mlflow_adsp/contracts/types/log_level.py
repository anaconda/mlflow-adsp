"""Log Level Definition"""

from enum import Enum


class LogLevel(str, Enum):
    """
    Python Log Levels
    https://docs.python.org/3.11/library/logging.html#levels
    """

    NOTSET = "notset"
    INFO = "info"
    WARN = "warn"
    WARNING = "warning"
    DEBUG = "debug"
    ERROR = "error"
    CRITICAL = "critical"
