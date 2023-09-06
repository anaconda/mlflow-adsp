"""Subprocess Failure Error Definition"""
import json


class SubprocessFailureError(Exception):
    """Subprocess Failure Error"""

    command: str
    message: str
    returncode: int

    def __init__(self, command: str, message: str, returncode: int):
        self.command = command
        self.message = message
        self.returncode = returncode

    def __str__(self):
        message: str = json.dumps({"command": self.command, "message": self.message, "returncode": self.returncode})
        return message
