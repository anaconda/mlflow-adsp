""" Process Handling Code  """

import logging
import shlex
import subprocess

from ..contracts.errors.plugin import ADSPMLFlowPluginError
from ..contracts.errors.subprocess_failure_error import SubprocessFailureError

logger = logging.getLogger(__name__)


def process_launch_wait(cwd: str, shell_out_cmd: str) -> None:
    """
    Internal function for wrapping process launches [and waiting].

    Parameters
    ----------
    shell_out_cmd: str
        The command to be executed.
    """

    args = shlex.split(shell_out_cmd)

    try:
        with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            for line in iter(process.stdout.readline, b""):
                logger.info(line)

        if process.returncode != 0:
            raise SubprocessFailureError(
                command=shell_out_cmd, message="subprocess failed", returncode=process.returncode
            )
    except Exception as error:
        if isinstance(error, SubprocessFailureError):
            raise error

        raise ADSPMLFlowPluginError("Exception was caught while executing task.") from error
