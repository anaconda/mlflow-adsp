"""
Worker Service Definition
Invoked through an anaconda-project run command when the Anaconda Data Science Platform launches the job.
"""

import logging
from typing import Optional

import click

from ae5_tools import demand_env_var

from ..common.log import set_log_level
from ..common.process import process_launch_wait
from ..contracts.types.log_level import LogLevel

logger = logging.getLogger(__name__)


@click.command(name="worker")
@click.option(
    "--log-level",
    type=click.Choice(["notset", "info", "warn", "warning", "debug", "error", "critical"]),
    help="Log level.",
)
def worker(log_level: Optional[str] = None) -> None:
    """
    Pulls the run-time command out of the environment variable declared during job creation, then launches it.
    """

    set_log_level(level=LogLevel(log_level) if log_level else None)
    logger.debug("Processing MLflow Step")
    training_entry_point: str = demand_env_var(name="TRAINING_ENTRY_POINT")
    logger.debug(training_entry_point)
    process_launch_wait(cwd=".", shell_out_cmd=training_entry_point)
    logger.debug("Complete")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    worker()
