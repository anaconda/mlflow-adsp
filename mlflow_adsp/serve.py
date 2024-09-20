"""
Wraps `mlflow serve`. This is not meant to be executed directly, but called through the cli command `mlflow-adsp`.
"""

import sys
from typing import Optional

import click

from .common.log import set_log_level
from .contracts.dto.endpoint_manager_parameters import EndpointManagerParameters
from .contracts.errors.plugin import ADSPMLFlowPluginError
from .contracts.types.log_level import LogLevel
from .services.endpoint_manager import EndpointManager


# pylint: disable=too-many-arguments
@click.command(name="serve")
@click.option(
    "--env-manager",
    type=str,
    help=(
        "The environment manager to use, (conda, pip, local) what is supported by MLflow. "
        "Local is the default and is recommended when running within the Anaconda Data Science Platform."
    ),
)
@click.option("--host", type=str, help="Host to bind to when running.")
@click.option("--port", type=str, help="Port to bind to when running.")
@click.option(
    "--model-uri",
    type=str,
    help=(
        "The MLflow compliant model URI to load. "
        "Only models defined in a stage or an alias can be dynamic reloaded. "
        "All other types will run without updating."
    ),
)
@click.option(
    "--heart-beat",
    type=int,
    help="The internal to poll the MLflow Tracking Server for model updates when running a reloadable model type.",
)
@click.option("--enable-mlserver", type=bool, help="Flag for mlserver functionality.")
@click.option(
    "--log-level",
    type=click.Choice(["notset", "info", "warn", "warning", "debug", "error", "critical"]),
    help="Log level.",
)
def serve(
    env_manager: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    model_uri: Optional[str] = None,
    heart_beat: Optional[int] = None,
    enable_mlserver: Optional[bool] = None,
    max_tries: Optional[int] = None,
    timeout: Optional[int] = None,
    log_level: Optional[str] = None,
) -> None:
    """
    Wraps `MLflow serve` allowing for reloading of models defined with stages or aliases.

    Attributes
    ----------
    env_manager: Optional[str]
        The environment manager to use, (conda, pip, local) that is supported by MLflow.
        Local is the default and is recommended when running within the Anaconda Data Science Platform.
    host: Optional[str]
        Host to bind to when running.
    port: Optional[int]
        Port to bind to when running.
    model_uri: Optional[str]
        The MLflow compliant model URI to load. Only models defined in a stage or an alias can be dynamically reloaded.
        All other types will run without updating.
    heart_beat: Optional[int]
        The internal to poll the MLflow Tracking Server for model updates when running a reloadable model type.
    enable_mlserver: Optional[bool]
        Enables mlserver functionality.
    max_tries: Optional[int]
        When attempting to start `mlflow serve`, this refers to the number of times the api endpoint is checked for
        health before giving up.
    timeout: Optional[int]
        When attempting to read streams from `mlflow serve` this refers to the timeout of the operation.
    log_level: Optional[str]
        Log level.
    """

    set_log_level(level=LogLevel(log_level) if log_level else None)

    params = EndpointManagerParameters()
    if env_manager:
        params.env_manager = env_manager
    if host:
        params.host = host
    if port:
        params.port = port
    if model_uri:
        params.model_uri = model_uri
    if heart_beat:
        params.heart_beat = heart_beat
    if enable_mlserver:
        params.enable_mlserver = enable_mlserver
    if max_tries:
        params.max_tries = max_tries
    if timeout:
        params.timeout = timeout

    if params.model_uri is None:
        raise ADSPMLFlowPluginError("Unable to determine model URI")

    EndpointManager.serve(params)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    serve(sys.argv)
