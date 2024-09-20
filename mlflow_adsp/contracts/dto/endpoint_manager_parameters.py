""" Endpoint Manager Parameters Definition"""

from typing import Optional

from ae5_tools import demand_env_var, demand_env_var_as_bool, get_env_var

from .base_model import BaseModel


# pylint: disable=too-many-instance-attributes
class EndpointManagerParameters(BaseModel):
    """
    Endpoint Manager Parameters DTO

    Attributes
    ----------
    env_manager: str = "local"
        Python package manger (local, pip, conda)
    host: str = "0.0.0.0"
        Host to bind server to.
    port: int = 8086
        Port to bind server to.
    model_uri: str
        The model URI to load.
    heart_beat: int = 5
        The internal (in seconds) to check for new model versions.
    enable_mlserver: bool = False
        Flag to control using mlserver rather than native MLflow serve.
    max_tries: int
        When attempting to start `mlflow serve`,
        this refers to the number of times the api endpoint is checked for health before giving up.
    timeout: int
        When attempting to read streams from `mlflow serve` this refers to the timeout of the operation.
    """

    env_manager: str = demand_env_var(name="ENV_MANAGER") if get_env_var(name="ENV_MANAGER") else "local"
    host: str = demand_env_var(name="APP_SERVER_HOST") if get_env_var(name="APP_SERVER_HOST") else "0.0.0.0"
    port: int = int(demand_env_var(name="APP_SERVER_PORT")) if get_env_var(name="APP_SERVER_PORT") else 8086
    model_uri: Optional[str] = demand_env_var(name="MLFLOW_MODEL_URI") if get_env_var(name="MLFLOW_MODEL_URI") else None
    heart_beat: int = (
        int(demand_env_var(name="APP_SERVER_TRACKING_HEART_BEAT"))
        if get_env_var(name="APP_SERVER_TRACKING_HEART_BEAT")
        else 5
    )
    enable_mlserver: bool = (
        demand_env_var_as_bool(name="APP_SERVER_MLSERVER") if get_env_var(name="APP_SERVER_MLSERVER") else False
    )

    # The Default behavior is to wait up to 15 minutes for the process to become healthy.
    # 180 attempts x 5 seconds = 900 seconds / 60 seconds = 15 minutes
    max_tries: int = (
        int(demand_env_var(name="MLFLOW_ADSP_SERVE_MAX_TRIES"))
        if get_env_var(name="MLFLOW_ADSP_SERVE_MAX_TRIES")
        else 180
    )
    timeout: int = (
        int(demand_env_var(name="MLFLOW_ADSP_SERVE_TIMEOUT")) if get_env_var(name="MLFLOW_ADSP_SERVE_TIMEOUT") else 5
    )
