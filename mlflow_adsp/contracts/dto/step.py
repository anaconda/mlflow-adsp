""" Execute Step Definition """

from typing import Dict, Optional

from .base_model import BaseModel


class Step(BaseModel):
    """
    Execute Step DTO

    backend: str = "adsp"
        The backend to leverage for the step execution. Default is `adsp`.
    backend_config: Dict
        The `adsp` backend configuration.
    entry_point: str
        The workflow step to execute. Default is `main`.
    env_manager: str
        Defaults to `local` for the adsp environment.
    experiment_id: Optional[str]
        The experiment ID to use for the execution.
    experiment_name: Optional[str]
        The experiment name to use for the execution.
    parameters: Dict
        The dictionary of parameters to pass to the workflow step.
    run_id: Optional[str] = None
        If provided it is supplied and used for reporting.
    run_name: Optional[str] = None
        If provided it is supplied and used for reporting.
    synchronous: bool = False
        Controls whether to return immediately or after run completion.
        For parallel processing this should be `False`.
    uri: str
        The URI of the MLproject to process.
    """

    backend: str = "adsp"
    backend_config: Optional[Dict] = None
    entry_point: str = "main"
    env_manager: str = "local"
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    parameters: Optional[Dict] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    synchronous: bool = False
    uri: str = "."
