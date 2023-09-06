""" Defines the structure of the internal job presentation used by the scheduler. """

from typing import List, Optional, Union

from mlflow.entities import RunStatus
from mlflow.projects.submitted_run import LocalSubmittedRun

from ...submitted_run import ADSPSubmittedRun
from .base_model import BaseModel
from .step import Step


class Job(BaseModel):
    """
    Scheduler Job DTO

    Attributes
    ----------
    id: str
        A unique (uuid) for the job.
    step: Step
        The origination request
    runs: List[Union[ADSPSubmittedRun, LocalSubmittedRun]] = []
        The runs associated with the job request
    last_status: Optional[RunStatus] = None
        The last seen mlflow status of the job.
    """

    id: str
    step: Step
    runs: List[Union[ADSPSubmittedRun, LocalSubmittedRun]] = []
    last_status: Optional[RunStatus] = None
