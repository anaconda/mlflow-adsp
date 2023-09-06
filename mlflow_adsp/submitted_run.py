""" Anaconda Data Science Platform Submitted Run Definition """

import logging
import time
from typing import Dict, List

from mlflow.entities import RunStatus
from mlflow.projects.submitted_run import SubmittedRun

from ae5_tools.api import AEUserSession

from .contracts.dto.base_model import BaseModel
from .contracts.errors.plugin import ADSPMLFlowPluginError
from .contracts.types.job_run_state import AEProjectJobRunStateType

logger = logging.getLogger(__name__)


class ADSPSubmittedRun(SubmittedRun, BaseModel):
    """
    Anaconda Data Science Platform Submitted Run
    Sub-classes the MLFlow `SubmittedRun` used for backend management.

    Attributes
    ----------
    ae_session: AEUserSession
        An Anaconda Data Science Platform session used for communication with the platform.
    mlflow_run_id: str
        The MLFlow Run ID for the current context.
    adsp_job_id: str
        The Anaconda Data Science Platform Job ID
    response: Dict
        A dictionary response of the job creation request.
    """

    ae_session: AEUserSession
    mlflow_run_id: str
    adsp_job_id: str
    response: Dict
    wait_interval: int = 15  # 15 seconds

    def get_log(self) -> str:
        """
        Gets the [current] logs for the job run.

        Returns
        -------
        log: str
            A string representation of the run output.
        """

        runs_status: List[Dict] = self.ae_session.job_runs(ident=self.adsp_job_id)
        ADSPSubmittedRun._validate_response(runs_status=runs_status)

        run_id: str = runs_status[0]["id"]
        return self.ae_session.run_log(ident=run_id)

    @staticmethod
    def _validate_response(runs_status: List[Dict]) -> None:
        """
        Ensures that only a single item is returned in the list, throws an exception otherwise.

        Parameters
        ---------
        runs_status: List[Dict]
            The list of runs to review
        """

        if len(runs_status) != 1:
            message: str = f"Unable to determine which run to analyze, saw: ({runs_status})"
            raise ADSPMLFlowPluginError(message)

    def wait(self) -> bool:
        """
        Waits for the run to complete then returns the success status.

        Returns
        -------
        success: bool
            Returns True/False based on successful run execution.
        """

        self._wait_on_job_run()

        runs_status: List[Dict] = self.ae_session.job_runs(ident=self.adsp_job_id)
        ADSPSubmittedRun._validate_response(runs_status=runs_status)

        run_state: str = runs_status[0]["state"]
        if run_state == AEProjectJobRunStateType.COMPLETED:
            return True
        return False

    def _wait_on_job_run(self) -> None:
        """Blocks while the run is still in active execution."""

        completed: bool = False

        while not completed:
            runs_status: List[Dict] = self.ae_session.job_runs(ident=self.adsp_job_id)
            ADSPSubmittedRun._validate_response(runs_status=runs_status)

            run_state: str = runs_status[0]["state"]
            if run_state in [
                AEProjectJobRunStateType.FAILED,
                AEProjectJobRunStateType.STOPPED,
                AEProjectJobRunStateType.COMPLETED,
            ]:
                completed = True
            else:
                time.sleep(self.wait_interval)

    def get_status(self) -> RunStatus:
        """
        Gets the current status of the run.

        Returns
        -------
        status: RunStatus
            Returns an MLFlow run status for the Anaconda Data Science Platform run status.
        """

        runs_status: List[Dict] = self.ae_session.job_runs(ident=self.adsp_job_id)
        ADSPSubmittedRun._validate_response(runs_status=runs_status)

        run_state: str = runs_status[0]["state"]

        if run_state == AEProjectJobRunStateType.INITIAL:
            return RunStatus.RUNNING

        if run_state == AEProjectJobRunStateType.RUNNING:
            return RunStatus.RUNNING

        if run_state == AEProjectJobRunStateType.FAILED:
            return RunStatus.FAILED

        if run_state == AEProjectJobRunStateType.STOPPED:
            return RunStatus.KILLED

        if run_state == AEProjectJobRunStateType.COMPLETED:
            return RunStatus.FINISHED

        message: str = f"Unknown job state seen: ({run_state})"
        raise ADSPMLFlowPluginError(message)

    def cancel(self) -> None:
        """Cancels a run's execution"""

        self.ae_session.run_stop(ident=self.adsp_job_id)

    @property
    def run_id(self):
        """
        `run_id` Property

        Returns
        -------
        run_id: str
            The MLFlow Run ID for the context.
        """

        return self.mlflow_run_id
