""" Scheduler for MLFlow Workflow Steps On ADSP """

import logging
import time
import uuid
from typing import Dict, List, Optional, Union

import mlflow
from mlflow.entities import RunStatus
from mlflow.projects.submitted_run import LocalSubmittedRun
from tqdm import tqdm

from ae5_tools import demand_env_var

from ..contracts.dto.job import Job
from ..contracts.dto.step import Step
from ..contracts.errors.plugin import ADSPMLFlowPluginError
from ..submitted_run import ADSPSubmittedRun

logger = logging.getLogger(__name__)


class Scheduler:
    """
    The Scheduler handles launching MLFlow workflow steps within ADSP.

    With built-in job limiting and retries the class provides an easy method
    for submitting serial or parallel workflow steps for processing within ADSP.

    Attributes
    ----------
    jobs: List[Job]
        The List of jobs the user requested for processing.
    todo: List[str]
        The jobs marked as TODO but that are not yet under execution.
    inprogress: List[str]
        Jobs which have started put are not yet finished.
    complete: List[str]
        Jobs which have completed (successfully or not).
    max_workers: int
        The maximum number of parallel jobs to execute in parallel.
    failed_execution_retry_max: int
        The maximum number of retries for a job that failed.
    """

    jobs: List[Job]

    # Used for tracking jobs in different states of execution.
    todo: List[str]
    inprogress: List[str]
    complete: List[str]

    max_workers: int
    failed_execution_retry_max: int

    def __init__(self, failed_execution_retry_max: int = 3, max_workers: Optional[int] = None):
        self._reset(failed_execution_retry_max=failed_execution_retry_max, max_workers=max_workers)

    def _reset(self, failed_execution_retry_max: int, max_workers: Optional[int] = None) -> None:
        """
        Resets the internal class state.  This is used during initialization and on each work queue processing request.

        Parameters
        ----------
        failed_execution_retry_max: int
            The maximum number of retries for a job that failed.
        max_workers: Optional[int] = None
            The maximum number of parallel jobs to execute in parallel.  If unset the default is used.
        """

        # Reset internal state

        self.failed_execution_retry_max = failed_execution_retry_max
        self.max_workers = max_workers if max_workers else int(demand_env_var(name="ADSP_WORKER_MAX"))

        self.jobs = []
        self.todo = []
        self.inprogress = []
        self.complete = []

    # We define a lot of parameters on this call.  It allows the caller to better control
    # how processing is handled.  This could be moved into a DTO but given that most
    # parameters will not be defined explicitly by the caller, I thought it best to
    # leave as-is.
    # pylint: disable=too-many-arguments
    def process_work_queue(
        self,
        steps: List[Step],
        interval: float = 2.0,
        exponent: float = 1.4,
        retry_max: int = 3,
        disable_progress_bar: bool = False,
    ) -> List[Job]:
        """
        Processing a list of execution requests.
        If the jobs are async the processing will occur in parallel, serial execution is used otherwise.

        Parameters
        ----------
        steps: List[Step]
            The list of execution requests to process.
        interval: float
            The wait internal to use during exponential backoff
        exponent: float
            The exponent used for calculated exponential backoff
        retry_max: int = 3
            The maximum number of retries for a job that failed.
        disable_progress_bar: bool = False
            Controls the display of the progress bar.

        Returns
        -------
        runs: List[Job]
            A list of results from the work queue.
        """

        self._reset(failed_execution_retry_max=retry_max)

        # Build of internal job representation
        for step in steps:
            self.jobs.append(Job(id=str(uuid.uuid4()), step=step, runs=[]))
        self.todo = [job.id for job in self.jobs]

        with tqdm(total=len(self.todo), disable=disable_progress_bar) as progress_bar:
            progress_bar.set_description(desc=self._stats_str())

            # Process jobs
            retries: int = 0
            while self._work_queue_in_progress():
                # Fill our processing queue
                self._fill_processing_queue()
                progress_bar.set_description(desc=self._stats_str())

                # review in progress jobs
                old_complete: int = len(self.complete)
                self._review_in_progress_jobs()

                new_complete: int = len(self.complete)
                delta: int = new_complete - old_complete
                if delta > 0:
                    progress_bar.update(n=delta)
                    progress_bar.set_description(desc=self._stats_str())

                # allow for processing time when the queue is full and there's work to do.
                if self._wait_on_work_queue(interval=interval, exponent=exponent, retries=retries):
                    # Determine how we update retries.
                    retries += 1
                else:
                    retries = 0

        # Returns job structure
        return self.jobs

    @staticmethod
    def execute_step(step: Step) -> ADSPSubmittedRun:
        """
        Execute a MLFlow Workflow Step

        Parameters
        ----------
        step: Step

        Returns
        -------
        submitted_job: ADSPSubmittedRun
            An instance of `SubmittedRun` for the requested workflow step run.
        """

        step_dict: Dict = step.dict(by_alias=False)

        message: str = f"Launching new background job for: {step_dict}"
        logger.debug(message)

        return mlflow.projects.run(**step_dict)

    def _stats_str(self) -> str:
        """
        Builds a report of the current internal processing state.  The output is suitable for logging
        or use with the progress bar description.

        Returns
        -------
        report: str
            Scheduler work queue report.
        """

        return (
            f"todo: {len(self.todo)}, "
            f"inprogress: {len(self.inprogress)}, "
            f"complete: {len(self.complete)}, "
            f"max_workers: {self.max_workers}"
        )

    def _fill_processing_queue(self) -> None:
        """Fills the job processing queue."""

        logger.debug("Filling queue")
        logger.debug(self._stats_str())
        while len(self.inprogress) < self.max_workers and len(self.todo) > 0:
            job_id: str = self.todo.pop()
            job: Job = [job for job in self.jobs if job.id == job_id][0]

            new_run: ADSPSubmittedRun = Scheduler.execute_step(step=job.step)
            job.runs.append(new_run)
            self.inprogress.append(job_id)

    def _review_in_progress_jobs(self) -> None:
        """
        Review inprogress jobs for completion.  Move them to the completed list if processing is complete.
        """

        logger.debug("Reviewing in progress jobs")
        logger.debug(self._stats_str())

        new_inprogress: List[str] = []
        while len(self.inprogress) > 0:
            job_id: str = self.inprogress.pop()
            popped_job: Job = [job for job in self.jobs if job.id == job_id][0]

            if len(popped_job.runs) < 1:
                raise ADSPMLFlowPluginError("Unable to find job run to review")

            latest_run: Union[ADSPSubmittedRun, LocalSubmittedRun] = popped_job.runs[-1]
            job_status: Union[str, RunStatus] = latest_run.get_status()
            popped_job.last_status = Scheduler._coerce_run_status(status=job_status)

            job_status_msg: str = f"Job ID: {job_id}, Last seen status: {popped_job.last_status}"
            logger.debug(job_status_msg)
            if popped_job.last_status in (RunStatus.RUNNING, RunStatus.SCHEDULED):
                logger.debug("Job still in progress")
                new_inprogress.append(job_id)
            elif popped_job.last_status == RunStatus.FINISHED:
                logger.debug("Job completed")
                Scheduler._add_log_to_run(run=latest_run)
                self.complete.append(job_id)
            else:
                logger.debug("Determining retry behavior ...")
                # job_status is either RunStatus.KILLED, or RunStatus.FAILED and retry logic kicks in.

                # Explicitly mark the job failed.
                Scheduler._add_log_to_run(run=latest_run)
                Scheduler._mark_mlflow_run_as_failed(run_id=latest_run.mlflow_run_id)

                if len(popped_job.runs) < self.failed_execution_retry_max:
                    # We are still under the try limit
                    logger.debug("Job will be retried")
                    self.todo.append(job_id)
                else:
                    # Max retry count has been reached, mark as completed (though unsuccessful)
                    logger.debug("Job will not be retried")
                    self.complete.append(job_id)
        self.inprogress = new_inprogress

    @staticmethod
    def _mark_mlflow_run_as_failed(run_id: str) -> None:
        """
        Tags the background run as failed.

        Originally the intent to was end that run with a failed status. However,
        this appears to have adverse effects on the parent run and subsequent child
        runs.  (See notes below)

        Parameters
        ----------
        run_id: MLFlow nested child run id to update.
        """

        with mlflow.start_run(run_id=run_id, nested=True, tags={"ADSP_FAILURE": True}):
            # pylint: disable=fixme
            # TODO: Review if this is fixed in later versions of mlflow.
            # There appears to be an MLFlow bug.
            # If we end a run in this way we also end the parent run.
            # mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))
            pass

    @staticmethod
    def _add_log_to_run(run: Union[ADSPSubmittedRun, LocalSubmittedRun]) -> None:
        """
        Adds background job log to the mlflow run as an artifact.

        Parameters
        ----------
        run: ADSPSubmittedRun
            Instance of the run to update.
        """

        if isinstance(run, ADSPSubmittedRun):
            with mlflow.start_run(run_id=run.mlflow_run_id, nested=True):
                mlflow.log_text(run.get_log(), artifact_file="job_log.txt")

    @staticmethod
    def _coerce_run_status(status: Union[str, RunStatus]) -> Union[int, RunStatus]:
        """
        The method `run.get_status()` signature is incorrect.  It should return an instance
        of `RunStatus` but returns a `str` value.

        Based on the generating sub-system you will get back the `RunStatus` enumeration as
        either an integer (key) or a string (value). The below code performs the coercion
        from either representation back into a `RunStatus` comparable type (int).

        Parameters
        ----------
        status: Union[str, RunStatus]

        Returns
        -------
        status: int
            `RunStatus` compatibility enumeration.
        """

        if isinstance(status, str):
            try:
                return int(status)
            except ValueError:
                return RunStatus.from_string(status)

        return status

    def _wait_on_work_queue(self, interval: float, exponent: float, retries: int) -> bool:
        """
        Performs exponential backoff waiting for inprogress jobs.
        Allows for processing time when the queue is full and there's work to do.

        Parameters
        ----------
        interval: float
            The base wait interval.
        exponent: float
            The exponent of the exponential backoff algorithm.
        retries: int
            The current retry count.

        Returns
        -------
        waited: bool
            Flag for whether a wait occurred.  `True` if it did, `False` otherwise.
        """

        logger.debug("Allowing for processing time when the queue is full")
        logger.debug(self._stats_str())
        if (len(self.inprogress) >= self.max_workers) or (len(self.todo) <= 0 < len(self.inprogress)):
            logger.debug("Queue is full (or only in progress work left), pausing before refilling the queue ...")
            time.sleep(pow(exponent, retries) * interval)  # exponential backoff
            logger.debug("done")
            return True
        return False

    def _work_queue_in_progress(self) -> bool:
        """
        Determines if the jobs in the work queue have completed.

        Returns
        -------
        complete: bool
            `True` if the job queue is complete, `False` otherwise.
        """

        logger.debug("Determining if we are complete ...")
        logger.debug(self._stats_str())

        if len(self.todo) <= 0 and len(self.inprogress) <= 0:
            return False
        return True
