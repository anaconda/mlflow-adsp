"""
This module holds the Endpoint Manager definition used for wrapping `mlflow serve` calls for stages.
"""

from __future__ import annotations

import logging
import os
import random
import shlex
import signal
import subprocess
import time
from subprocess import TimeoutExpired
from typing import Optional

import psutil
import requests
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from requests import Response

from ae5_tools import load_ae5_user_secrets

from ..contracts.dto.endpoint_manager_parameters import EndpointManagerParameters
from ..contracts.dto.target_metadata import TargetMetadata
from ..contracts.errors.plugin import ADSPMLFlowPluginError
from ..contracts.types.reloadable_model_uri_type import ReloadableModelUriType

logger = logging.getLogger(__name__)


class EndpointManager:
    """
    The EndpointManager handles wrapping `mlflow serve` and monitoring changes to the designation model version.

    Attributes
    ----------
    client: MlflowClient
        MLflow client
    params: EndpointManagerParameters
        Endpoint manager parameters
    metadata: TargetMetadata
        The calculated metadata about the model
    version: Optional[str] = None
        The current (known) model version if reloadable.
    """

    SUBPROCESSES: list[subprocess.Popen] = []

    def __init__(self, client: MlflowClient, params: EndpointManagerParameters):
        self.client = client

        self.params = params
        self.metadata = TargetMetadata(model_uri=self.params.model_uri)
        if self.metadata.reloadable:
            self.version = self._get_latest_version()
            message: str = f"Loaded model version ({self.version})"
            logger.info(message)

        self._launch()

    def _process_launch(self, shell_out_cmd: str, cwd: str = ".") -> subprocess.Popen:
        logger.info(shell_out_cmd)
        args = shlex.split(shell_out_cmd)

        try:
            # pylint: disable=consider-using-with
            process = subprocess.Popen(args=args, cwd=cwd)
            EndpointManager._proc_comm(process=process, timeout=self.params.timeout)
            return process
        except ADSPMLFlowPluginError as error:
            # Any failure here is a failed start up of the process.
            message: str = f"Failed to launch process: {str(error)}"
            logger.error(message)
            raise error
        except Exception as error:
            # Any failure here is a failed start up of the process.
            message: str = (
                f"Failed to launch process: {str(error)}, review error type: {type(error)} for launch scenarios"
            )
            logger.error(message)
            raise ADSPMLFlowPluginError(message) from error

    def _poll_network_service(self, process: subprocess.Popen) -> bool:
        """
        Polls the network service endpoint of the process and determines if its online.

        Parameters
        ----------
        process: subprocess.POpen
            The process for the network endpoint.

        Returns
        -------
        ready: bool
            A boolean indicating the service state.
        """

        try:
            version_endpoint_url: str = f"http://{self.params.host}:{self.params.port}/version"
            response: Response = requests.get(url=version_endpoint_url, timeout=self.params.timeout)
            if response.status_code != 200:
                logger.debug("Service not yet healthy, waiting ..")
                EndpointManager._proc_comm(process=process, timeout=self.params.timeout)
            else:
                logger.debug("Service healthy")
                return True

        except requests.exceptions.ConnectionError:
            logger.debug("Unable to connect to service, waiting ..")
            EndpointManager._proc_comm(process=process, timeout=self.params.timeout)

        return False

    def _process_launch_wrapper(self, shell_out_cmd: str, cwd: str = ".") -> None:
        # Launch the process
        process: subprocess.Popen = self._process_launch(shell_out_cmd=shell_out_cmd, cwd=cwd)

        # Monitor the service until its only, or we've failed to start up.
        current_try: int = 0
        while current_try < self.params.max_tries:
            current_try += 1
            message: str = f"Wait cycle {current_try} of {self.params.max_tries}"
            logger.debug(message)
            EndpointManager._proc_comm(process=process, timeout=self.params.timeout)

            # Ensure the process started up (and hasn't terminated)
            if process.poll():
                # Then the process terminated unexpectedly,
                EndpointManager._stop()
                logger.error("Subprocess failed to start up successfully")
                break

            # Determine if the network service is responding
            if self._poll_network_service(process=process):
                break

        # Provide the last output available before moving on.
        EndpointManager._proc_comm(process=process, timeout=self.params.timeout)

        # Check our failure (timout, terminated process) states.

        if current_try >= self.params.max_tries:
            EndpointManager._stop()
            raise ADSPMLFlowPluginError("Unable to start process within timeframe")

        if process.poll():
            EndpointManager._stop()
            raise ADSPMLFlowPluginError("Subprocess failed to start up successfully")

        # Add process to class level list for management.
        EndpointManager.SUBPROCESSES.append(process)

        logger.info("Done monitoring startup, moving on ..")

    @staticmethod
    def _proc_comm(process: subprocess.Popen, timeout: int = 5):
        try:
            process.communicate(timeout=timeout)
            (stdout, stderr) = process.communicate(timeout=timeout)
            if stdout:
                for line in iter(stdout.readline, b""):
                    logger.info(line)
            if stderr:
                for line in iter(stderr.readline, b""):
                    logger.info(line)
        except TimeoutExpired:
            pass

    def _launch(self) -> None:
        """Launches the `mlflow serve` process."""

        load_ae5_user_secrets()
        serve_cmd: str = (
            "mlflow models serve "
            f"--env-manager {self.params.env_manager} "
            f"--host {self.params.host} "
            f"--port {self.params.port} "
            f"--model-uri {self.params.model_uri}"
        )
        if self.params.enable_mlserver:
            serve_cmd += " --enable-mlserver"
        self._process_launch_wrapper(shell_out_cmd=serve_cmd)

    def update(self) -> None:
        """(Re)starts the `mlflow serve` process if the model version changed."""

        if self.metadata.reloadable:
            latest_version: str = self._get_latest_version()
            if latest_version != self.version:
                message: str = f"Current version: ({self.version}), Latest version: ({latest_version}), Reloading .."
                logger.info(message)
                EndpointManager._stop()
                self.version: str = latest_version
                self._launch()

    @staticmethod
    def _stop() -> None:
        """Stops the `mlflow serve` process and children."""

        for process in EndpointManager.SUBPROCESSES:
            try:
                app_subprocess = psutil.Process(pid=process.pid)
                children = app_subprocess.children(recursive=True)

                os.kill(app_subprocess.pid, signal.SIGKILL)
                for child in children:
                    os.kill(child.pid, signal.SIGKILL)
            except psutil.NoSuchProcess:
                # Allow for the cases where the subprocess died, or never started up.
                pass
        EndpointManager.SUBPROCESSES = []

    def _get_latest_version(self) -> str:
        """
        Gets the model version of the designation marked model URI.

        Returns
        -------
        version: str
            The model version.
        """

        if self.metadata.type == ReloadableModelUriType.STAGE:
            # Then attempt to load
            return self._get_version_by_stage()

        if self.metadata.type == ReloadableModelUriType.ALIAS:
            # Then we are an alias designation.
            return self._get_version_by_alias()

        message: str = f"Unknown designation type encountered: ({self.metadata.type})"
        raise ADSPMLFlowPluginError(message)

    def _get_version_by_stage(self) -> str:
        """
        Responsible for returning the version of the model at the tracked stage.

        Returns
        -------
        version: str
            The registered model version.
        """

        try:
            models: list[ModelVersion] = self.client.get_latest_versions(
                name=self.metadata.registry, stages=[self.metadata.name]
            )
            assert len(models) == 1
            return models[0].version
        except (MlflowException, AssertionError) as error:
            message: str = f"Unable to find model: ({self.params.model_uri}), {str(error)}"
            raise ADSPMLFlowPluginError(message) from error

    def _get_version_by_alias(self) -> str:
        """
        Responsible for returning the version of the model at the tracked alias.

        Returns
        -------
        version: str
            The registered model version.
        """

        try:
            return self.client.get_model_version_by_alias(name=self.metadata.registry, alias=self.metadata.name).version
        except MlflowException as error:
            message: str = f"Unable to find model: ({self.params.model_uri}), {str(error)}"
            raise ADSPMLFlowPluginError(message) from error

    @staticmethod
    def _exponential_backoff(attempt: int) -> float:
        """
        Calculate an exponential backoff delay.

        Parameters
        ----------
        attempt: int (> 0)
            The current retry attempt.

        Returns
        -------
        delay: float
            The delay in seconds to wait.
        """

        return round(random.uniform(0, 1) + (2 ^ attempt), 1)

    @staticmethod
    def serve(params: EndpointManagerParameters) -> None:
        """
        Serves the designation marked model for REST API consumption.
        Reloads the model if changed.

        Parameters
        ----------
        params: EndpointManagerParameters
            Endpoint Manager Parameters DTO
        """

        start_attempts: int = 0

        # pylint: disable=broad-exception-caught
        while True:
            manager: Optional[EndpointManager] = None
            try:
                logger.info("Starting ..")
                load_ae5_user_secrets()
                manager = EndpointManager(client=MlflowClient(), params=params)
                logger.info("Startup complete")
                start_attempts = 0

                if manager.metadata.reloadable:
                    logger.info("Watching for version changes ..")
                    while True:
                        EndpointManager._replace(manager=manager)
                        time.sleep(params.heart_beat)
                else:
                    # https://docs.python.org/3/library/signal.html#signal.pause
                    signal.pause()

            except Exception as error:
                start_attempts = EndpointManager._exception_handler(error=error, attempt=start_attempts)
            finally:
                logger.info("Stopping ..")
                if manager:
                    EndpointManager._stop()

    # pylint: disable=broad-exception-caught
    @staticmethod
    def _replace(manager: EndpointManager) -> None:
        """
        Endpoint Replacement Workflow
        This method is responsible for restarting the `mlflow serve` managed model.

        Parameters
        ----------
        manager: EndpointManager
            An instance endpoint manager.
        """

        complete: bool = False
        update_attempts: int = 0
        while not complete:
            try:
                load_ae5_user_secrets()
                manager.update()
                complete = True
            except Exception as error:
                update_attempts = EndpointManager._exception_handler(error=error, attempt=update_attempts)

    @staticmethod
    def _exception_handler(error: Exception, attempt: int) -> int:
        """
        Handles exceptions for looping and waiting.

        Parameters
        ----------
        error: Exception
            The caught exception.
        attempt: int
            The current retry attempt.

        Returns
        -------
            attempt: int
                The attempt + 1.
        """

        if not isinstance(error, ADSPMLFlowPluginError):
            message: str = f"Review exceptions of type ({type(error)}) seen while replacing model endpoint"
            logger.debug(message)

        attempt += 1
        logger.error(str(error))

        delay: float = EndpointManager._exponential_backoff(attempt=attempt)
        message: str = f"Retrying, attempt {attempt}, pausing for {delay} seconds .."
        logger.error(message)
        time.sleep(delay)

        return attempt
