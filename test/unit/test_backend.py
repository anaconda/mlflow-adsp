import os
import uuid
from typing import Dict
from unittest.mock import MagicMock

import pytest

from ae5_tools.api import AEUserSession
from mlflow_adsp import ADSPProjectBackend, ADSPSubmittedRun


@pytest.fixture(scope="function")
def get_token_fixture():
    return {
        "access_token": str(uuid.uuid4()),
        "refresh_token": str(uuid.uuid4()),
    }


@pytest.fixture(scope="function")
def get_ae_user_session(get_token_fixture) -> AEUserSession:
    user_session = AEUserSession(
        hostname="MOCK-HOSTNAME", username="MOCK-AE-USERNAME", password="MOCK-AE-USER-PASSWORD"
    )
    user_session._load = MagicMock()
    user_session._sdata = get_token_fixture
    return user_session


@pytest.fixture(scope="function")
def submitted_run(get_ae_user_session) -> ADSPSubmittedRun:
    mock_mlflow_run_id: str = str(uuid.uuid4())
    mock_adsp_job_id: str = str(uuid.uuid4())
    mock_response: Dict = {}

    submitted_run: ADSPSubmittedRun = ADSPSubmittedRun(
        ae_session=get_ae_user_session,
        mlflow_run_id=mock_mlflow_run_id,
        adsp_job_id=mock_adsp_job_id,
        response=mock_response,
    )

    return submitted_run


def test_run(get_ae_user_session):
    # Set up test
    mock_session = get_ae_user_session
    mock_session.job_create = MagicMock(return_value={"id": "MOCK-JOB-ID"})
    backend = ADSPProjectBackend(ae_session=mock_session)

    params: Dict = {
        "project_uri": "./test/fixtures/consumer",
        "entry_point": "main",
        "params": {"param_one": "MOCK-PARAM-VALUE"},
        "version": "2d0335398980b62e556a79b9d2198bbec7964a7b",  # `git rev-parse HEAD` of the project repo
        "backend_config": {
            "resource_profile": "MOCK-PROFILE",
            "PROJECT_STORAGE_DIR": "./test/fixtures/consumer",
            "STORAGE_DIR": "./test/fixtures/consumer",
        },
        "tracking_uri": "MOCK-TRACKING-URI",
        "experiment_id": "0",
    }

    # Execute test
    submitted_run = backend.run(**params)

    # Review the results
    assert submitted_run.adsp_job_id == "MOCK-JOB-ID"

    mock: MagicMock = mock_session.job_create
    call_arguments = mock.call_args[1]
    del call_arguments["format"]
    assert call_arguments == {
        "ident": "a0-mock-tool-project-url-id",
        "name": submitted_run.mlflow_run_id,
        "command": "Worker",
        "resource_profile": "MOCK-PROFILE",
        "variables": {
            "MLFLOW_RUN_ID": submitted_run.mlflow_run_id,
            "MLFLOW_EXPERIMENT_ID": "0",
            "TRAINING_ENTRY_POINT": "python -m steps.main --param-one MOCK-PARAM-VALUE",
        },
        "run": True,
    }
