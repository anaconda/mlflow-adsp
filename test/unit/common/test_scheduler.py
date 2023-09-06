import time
import uuid
from typing import Optional
from unittest.mock import MagicMock

import mlflow
import pytest
from mlflow.entities import RunStatus

import mlflow_adsp
from ae5_tools.api import AEUserSession
from mlflow_adsp import ADSPMLFlowPluginError, ADSPSubmittedRun, Job, Scheduler, Step


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


###############################################################################
# execute_step Tests
###############################################################################


def test_execute_step(monkeypatch):
    # Set up the test
    mock_request: Step = Step()
    monkeypatch.setattr(mlflow.projects, "run", MagicMock())

    # Execute the test
    Scheduler.execute_step(step=mock_request)

    # Review the results
    mock: MagicMock = mlflow.projects.run
    mock.assert_called_once_with(**Step.parse_obj({}).dict(by_alias=False))


###############################################################################
# _stats_str Tests
###############################################################################


def test_stats_str():
    # Execute the test
    result: str = Scheduler()._stats_str()

    # Review the results
    assert result == "todo: 0, inprogress: 0, complete: 0, max_workers: 3"


###############################################################################
# _fill_processing_queue Tests
###############################################################################


def test_fill_processing_queue(monkeypatch):
    # Scenario:
    # 1 job to process

    # Set up the test
    job_id: str = str(uuid.uuid4())

    mock_job: MagicMock = MagicMock()
    mock_job.id = job_id
    mock_job.step = Step()
    mock_job.runs = []

    mock_run: MagicMock = MagicMock(return_value="mock_new_run")

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "execute_step", mock_run)

    scheduler: Scheduler = Scheduler()
    scheduler.jobs.append(mock_job)
    scheduler.todo.append(job_id)

    # Execute the test
    scheduler._fill_processing_queue()

    # Review the results
    assert len(scheduler.todo) == 0
    assert len(scheduler.inprogress) == 1
    assert scheduler.inprogress[0] == job_id

    mock_run.assert_called_once_with(step=mock_job.step)

    assert mock_job.runs[0] == "mock_new_run"


def test_fill_processing_queue_more_jobs_than_workers(monkeypatch):
    # Scenario:
    # 2 jobs to process, 1 worker

    # Set up the test
    job_id_one: str = str(uuid.uuid4())
    job_id_two: str = str(uuid.uuid4())

    mock_job_one: MagicMock = MagicMock()
    mock_job_two: MagicMock = MagicMock()
    mock_job_one.id = job_id_one
    mock_job_two.id = job_id_two
    mock_job_one.request = Step()
    mock_job_two.request = Step()
    mock_job_one.runs = []
    mock_job_two.runs = []

    mock_run: MagicMock = MagicMock(side_effect=["mock_new_run_one", "mock_new_run_two"])
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "execute_step", mock_run)

    scheduler: Scheduler = Scheduler()
    scheduler.max_workers = 1
    scheduler.jobs.append(mock_job_one)
    scheduler.jobs.append(mock_job_two)
    scheduler.todo.append(job_id_one)
    scheduler.todo.append(job_id_two)

    # Execute the test
    scheduler._fill_processing_queue()

    # Review the results
    assert len(scheduler.todo) == 1
    assert len(scheduler.inprogress) == 1
    assert scheduler.inprogress[0] == job_id_two

    # Queues should not change
    scheduler._fill_processing_queue()
    assert len(scheduler.todo) == 1
    assert len(scheduler.inprogress) == 1
    assert scheduler.inprogress[0] == job_id_two

    # First item completes
    scheduler.complete.append(scheduler.inprogress.pop())

    scheduler._fill_processing_queue()
    assert len(scheduler.todo) == 0
    assert len(scheduler.inprogress) == 1
    assert scheduler.inprogress[0] == job_id_one


def test_fill_processing_queue_no_work(monkeypatch):
    # Scenario:
    # No jobs to process

    # Set up the test
    scheduler: Scheduler = Scheduler()

    # Execute the test
    scheduler._fill_processing_queue()

    # Review the results
    assert len(scheduler.todo) == 0
    assert len(scheduler.inprogress) == 0


###############################################################################
# _coerce_run_status Tests
###############################################################################


def test_coerce_run_status():
    # Set up the tests
    test_cases = [
        {"input": RunStatus.RUNNING, "output": RunStatus.RUNNING},
        {"input": "1", "output": 1},
        {"input": "RUNNING", "output": 1},
    ]

    # Execute the tests
    for test_case in test_cases:
        output = Scheduler._coerce_run_status(status=test_case["input"])

        # Review the results
        assert output == test_case["output"]


###############################################################################
# _wait_on_work_queue Tests
###############################################################################


def test_wait_on_work_queue_wait_on_completion(monkeypatch):
    # Set up the test
    test_cases = [
        {"todo": [], "inprogress": [], "max_workers": 2, "result": False},  # no work
        {"todo": [], "inprogress": [str(uuid.uuid4())], "max_workers": 1, "result": True},  # work queue is full
        {"todo": [str(uuid.uuid4())], "inprogress": [], "max_workers": 2, "result": False},  # item waiting
        {
            "todo": [str(uuid.uuid4())],
            "inprogress": [str(uuid.uuid4())],
            "max_workers": 2,
            "result": False,
        },  # room in queue
    ]

    scheduler = Scheduler()
    monkeypatch.setattr(time, "sleep", MagicMock())

    for test_case in test_cases:
        scheduler.todo = test_case["todo"]
        scheduler.inprogress = test_case["inprogress"]
        scheduler.max_workers = test_case["max_workers"]

        # Execute the test
        result = scheduler._wait_on_work_queue(interval=1, exponent=1, retries=1)

        # Review the results
        assert result == test_case["result"]


###############################################################################
# _work_queue_in_progress Tests
###############################################################################


def test_work_queue_in_progress():
    # Set up the tests
    test_cases = [
        {"todo": [], "inprogress": [], "result": False},
        {"todo": [str(uuid.uuid4())], "inprogress": [], "result": True},
        {"todo": [], "inprogress": [str(uuid.uuid4())], "result": True},
        {"todo": [str(uuid.uuid4())], "inprogress": [str(uuid.uuid4())], "result": True},
    ]

    scheduler = Scheduler()

    # Execute the tests
    for test_case in test_cases:
        scheduler.todo = test_case["todo"]
        scheduler.inprogress = test_case["inprogress"]

        result = scheduler._work_queue_in_progress()

        # Review the result
        assert result == test_case["result"]


###############################################################################
# _review_in_progress_jobs Tests
###############################################################################


def generate_adsp_meta_job(id: Optional[str] = None, request: Optional[Step] = None) -> Job:
    params = {}

    if id:
        params["id"] = id
    else:
        params["id"] = str(uuid.uuid4())

    if request:
        params["step"] = request
    else:
        params["step"] = Step()

    return Job(**params)


def test_review_in_progress_jobs_no_work(monkeypatch):
    # test cases

    test_case = {
        "initial": {"jobs": [], "todo": [], "inprogress": [], "complete": [], "max_workers": 3},
        "result": {"jobs": [], "todo": [], "inprogress": [], "complete": []},
    }  # no work to perform

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    # Set up the test
    scheduler = Scheduler()
    scheduler.jobs = test_case["initial"]["jobs"]
    scheduler.todo = test_case["initial"]["todo"]
    scheduler.inprogress = test_case["initial"]["inprogress"]
    scheduler.complete = test_case["initial"]["complete"]
    scheduler.max_workers = test_case["initial"]["max_workers"]

    # Execute the test
    scheduler._review_in_progress_jobs()

    # Review the results
    assert scheduler.jobs == test_case["result"]["jobs"]
    assert scheduler.todo == test_case["result"]["todo"]
    assert scheduler.inprogress == test_case["result"]["inprogress"]
    assert scheduler.complete == test_case["result"]["complete"]


def test_review_in_progress_jobs_inprogress_and_running(monkeypatch):
    # test case
    job_id: str = str(uuid.uuid4())
    job = generate_adsp_meta_job(id=job_id)
    mock_run: MagicMock = MagicMock()
    mock_run.get_status = MagicMock(return_value=RunStatus.RUNNING)
    job.runs.append(mock_run)

    test_case = {
        "initial": {"jobs": [job], "todo": [], "inprogress": [job_id], "complete": [], "max_workers": 3},
        "result": {"todo": [], "inprogress": [job_id], "complete": []},
    }

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    # Set up the test
    scheduler = Scheduler()
    scheduler.jobs = test_case["initial"]["jobs"]
    scheduler.todo = test_case["initial"]["todo"]
    scheduler.inprogress = test_case["initial"]["inprogress"]
    scheduler.complete = test_case["initial"]["complete"]
    scheduler.max_workers = test_case["initial"]["max_workers"]

    # Execute the test
    scheduler._review_in_progress_jobs()

    # Review the results
    assert scheduler.todo == test_case["result"]["todo"]
    assert scheduler.inprogress == test_case["result"]["inprogress"]
    assert scheduler.complete == test_case["result"]["complete"]


def test_review_in_progress_jobs_inprogress_and_finished(monkeypatch):
    # test case
    job_id: str = str(uuid.uuid4())
    job = generate_adsp_meta_job(id=job_id)
    mock_run: MagicMock = MagicMock()
    mock_run.get_status = MagicMock(return_value=RunStatus.FINISHED)
    job.runs.append(mock_run)

    test_case = {
        "initial": {"jobs": [job], "todo": [], "inprogress": [job_id], "complete": [], "max_workers": 3},
        "result": {"todo": [], "inprogress": [], "complete": [job_id]},
    }

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    # Set up the test
    scheduler = Scheduler()
    scheduler.jobs = test_case["initial"]["jobs"]
    scheduler.todo = test_case["initial"]["todo"]
    scheduler.inprogress = test_case["initial"]["inprogress"]
    scheduler.complete = test_case["initial"]["complete"]
    scheduler.max_workers = test_case["initial"]["max_workers"]

    # Execute the test
    scheduler._review_in_progress_jobs()

    # Review the results
    assert scheduler.todo == test_case["result"]["todo"]
    assert scheduler.inprogress == test_case["result"]["inprogress"]
    assert scheduler.complete == test_case["result"]["complete"]


def test_review_in_progress_jobs_inprogress_and_failed(monkeypatch):
    # test case
    job_id: str = str(uuid.uuid4())
    job = generate_adsp_meta_job(id=job_id)
    mock_run: MagicMock = MagicMock()
    mock_run.get_status = MagicMock(return_value=RunStatus.FAILED)
    job.runs.append(mock_run)

    test_case = {
        "initial": {"jobs": [job], "todo": [], "inprogress": [job_id], "complete": [], "max_workers": 3},
        "result": {"todo": [job_id], "inprogress": [], "complete": []},
    }

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    # Set up the test
    scheduler = Scheduler()
    scheduler.jobs = test_case["initial"]["jobs"]
    scheduler.todo = test_case["initial"]["todo"]
    scheduler.inprogress = test_case["initial"]["inprogress"]
    scheduler.complete = test_case["initial"]["complete"]
    scheduler.max_workers = test_case["initial"]["max_workers"]

    # Execute the test
    scheduler._review_in_progress_jobs()

    # Review the results
    assert scheduler.todo == test_case["result"]["todo"]
    assert scheduler.inprogress == test_case["result"]["inprogress"]
    assert scheduler.complete == test_case["result"]["complete"]


def test_review_in_progress_jobs_inprogress_and_failed_too_many_times(monkeypatch):
    # test case
    job_id: str = str(uuid.uuid4())
    job = generate_adsp_meta_job(id=job_id)
    mock_run: MagicMock = MagicMock()
    mock_run.get_status = MagicMock(return_value=RunStatus.FAILED)
    job.runs.append(mock_run)
    job.runs.append(mock_run)
    job.runs.append(mock_run)

    test_case = {
        "initial": {"jobs": [job], "todo": [], "inprogress": [job_id], "complete": [], "max_workers": 3},
        "result": {"todo": [], "inprogress": [], "complete": [job_id]},
    }

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    # Set up the test
    scheduler = Scheduler()
    scheduler.jobs = test_case["initial"]["jobs"]
    scheduler.todo = test_case["initial"]["todo"]
    scheduler.inprogress = test_case["initial"]["inprogress"]
    scheduler.complete = test_case["initial"]["complete"]
    scheduler.max_workers = test_case["initial"]["max_workers"]

    # Execute the test
    scheduler._review_in_progress_jobs()

    # Review the results
    assert scheduler.todo == test_case["result"]["todo"]
    assert scheduler.inprogress == test_case["result"]["inprogress"]
    assert scheduler.complete == test_case["result"]["complete"]


def test_review_in_progress_gracefully_fails(monkeypatch):
    # test case
    job_id: str = str(uuid.uuid4())
    job = generate_adsp_meta_job(id=job_id)

    test_case = {"initial": {"jobs": [job], "todo": [], "inprogress": [job_id], "complete": [], "max_workers": 3}}

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    # Set up the test
    scheduler = Scheduler()
    scheduler.jobs = test_case["initial"]["jobs"]
    scheduler.todo = test_case["initial"]["todo"]
    scheduler.inprogress = test_case["initial"]["inprogress"]
    scheduler.complete = test_case["initial"]["complete"]
    scheduler.max_workers = test_case["initial"]["max_workers"]

    # Execute the test
    with pytest.raises(ADSPMLFlowPluginError) as context:
        scheduler._review_in_progress_jobs()
        assert str(context.value) == "Unable to find job run to review"


###############################################################################
# _mark_mlflow_run_as_failed Tests
###############################################################################


def test_mark_mlflow_run_as_failed(monkeypatch):
    # Set up test
    mock_run_id: str = str(uuid.uuid4())
    mock_start_run = MagicMock()
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    scheduler = Scheduler()

    # Execute the test
    scheduler._mark_mlflow_run_as_failed(run_id=mock_run_id)

    # Review the results
    mock_start_run.assert_called_once_with(run_id=mock_run_id, nested=True, tags={"ADSP_FAILURE": True})


###############################################################################
# _add_log_to_run Tests
###############################################################################


def test_add_log_to_run(monkeypatch, get_ae_user_session):
    # Set up test
    mock_mlflow_run_id: str = str(uuid.uuid4())
    mock_adsp_job_id: str = str(uuid.uuid4())
    mock_start_run = MagicMock()
    mock_log_text = MagicMock()
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "log_text", mock_log_text)
    scheduler = Scheduler()

    mock_log: str = "mock log data"

    def mock_get_log(run_id: str):
        return mock_log

    monkeypatch.setattr(ADSPSubmittedRun, "get_log", mock_get_log)
    mock_run = ADSPSubmittedRun(
        ae_session=get_ae_user_session, mlflow_run_id=mock_mlflow_run_id, adsp_job_id=mock_adsp_job_id, response={}
    )

    # Execute the test
    scheduler._add_log_to_run(run=mock_run)

    # Review the results
    mock_start_run.assert_called_once_with(run_id=mock_mlflow_run_id, nested=True)
    mock_log_text.assert_called_once_with(mock_log, artifact_file="job_log.txt")


def test_add_log_to_run_with_local_run(monkeypatch, get_ae_user_session):
    # Set up test
    mock_start_run = MagicMock()
    mock_log_text = MagicMock()
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "log_text", mock_log_text)
    scheduler = Scheduler()

    # Execute the test
    scheduler._add_log_to_run(run=MagicMock())

    # Review the results
    mock_start_run.assert_not_called()
    mock_log_text.assert_not_called()


###############################################################################
# process_work_queue Tests
###############################################################################


def test_process_work_queue_nothing_to_do():
    # Set up the test
    scheduler = Scheduler()

    # Execute the test
    results = scheduler.process_work_queue(steps=[])

    # Review the results
    assert len(results) == 0


def test_process_work_queue(monkeypatch, get_ae_user_session):
    # Set up the test
    mock_mlflow_run_id: str = str(uuid.uuid4())
    mock_adsp_job_id: str = str(uuid.uuid4())
    mock_request = Step()
    scheduler = Scheduler()

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    monkeypatch.setattr(mlflow_adsp.ADSPSubmittedRun, "get_status", MagicMock(return_value=RunStatus.FINISHED))

    mock_run = ADSPSubmittedRun(
        ae_session=get_ae_user_session, mlflow_run_id=mock_mlflow_run_id, adsp_job_id=mock_adsp_job_id, response={}
    )

    def mock_mlflow_run(**kwargs):
        return mock_run

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "execute_step", mock_mlflow_run)

    # Execute the test
    results = scheduler.process_work_queue(steps=[mock_request])

    # Review the results
    assert len(results) == 1

    result = results[0]
    assert result.step == mock_request
    assert result.last_status == RunStatus.FINISHED

    assert len(result.runs) == 1
    run = result.runs[0]
    assert run == mock_run


def test_process_work_queue_with_wait(monkeypatch, get_ae_user_session):
    # Set up the test
    mock_mlflow_run_id: str = str(uuid.uuid4())
    mock_adsp_job_id: str = str(uuid.uuid4())
    mock_request = Step()
    scheduler = Scheduler()

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    monkeypatch.setattr(
        mlflow_adsp.ADSPSubmittedRun, "get_status", MagicMock(side_effect=[RunStatus.RUNNING, RunStatus.FINISHED])
    )

    mock_run = ADSPSubmittedRun(
        ae_session=get_ae_user_session, mlflow_run_id=mock_mlflow_run_id, adsp_job_id=mock_adsp_job_id, response={}
    )

    def mock_mlflow_run(**kwargs):
        return mock_run

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "execute_step", mock_mlflow_run)

    # Execute the test
    results = scheduler.process_work_queue(steps=[mock_request])

    # Review the results
    assert len(results) == 1

    result = results[0]
    assert result.step == mock_request
    assert result.last_status == RunStatus.FINISHED

    assert len(result.runs) == 1
    run = result.runs[0]
    assert run == mock_run


def test_process_work_queue_with_multiple_requests(monkeypatch, get_ae_user_session):
    # Set up the test
    mock_mlflow_run_id_one: str = str(uuid.uuid4())
    mock_mlflow_run_id_two: str = str(uuid.uuid4())
    mock_adsp_job_id_one: str = str(uuid.uuid4())
    mock_adsp_job_id_two: str = str(uuid.uuid4())
    mock_request = Step()
    scheduler = Scheduler()
    scheduler.max_workers = 1

    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_mark_mlflow_run_as_failed", MagicMock())
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "_add_log_to_run", MagicMock())

    monkeypatch.setattr(
        mlflow_adsp.ADSPSubmittedRun,
        "get_status",
        MagicMock(side_effect=[RunStatus.RUNNING, RunStatus.FINISHED, RunStatus.RUNNING, RunStatus.FINISHED]),
    )

    mock_run_one = ADSPSubmittedRun(
        ae_session=get_ae_user_session,
        mlflow_run_id=mock_mlflow_run_id_one,
        adsp_job_id=mock_adsp_job_id_one,
        response={},
    )
    mock_run_two = ADSPSubmittedRun(
        ae_session=get_ae_user_session,
        mlflow_run_id=mock_mlflow_run_id_two,
        adsp_job_id=mock_adsp_job_id_two,
        response={},
    )
    mock_execute_step = MagicMock(side_effect=[mock_run_one, mock_run_two])
    monkeypatch.setattr(mlflow_adsp.common.scheduler.Scheduler, "execute_step", mock_execute_step)

    # Execute the test
    results = scheduler.process_work_queue(steps=[mock_request, mock_request])

    # Review the results
    assert len(results) == 2

    assert results[0].step == mock_request
    assert results[1].step == mock_request
    assert results[0].last_status == RunStatus.FINISHED
    assert results[1].last_status == RunStatus.FINISHED

    assert len(results[0].runs) == 1
    assert len(results[1].runs) == 1

    assert results[0].runs[0] == mock_run_two
    assert results[1].runs[0] == mock_run_one
