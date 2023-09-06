import os
import warnings

import pytest

from ae5_tools.api import AEUserSession
from mlflow_adsp import ADSPMLFlowPluginError, create_session, get_project_id

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MockState(object):
    calls = []
    responses = []

    @classmethod
    def mockreturn(cls, *args, **kwargs):
        cls.calls.append([args, kwargs])
        return cls.responses.pop()

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.responses = []


def test_create_session(monkeypatch):
    # Setup Test
    MockState.reset()
    MockState.responses = [None]
    monkeypatch.setattr(AEUserSession, "_connect", MockState.mockreturn)

    # Execute Test
    session: AEUserSession = create_session()

    # Review Test Outcome
    assert len(MockState.calls) == 1

    generated_session: AEUserSession = MockState.calls[0][1]["password"]
    assert session.hostname == generated_session.hostname == os.environ["AE5_HOSTNAME"]
    assert session.username == generated_session.username == os.environ["AE5_USERNAME"]
    assert session.password == generated_session.password == os.environ["AE5_PASSWORD"]


###############################################################################
# _get_project_id Tests
###############################################################################


def test_get_project_id_in_job():
    project_id: str = get_project_id()
    assert project_id == "a0-mock-tool-project-url-id"


def test_get_project_id_in_session():
    if "TOOL_PROJECT_URL" in os.environ:
        del os.environ["TOOL_PROJECT_URL"]
    project_id: str = get_project_id()
    assert project_id == "a0-mock-app-source-id"


def test_get_project_id_should_gracefully_fail():
    if "TOOL_PROJECT_URL" in os.environ:
        del os.environ["TOOL_PROJECT_URL"]
    if "APP_SOURCE" in os.environ:
        del os.environ["APP_SOURCE"]
    with pytest.raises(ADSPMLFlowPluginError) as context:
        get_project_id()
    assert str(context.value) == "Unable to determine execution context.  Did this execute in ADSP?"
