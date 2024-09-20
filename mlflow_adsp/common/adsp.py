""" Authentication / Authorization Helper Functions """

import os

from ae5_tools import demand_env_var, load_ae5_user_secrets
from ae5_tools.api import AEUserSession

from ..contracts.errors.plugin import ADSPMLFlowPluginError


def create_session() -> AEUserSession:
    """
    This function is responsible for pulling Anaconda Data Science Platform credentials out
    of the environment definition and creating an instance of a session.

    Returns
    -------
    session: AEUserSession
        An instance of an Anaconda Data Science Platform user session.
    """

    # Load defined environmental variables
    load_ae5_user_secrets()

    # Create the session directly and provide the AE5 config and credentials:
    ae_session: AEUserSession = AEUserSession(
        hostname=demand_env_var(name="AE5_HOSTNAME"),
        username=demand_env_var(name="AE5_USERNAME"),
        password=demand_env_var(name="AE5_PASSWORD"),
    )

    # Connect to Anaconda Data Science Platform
    # This is currently accomplished this by accessing a private method.
    # pylint: disable=protected-access
    ae_session._connect(password=ae_session)
    return ae_session


def get_project_id() -> str:
    """
    Inspected the run-time environment for the project Id.

    Returns
    -------
    project_id: str
        The project ID for the Anaconda Data Science Platform project which owns the runtime context.
    """

    if "TOOL_PROJECT_URL" in os.environ:
        # When executing within a session, this seems to be the most reliable method for getting a context id.
        var_name: str = "TOOL_PROJECT_URL"
    elif "APP_SOURCE" in os.environ:
        # When executing within a scheduled job this seems to be the most reliable method for getting a context id.
        var_name: str = "APP_SOURCE"
    else:
        raise ADSPMLFlowPluginError("Unable to determine execution context.  Did this execute in ADSP?")
    return f"a0-{demand_env_var(name=var_name).split(sep='/')[4]}"
