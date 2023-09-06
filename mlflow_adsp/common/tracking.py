""" MLFlow Tracking Server Helpers """

import secrets
import string
from typing import Optional

import mlflow
from mlflow.entities import Experiment

from ae5_tools import get_env_var


def _resolve_experiment_name(name: Optional[str] = None) -> str:
    """
    Resolve the experiment name to track to.
    If a name is not provided we look for an environment variable called `MLFLOW_EXPERIMENT_NAME`.
    If one is defined the value is used, otherwise we fall back to `Default`.

    Parameters
    ----------
    name: Optional[str]
        The experiment name.

    Returns
    -------
    name: str
        The resolved experiment name.
    """

    if name is None:
        # Attempt to load the name from environment variable
        name: Optional[str] = get_env_var(name="MLFLOW_EXPERIMENT_NAME")
        name = "Default" if name is None else name
    return name


def upsert_experiment(name: Optional[str] = None) -> str:
    """
    This function returns the experiment id for the provided experiment name.
    If the experiment does not exist, it is created.

    Parameters
    ----------
    name: Optional[str]
        The experiment name.

    Returns
    -------
    experiment_id: str
        The experiment ID.
    """

    # Resolve the experiment name.
    name: str = _resolve_experiment_name(name=name)

    experiment: Optional[Experiment] = mlflow.get_experiment_by_name(name=name)
    if experiment is None:
        # Then the experiment does not exist and needs to be created.
        experiment_id: str = mlflow.create_experiment(name=name)
    else:
        experiment_id: str = experiment.experiment_id

    return experiment_id


def create_unique_name(name: str) -> str:
    """
    Given a name will generate unique suffix and return the updated name.

    Parameters
    ----------
    name: str
        The original run name.

    Returns
    -------
    run_name: str
        Returns the run name with a unique suffix.
    """

    alphabet: str = string.ascii_letters + string.digits
    unique_suffix: str = "".join(secrets.choice(alphabet) for _ in range(10))
    return name + "-" + unique_suffix
