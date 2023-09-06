"""
Our CLI Entry Point
Note that this is called by `mlflow-adsp` and exposed through that vector.
"""

import click

from mlflow_adsp.serve import serve
from mlflow_adsp.services.worker import worker


@click.group(name="mlflow-adsp")
def cli():
    """
    The click entry point group for the `mlflow-adsp` command.
    """


cli.add_command(cmd=serve)
cli.add_command(cmd=worker)
