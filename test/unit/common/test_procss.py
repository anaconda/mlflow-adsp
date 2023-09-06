from unittest.mock import MagicMock

import pytest

from mlflow_adsp import SubprocessFailureError, process_launch_wait


def test_process_launch_wait():
    # Execute the test
    process_launch_wait(cwd=".", shell_out_cmd="python -m test.fixtures.worker.success")


def test_process_launch_wait_with_failure():
    # Execute the test
    with pytest.raises(SubprocessFailureError) as context:
        process_launch_wait(cwd=".", shell_out_cmd="python -m test.fixtures.worker.fail")

    # Review the results
    assert (
        str(context.value)
        == '{"command": "python -m test.fixtures.worker.fail", "message": "subprocess failed", "returncode": 1}'
    )
