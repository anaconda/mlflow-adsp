import os
import shlex
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def shell_out(shell_out_cmd: str) -> tuple[str, str, int]:
    args = shlex.split(shell_out_cmd)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
    return outs.decode(encoding="utf-8"), errs.decode(encoding="utf-8"), proc.returncode


if __name__ == "__main__":
    # load locally defined environmental variables
    local_env_config: str = (Path(os.path.dirname(os.path.realpath(__file__))) / "env.offline").as_posix()
    load_dotenv(dotenv_path=local_env_config, override=True)  # take environment variables from .env.

    # Start Tests
    cmd: str = (
        "python -m pytest --cov-branch --cov=mlflow_adsp -vv --cov-append --cov-report=xml --show-capture=all -rP test/unit"
    )
    stdout, stderr, returncode = shell_out(shell_out_cmd=cmd)

    # Report Test Results
    print(stdout)
    print(stderr)
    if returncode != 0:
        raise Exception("Unit Test Execute Failed")
