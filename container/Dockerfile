FROM --platform=linux/amd64 registry.access.redhat.com/ubi8/ubi:latest@sha256:d5e2d1ddf34b573673581940f1341c7b3301ff8efde28f17100b31a3df7d94b6

# Perform Environment Maintenance
RUN yum update && yum install -y gcc clang cmake

# miniconda
# https://docs.conda.io/en/latest/miniconda.html
# amd/x86
#   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# aarch
#   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
# ppc64le
#   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
# s390x
#   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-s390x.sh

RUN mkdir -p ~/miniconda3
RUN cd ~/miniconda3 && curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ~/miniconda3/Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/Miniconda3-latest-Linux-x86_64.sh
RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda init zsh
RUN ~/miniconda3/bin/conda config --set auto_activate_base true

RUN source ~/.bashrc && conda update conda
RUN source ~/.bashrc && conda install -c defaults conda-build setuptools anaconda-project -y

# Configure Conda Runtime Environment
ADD container/conda.yml /tmp/build/
RUN source ~/.bashrc && conda env update --name base --file /tmp/build/conda.yml --prune

# MLflow Environmental Configuration
ENV MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING=TRUE

ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

# Application serving hook
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["mlflow-adsp serve"]
