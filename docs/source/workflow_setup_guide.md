#  Workflow Setup Guide

## Process

### 1. Add the Anaconda Project Worker Command:


    Worker: 
        env_spec: worker
        unix: mlflow-adsp worker


### 2. Ensure the `worker` environment contains at least:

     worker:
       description: Worker Environment
       packages:
         # Language Level
         - python>=3.8
   
         # MLFlow
         - mlflow>=2.3.0
         - make
         - virtualenv
         - pip
         - click
   
         # AE5
         - ipykernel
         - ae5-tools>=0.6.1
   
         # AE5 [MLFlow]
         - mlflow-adsp


