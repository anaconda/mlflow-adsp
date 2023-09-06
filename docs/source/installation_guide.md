#  Installation Guide

## Process

### 1. Configure Environment

The below variables control authorization to the Anaconda Data Science Platform.
These should be defined as AE5 secrets for the current user.  Alternatively they can also be set within the `anaconda-project.yml` project files.
See `Variables` below for specific details on each.

| Variable              | Workflow | Serving |
|-----------------------|----------|---------|
| AE5_HOSTNAME          | X        |         |
| AE5_USERNAME          | X        |         |
| AE5_PASSWORD          | X        |         |
| ADSP_WORKER_MAX       | X        |         |
| MLFLOW_TRACKING_URI   | X[1]     | X[1]    |
| MLFLOW_REGISTRY_URI   | X[1]     | X[1]    |
| MLFLOW_TRACKING_TOKEN | X[2]     | X[2]    |

[1] It's recommended to set these AE5 installation wide as per [Adding MLflow to Anaconda Enterprise 5](https://enterprise-docs.anaconda.com/en/latest/admin/advanced/mlflow_install.html).

[2] This should be provided by your admin as per [Adding MLflow to Anaconda Enterprise 5](https://enterprise-docs.anaconda.com/en/latest/admin/advanced/mlflow_install.html).

### 2. Install Plugin

```shell
conda install -c https://conda.anaconda.org/ae5-admin mlflow-adsp 
```

## Variables

**Order of resolution**

1. Anaconda Project Variables
2. Anaconda Data Science User Secrets

If the below variables are defined within an Anaconda Project and as secrets within the Anaconda Data Science Platform, then the Secrets will have a higher precedence and be used. 


1. `AE5_HOSTNAME`

    **Description**
    
    * The FQDN (Fully Qualified Domain Name) for the Anaconda Data Science Platform instance.


2. `AE5_USERNAME`

    **Description**
    
    * The username of the account to create the jobs with.  This should be an account who owns the project or is a collaborator on the project.


3. `AE5_PASSWORD`

    **Description**

    * The password for the user account used for start jobs.


4. `ADSP_WORKER_MAX`

    **Description**

    * The default per-project number of background jobs the user will leverage during parallel execution.
