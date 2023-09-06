## Workflow Usage

1. Update usages of `mlflow.projects.run` to leverage the new backend plugin.

* MLFlow documentation for this command is located within [mlflow.projects.run Documentation](https://mlflow.org/docs/2.3.0/python_api/mlflow.projects.html#mlflow.projects.run).

**Paramater Changes**

When using `mlflow.projects.run` ensure to set the below parameters:

* `backend` = `adsp`
* `env_manager` = `local`

**Example**

```python
import mlflow
import uuid
 
with mlflow.start_run(run_name=f"training-{str(uuid.uuid4())}", nested=True) as run:  
   project_run = mlflow.projects.run(
      uri = ".",
      entry_point = "workflow_step_entry_point",
      run_id = run.info.run_id,
      env_manager = "local",
      backend = "adsp",
      parameters = {
         "training_data": training_data
      },
      experiment_id = run.info.experiment_id,
      synchronous = True,
      backend_config = {
         "resource_profile": "default"
      }
   )
```

## Configuration Options

This plugin supports the MLFlow standard for `backend_config`.

The below options are supported:
 
1. Resource Profile Specification

   * resource_profile: str

   This can be used to define a resource profile to run the worker on.


   **Example Anaconda Data Science Platform Backend Configuration**
   
   ```json
   {
     "resource_profile": "large"
   }
   ```
