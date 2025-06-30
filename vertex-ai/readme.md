# Demo Vertex AI

## Notebook

https://6c81e36526af783a-dot-us-central1.notebooks.googleusercontent.com/lab/tree/Demo.ipynb

Here in `trainer`, we are having a custom model that trains on some data and can be launched into a job via the API in this notebook.

Notes:

- There is a management of dataset: https://console.cloud.google.com/vertex-ai/datasets?authuser=1&hl=en&inv=1&invt=Ab1iYg&project=prolaio-data-testing. But same than MLFlow or DVC as it uses GCS.

- It's very complicated to have the status of a training: you need to navigate to different UI that is not convenient to have all the jobs statuses

- There is a feature store to handle consistency in the code for training and predicting on giving features.

- It's also taking a lot of time to run with `INFO 2025-06-30T22:50:45.961046131Z [resource.labels.taskName: service] Vertex AI is provisioning job running framework. First time usage might take couple of minutes, and subsequent runs can be much faster.`

- Experiments are the new feature of Vertex AI which is exactly what MLFlow provides except its not part of the pipelines themselves and the api is clunky. It uses also Tensorboard which is a poorer UI than MLFlow.

- Pipelines: see `pipeline.py`. The job is being run programmatically meaning not very convenient, need a google function: see Pipeline Deployment instructions beneath. Otherwise same benefits than metaflow. No dev experience locally.

- Vertex AI uses also Ray to do multitask computing framework (https://docs.ray.io/en/latest/ray-overview/getting-started.html) - it would be like using Metaflow but without the flexibility of the Metaflow dev framework

Globally:
- You cannot run jobs/pipelines/model training/model comparaison locally - always in the cloud console. Super Slow.
- Everything is "disconnected" to see all your jobs, you need to navigate in different places to see the logs etc... not convenient
- Seems much more expensive than k8s
- The serving might be nice to register first the model being developped in MLFlow and then deployed behind a endpoint managed by Google (see Model Management beneath)

## Model management

With another example, we can only use the model registry and serving ability of Vertex AI with the `model_management.py`

## Pipeline Deployment Instructions

Cloud Function Example to Trigger a Pipeline:

```
from google.cloud import aiplatform

def trigger_pipeline(request):
    aiplatform.init(project="YOUR_PROJECT_ID", location="YOUR_REGION")

    job = aiplatform.PipelineJob(
        display_name="scheduled-pipeline-job",
        template_path="gs://YOUR_BUCKET/simple_pipeline.json",
        pipeline_root="gs://YOUR_BUCKET/pipeline-root/",
    )

    job.run(sync=False)  # async run
    return "Pipeline triggered", 200
```
Cloud Scheduler Example:

Create a Cloud Scheduler job to make an HTTP request to that Cloud Function on a cron schedule.

```
gcloud scheduler jobs create http pipeline-scheduler \
  --schedule="0 9 * * *" \  # Every day at 9 AM
  --uri="https://REGION-PROJECT_ID.cloudfunctions.net/trigger_pipeline" \
  --http-method=POST
```

Event driven the same.