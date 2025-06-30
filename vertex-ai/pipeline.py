from kfp.v2 import dsl
from kfp.v2.dsl import component
from google.cloud import aiplatform

# Define a simple component - step 1
@component
def step_one() -> str:
    message = "Hello from step one!"
    print(message)
    return message

# Define a simple component - step 2, which takes input from step 1
@component
def step_two(input_message: str):
    print(f"Received message in step two: {input_message}")

# Define the pipeline using the components above
@dsl.pipeline(
    name="simple-vertex-ai-pipeline",
    description="A simple pipeline with two steps on Vertex AI"
)
def simple_pipeline():
    step1_task = step_one()
    step2_task = step_two(input_message=step1_task.output)

# Initialize the AI Platform client
aiplatform.init(project="prolaio-data-testing", location="us-central1")

# Compile the pipeline to a JSON file
from kfp.v2 import compiler
compiler.Compiler().compile(
    pipeline_func=simple_pipeline,
    package_path="simple_pipeline.json"
)

# Submit the pipeline job to Vertex AI
job = aiplatform.PipelineJob(
    display_name="simple-pipeline-job",
    template_path="simple_pipeline.json",
    pipeline_root="gs://vpasquier-demo/pipeline-root/",  # GCS bucket path for pipeline artifacts
    location="us-central1",
)

job.run()
