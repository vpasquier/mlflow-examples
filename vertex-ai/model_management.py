from google.cloud import aiplatform
import joblib
import os

# ----------- CONFIGURATION ----------------
PROJECT_ID = "prolaio-data-testing"
LOCATION = "us-central1"
MODEL_DISPLAY_NAME = "my-sklearn-model"
ENDPOINT_DISPLAY_NAME = "my-sklearn-endpoint"
MODEL_FILE = "model.joblib"

# ----------- INITIALIZE VERTEX AI ---------------
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ----------- SAVE YOUR MODEL LOCALLY (SKLEARN EXAMPLE) ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier()
clf.fit(X, y)

joblib.dump(clf, MODEL_FILE)

# ----------- UPLOAD MODEL ARTIFACT -----------------
model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=f"gs://vpasquier-demo/model.joblib",
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest',
    serving_container_environment_variables={"SKLEARN_SERVER_TIMEOUT": "60"},
    serving_container_command=[],
    serving_container_args=[],
    model_file_path=MODEL_FILE
)

model.wait()
print(f"Model uploaded: {model.resource_name}")

# ----------- CREATE ENDPOINT ----------------------
endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
print(f"Endpoint created: {endpoint.resource_name}")

# ----------- DEPLOY MODEL TO ENDPOINT --------------
deployed_model = model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    traffic_percentage=100
)

print(f"Model deployed to endpoint: {endpoint.resource_name}")


# --- Online Prediction using endpoint api ---
# Get a sample from the test set (using the test_df we created earlier)
# print("\nSending prediction request...")
# predictions = endpoint.predict(instances=instances)
# print("\nPrediction results:")
# for i, pred in enumerate(predictions.predictions):
#     print(f"  Sample {i+1}: Predicted MPG = {pred[0]:.2f}, True MPG = {true_mpg[i]:.2f}")
