import mlflow.data
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:3000")
mlflow.set_experiment("Dataset Tracking Demo")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Load your data
dataset_source_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
raw_data = pd.read_csv(dataset_source_url, delimiter=";")

print(f"Loaded dataset with shape: {raw_data.shape}")
print(f"Columns: {list(raw_data.columns)}")

# Create a Dataset object
dataset = mlflow.data.from_pandas(
    raw_data, source=dataset_source_url, name="wine-quality-white-v2", targets="quality"
)

print(f"Created MLflow dataset: {dataset.name}")
print(f"Dataset source: {dataset.source}")
print(f"Dataset targets: {dataset.targets}")

# Log the dataset to an MLflow run
with mlflow.start_run(run_name="wine-quality-dataset-tracking"):
    mlflow.log_input(dataset, context="training")
    
    print("Logged dataset to MLflow run")
    
    # Your training code here
    # Split the data
    train, test = train_test_split(raw_data, test_size=0.25, random_state=42)
    
    # Prepare features and target
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    # Train model
    model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    model.fit(train_x, train_y)
    
    # Make predictions
    predicted_qualities = model.predict(test_x)
    
    # Calculate metrics
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    
    print("Elasticnet model (alpha=0.5, l1_ratio=0.5):")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    print("Model logged successfully")
    
    # Show dataset information
    print("\nDataset Information:")
    print(f"  Name: {dataset.name}")
    print(f"  Source: {dataset.source}")
    print(f"  Targets: {dataset.targets}")
    print(f"  Schema: {dataset.schema}")
    print(f"  Profile: {dataset.profile}") 