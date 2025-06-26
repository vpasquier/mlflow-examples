import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:3000")
mlflow.set_experiment("Dataset Tracking Example")

def eval_metrics(actual, pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    # Dataset source URL (using the same wine quality dataset)
    dataset_source_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
    
    logger.info(f"Loading data from: {dataset_source_url}")
    
    # Load your data
    raw_data = pd.read_csv(dataset_source_url, delimiter=";")
    
    logger.info(f"Loaded dataset with shape: {raw_data.shape}")
    logger.info(f"Columns: {list(raw_data.columns)}")
    
    # Create a Dataset object with mlflow.data
    dataset = mlflow.data.from_pandas(
        raw_data, 
        source=dataset_source_url, 
        name="wine-quality-white", 
        targets="quality"
    )
    
    logger.info(f"Created MLflow dataset: {dataset.name}")
    logger.info(f"Dataset source: {dataset.source}")
    logger.info(f"Dataset targets: {dataset.targets}")
    
    # Start MLflow run and log the dataset
    with mlflow.start_run(run_name="wine-quality-with-dataset-tracking"):
        
        # Log the dataset to the MLflow run
        mlflow.log_input(dataset, context="training")
        
        logger.info("Logged dataset to MLflow run")
        
        # Log dataset metadata
        mlflow.log_param("dataset_name", dataset.name)
        mlflow.log_param("dataset_source", dataset.source)
        mlflow.log_param("dataset_rows", raw_data.shape[0])
        mlflow.log_param("dataset_columns", raw_data.shape[1])
        mlflow.log_param("target_column", "quality")
        
        # Split the data into training and test sets
        train, test = train_test_split(raw_data, test_size=0.25, random_state=42)
        
        # Prepare features and target
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]
        
        logger.info(f"Training set shape: {train_x.shape}")
        logger.info(f"Test set shape: {test_x.shape}")
        
        # Model hyperparameters
        alpha = 0.5
        l1_ratio = 0.5
        
        # Train the model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        
        # Make predictions
        predicted_qualities = lr.predict(test_x)
        
        # Calculate metrics
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        
        logger.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        logger.info("  RMSE: %s" % rmse)
        logger.info("  MAE: %s" % mae)
        logger.info("  R2: %s" % r2)
        
        # Log model parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Log model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(
                lr, 
                "model", 
                registered_model_name="WineQualityModelWithDatasetTracking"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")
        
        logger.info("Model logged successfully")
        
        # Demonstrate dataset lineage
        logger.info("Dataset lineage information:")
        logger.info(f"  - Dataset name: {dataset.name}")
        logger.info(f"  - Dataset source: {dataset.source}")
        logger.info(f"  - Dataset schema: {dataset.schema}")
        logger.info(f"  - Dataset profile: {dataset.profile}")
        
        return {
            "dataset": dataset,
            "model": lr,
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2}
        }

if __name__ == "__main__":
    main() 