# MLflow Dataset Tracking Examples

This directory now includes comprehensive examples of MLflow dataset tracking capabilities using `mlflow.data`.

## What is Dataset Tracking?

Dataset tracking in MLflow allows you to:

- Track the lineage of datasets used in your ML experiments
- Log dataset metadata, schema, and profiles
- Link datasets to specific MLflow runs
- Maintain data provenance and reproducibility

## Files with Dataset Tracking

### 1. `dataset-tracking-example.py`

A focused example that demonstrates the exact pattern you requested:

```python
import mlflow.data
import pandas as pd

# Load your data
dataset_source_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
raw_data = pd.read_csv(dataset_source_url, delimiter=";")

# Create a Dataset object
dataset = mlflow.data.from_pandas(
    raw_data, source=dataset_source_url, name="wine-quality-white", targets="quality"
)

# Log the dataset to an MLflow run
with mlflow.start_run():
    mlflow.log_input(dataset, context="training")
    # Your training code here
```

### 2. `mlflow-dataset-tracking.py`

A comprehensive example that shows:

- Dataset creation and logging
- Complete ML pipeline with dataset tracking
- Dataset lineage information
- Detailed logging and metrics

### 3. `mlflow-only.py` (Updated)

Now includes dataset tracking for the Iris dataset:

- Creates a dataset object from sklearn's iris data
- Logs dataset metadata and lineage
- Maintains the original functionality

### 4. `mlflow-with-dvc.py` (Updated)

Enhanced with dataset tracking for DVC-managed data:

- Creates dataset objects from DVC-tracked wine quality data
- Logs dataset source and version information
- Combines DVC data versioning with MLflow dataset tracking

## Key Features Demonstrated

### Dataset Creation

```python
dataset = mlflow.data.from_pandas(
    raw_data,
    source=dataset_source_url,
    name="wine-quality-white",
    targets="quality"
)
```

### Dataset Logging

```python
with mlflow.start_run():
    mlflow.log_input(dataset, context="training")
```

### Dataset Metadata

```python
mlflow.log_param("dataset_name", dataset.name)
mlflow.log_param("dataset_source", dataset.source)
mlflow.log_param("dataset_rows", raw_data.shape[0])
mlflow.log_param("dataset_columns", raw_data.shape[1])
mlflow.log_param("target_column", "quality")
```

### Dataset Information Access

```python
print(f"Dataset name: {dataset.name}")
print(f"Dataset source: {dataset.source}")
print(f"Dataset targets: {dataset.targets}")
print(f"Dataset schema: {dataset.schema}")
print(f"Dataset profile: {dataset.profile}")
```

## Running the Examples

1. **Start MLflow tracking server:**

   ```bash
   mlflow server --host 127.0.0.1 --port 3000
   ```

2. **Run the dataset tracking examples:**
   ```bash
   python dataset-tracking-example.py
   python mlflow-dataset-tracking.py
   python mlflow-only.py
   python mlflow-with-dvc.py
   ```

## Benefits of Dataset Tracking

1. **Data Lineage**: Track where your data came from and how it was processed
2. **Reproducibility**: Ensure experiments can be reproduced with the same data
3. **Data Quality**: Monitor dataset characteristics and detect data drift
4. **Collaboration**: Share dataset information with team members
5. **Compliance**: Maintain audit trails for regulatory requirements

## MLflow UI Integration

When you run these examples, you can view the dataset information in the MLflow UI:

- Navigate to `http://127.0.0.1:3000`
- Select your experiment
- View run details to see dataset information
- Check the "Inputs" section for dataset lineage

## Requirements

Make sure you have the required dependencies:

```bash
pip install mlflow pandas scikit-learn numpy
```

For DVC integration:

```bash
pip install dvc
```
