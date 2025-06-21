import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:3000")
mlflow.set_experiment("MLflow test")

with mlflow.start_run():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=25)

    # Define the model hyperparameters
    # params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}

     # Modified hyperparameters
    params = {
        "solver": "saga",              # Better for large dataset or L1 regularization
        "penalty": "l2",               # Explicit regularization
        "C": 0.5,                      # Regularization strength (lower -> stronger regularization)
        "max_iter": 2,              # Increase iterations to ensure convergence
        "multi_class": "multinomial", # Explicitly using multinomial strategy
        "random_state": 8888
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate accuracy as a target loss metric
    accuracy = accuracy_score(y_test, y_pred)
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))
    print("Tracking URI:", mlflow.get_tracking_uri())

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names

    # Convert X_test validation feature data to a Pandas DataFrame
    result = pd.DataFrame(X_test, columns=iris_feature_names)

    # Add the actual classes to the DataFrame
    result["actual_class"] = y_test

    # Add the model predictions to the DataFrame
    result["predicted_class"] = predictions

    result[:4]

