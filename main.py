import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:3000")
mlflow.set_experiment("IrisExperimentAgain")

with mlflow.start_run():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    print("Tracking URI:", mlflow.get_tracking_uri())
    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            name="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()

