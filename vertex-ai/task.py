import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(train_file_path, model_dir):
    logger.info(f"Loading data from: {train_file_path}")
    df = pd.read_csv(train_file_path)

    X = df.drop('mpg', axis=1)
    y = df['mpg']

    logger.info("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Model training complete.")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file-path', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('AIP_MODEL_DIR', '/tmp/model_output'))
    args = parser.parse_args()

    train_model(args.train_file_path, args.model_dir)
