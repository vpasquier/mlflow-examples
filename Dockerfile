FROM ghcr.io/mlflow/mlflow:v3.0.0

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Note: main.py is now synced via volume mount in docker-compose.yml

