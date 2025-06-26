# ML Tools Spikes

## MLFlow Only

### With embedded Mlflow server

```bash
mlflow ui -p 3000
```

In another shell window:

```bash
python mlflow-only.py
```

### With Docker compose

- Run MLFlow

```bash
docker-compose up --build
```

- Open IDE on the current root folder

- Get inside the MLFlow container:

```bash
docker exec -it $(docker ps -q | head -n 1) bash
```

- Run the script:

```bash
python mlflow-only.py
```

- Open browser to `localhost:3000`