# ML Flows

## Get started

- Run MLFlow:

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
python main.py
```

- Open browser to `localhost:3000`