# MLOps Project — NYC Taxi Duration Prediction

## Project Structure
```
mlops/
├── data/                      # raw data samples
├── experiment-tracking/       # MLflow experiment notebooks
├── intro/                     # course intro notebooks
└── orchestration/             # Airflow pipeline (this unit)
    ├── dags/
    │   └── taxi_training_dag.py   # main Airflow DAG
    ├── logs/                      # Airflow logs (auto-generated)
    ├── models/                    # saved models and preprocessor
    ├── docker-compose.yaml        # Airflow multi-container setup
    ├── requirements.txt           # ML dependencies
    └── .env                       # AIRFLOW_UID (auto-generated)
```

## Prerequisites
- Windows with WSL2 (Ubuntu)
- Docker installed inside WSL2

## How to run (inside WSL2 Ubuntu terminal)

### 1. Clone and enter the project
```bash
git clone https://github.com/mohamedabdalltif/mlops.git
cd mlops/orchestration
```

### 2. Create required folders
```bash
mkdir -p dags logs plugins models
echo "AIRFLOW_UID=$(id -u)" > .env
```

### 3. Start Airflow
```bash
docker compose up airflow-init   # first time only
docker compose up -d
```

Open http://localhost:8080 — login: airflow / airflow

### 4. Enable and trigger the DAG
In the UI, toggle **taxi_duration_training** ON and click the Run button.

### 5. Backfill past months
```bash
docker compose exec airflow-scheduler \
  airflow dags backfill \
  --start-date 2024-03-01 \
  --end-date 2024-06-01 \
  taxi_duration_training
```

### 6. Stop Airflow
```bash
docker compose down
```

## Pipeline steps
The DAG runs monthly and automatically computes:
- **Train data** = 2 months before run date
- **Val data** = 1 month before run date

| Task | What it does |
|---|---|
| `read_train_data` | Downloads green taxi parquet for train month |
| `read_val_data` | Downloads green taxi parquet for val month |
| `create_features` | Vectorizes features using DictVectorizer |
| `train_model` | Trains XGBoost, logs to MLflow |