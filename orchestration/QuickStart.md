# Start Airflow
cd ~/mlops/orchestration && docker compose up -d

# Check all containers are healthy
docker compose ps

# See logs if something fails
docker compose logs airflow-worker
docker compose logs airflow-scheduler

# Run a backfill
docker compose exec airflow-scheduler \
  airflow dags backfill --start-date 2024-03-01 --end-date 2024-05-01 taxi_duration_training

# Stop everything
docker compose down

# Nuclear reset (deletes DB, start fresh)
docker compose down --volumes --remove-orphans