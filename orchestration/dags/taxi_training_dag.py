# =============================================================================
# dags/taxi_training_dag.py
# =============================================================================

# ── standard library ──────────────────────────────────────────────────────────
import os
import pickle
from pathlib import Path

# ── data / ml ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from dateutil.relativedelta import relativedelta

# ── airflow ───────────────────────────────────────────────────────────────────
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
# PythonOperator is what wraps a Python function into an Airflow task


# =============================================================================
# SECTION 1 — MLflow + folder setup
# This runs when the DAG file is loaded, not when a task runs.
# host.docker.internal lets the container reach your Windows MLflow server.
# =============================================================================


Path("models").mkdir(exist_ok=True)


# =============================================================================
# SECTION 2 — YOUR ORIGINAL FUNCTIONS (completely unchanged)
# These are the exact same functions from your train.py.
# We do NOT change them. We just call them from inside Airflow tasks below.
# =============================================================================

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    # ✅ MLFLOW SETUP MOVED HERE
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("/opt/airflow/models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(
            "/opt/airflow/models/preprocessor.b",
            artifact_path="preprocessor"
        )
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


# =============================================================================
# SECTION 3 — AIRFLOW TASK FUNCTIONS
# These are NEW wrapper functions. Each one:
#   1. accepts **context  (required by Airflow — never remove this)
#   2. reads the logical_date from context to compute the right month
#   3. calls your original function
#   4. pushes results to XCom so the next task can read them
# =============================================================================

def task_read_train(**context):
    """
    Task 1: Download the training dataframe.
    Train month = logical_date minus 2 months.
    Example: if logical_date = June 1 → train month = April
    """
    logical_date = context["logical_date"]
    train_date = logical_date - relativedelta(months=2)

    print(f"[task_read_train] logical_date={logical_date.date()}")
    print(f"[task_read_train] fetching train data for {train_date.year}-{train_date.month:02d}")

    # call YOUR original function — nothing changed inside it
    df = read_dataframe(year=train_date.year, month=train_date.month)

    print(f"[task_read_train] shape: {df.shape}")

    # save to shared volume (XCom is too slow/small for DataFrames)
    df.to_parquet("/opt/airflow/models/train_df.parquet")


def task_read_val(**context):
    """
    Task 2: Download the validation dataframe.
    Val month = logical_date minus 1 month.
    Example: if logical_date = June 1 → val month = May
    """
    logical_date = context["logical_date"]
    val_date = logical_date - relativedelta(months=1)

    print(f"[task_read_val] logical_date={logical_date.date()}")
    print(f"[task_read_val] fetching val data for {val_date.year}-{val_date.month:02d}")

    df = read_dataframe(year=val_date.year, month=val_date.month)

    # save to shared volume
    df.to_parquet("/opt/airflow/models/val_df.parquet")


def task_create_features(**context):
    """
    Task 3: Vectorize features from both dataframes.
    Pulls train_df and val_df from XCom.
    Saves X matrices to disk (too big for XCom).
    Pushes y arrays via XCom (small enough — just a list of numbers).
    """
    ti = context["ti"]

    # load the dataframes from the shared volume
    df_train = pd.read_parquet("/opt/airflow/models/train_df.parquet")
    df_val   = pd.read_parquet("/opt/airflow/models/val_df.parquet")

    print(f"[task_create_features] train shape: {df_train.shape}")
    print(f"[task_create_features] val shape:   {df_val.shape}")

    # call YOUR original function
    X_train, dv = create_X(df_train)        # dv is fit on train
    X_val, _    = create_X(df_val, dv)      # dv is only transformed on val

    # X_train and X_val are sparse matrices — too big for XCom
    # save them to the shared /opt/airflow/models/ volume instead
    sp.save_npz("/opt/airflow/models/X_train.npz", X_train)
    sp.save_npz("/opt/airflow/models/X_val.npz",   X_val)

    # save the fitted vectorizer (needed by train task for mlflow artifact)
    with open("/opt/airflow/models/dv.pkl", "wb") as f:
        pickle.dump(dv, f)

    # y values are just lists of floats — small enough for XCom
    ti.xcom_push(key="y_train", value=df_train["duration"].tolist())
    ti.xcom_push(key="y_val",   value=df_val["duration"].tolist())

    print("[task_create_features] features saved to /opt/airflow/models/")


def task_train_model(**context):
    """
    Task 4: Train XGBoost and log to MLflow.
    Loads X matrices from disk.
    Pulls y arrays from XCom.
    Calls your original train_model() function.
    Saves run_id to disk for reference.
    """
    ti = context["ti"]

    # load the sparse matrices saved by task 3
    X_train = sp.load_npz("/opt/airflow/models/X_train.npz")
    X_val   = sp.load_npz("/opt/airflow/models/X_val.npz")

    # load the vectorizer saved by task 3
    with open("/opt/airflow/models/dv.pkl", "rb") as f:
        dv = pickle.load(f)

    # pull y arrays from XCom and convert back to numpy arrays
    y_train = np.array(ti.xcom_pull(key="y_train"))
    y_val   = np.array(ti.xcom_pull(key="y_val"))

    print(f"[task_train_model] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[task_train_model] X_val:   {X_val.shape},   y_val:   {y_val.shape}")

    # call YOUR original function — nothing changed inside it
    run_id = train_model(X_train, y_train, X_val, y_val, dv)

    print(f"[task_train_model] MLflow run_id: {run_id}")

    # save run_id to file and push to XCom for reference
    with open("/opt/airflow/models/run_id.txt", "w") as f:
        f.write(run_id)

    ti.xcom_push(key="run_id", value=run_id)


# =============================================================================
# SECTION 4 — THE DAG DEFINITION
# This block defines the pipeline: its schedule, its tasks, and their order.
# Everything inside `with DAG(...) as dag:` belongs to this pipeline.
# =============================================================================

with DAG(
    dag_id="taxi_duration_training",

    # When to start scheduling. Airflow runs for every interval from this
    # date onward. Use a fixed past date — never use datetime.now().
    start_date=pendulum.datetime(2024, 3, 1, tz="UTC"),

    # Cron expression: "0 0 1 * *" = midnight on the 1st of every month
    # You can use @monthly instead but cron is more explicit
    schedule="0 0 1 * *",

    # catchup=True: when you first enable the DAG, Airflow runs all past
    # months from start_date to today automatically. This is backfilling.
    catchup=True,

    # max_active_runs=1: don't run two months in parallel
    # (important because all tasks write to the same models/ folder)
    max_active_runs=1,

    # default_args apply to every task unless overridden on the task itself
    default_args={
        "retries": 2,           # retry a failed task 2 times before giving up
    },

    tags=["ml", "taxi", "xgboost"],
) as dag:

    # ── Define the 4 tasks ────────────────────────────────────────────────────
    # PythonOperator takes:
    #   task_id   → the name shown in the Airflow UI
    #   python_callable → the wrapper function to call

    read_train = PythonOperator(
        task_id="read_train_data",
        python_callable=task_read_train,
    )

    read_val = PythonOperator(
        task_id="read_val_data",
        python_callable=task_read_val,
    )

    create_features = PythonOperator(
        task_id="create_features",
        python_callable=task_create_features,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    # ── Wire up the dependencies ──────────────────────────────────────────────
    # The >> operator means "must finish before"
    # [A, B] >> C means "both A and B must finish before C starts"
    #
    # read_train ──┐
    #              ├──► create_features ──► train_model
    # read_val   ──┘

    [read_train, read_val] >> create_features >> train_model_task