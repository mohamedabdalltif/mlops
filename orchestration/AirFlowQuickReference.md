## 🚀 Airflow Quick Reference (MLOps)

| Concept | What it is | In your code |
| :--- | :--- | :--- |
| **DAG** | The whole pipeline file | `with DAG(...) as dag:` |
| **Task** | One step, one function | `PythonOperator(task_id=..., python_callable=...)` |
| **`**context`** | Airflow injects this into every task | `def task_read_train(**context):` |
| **`logical_date`** | The date this specific run represents | `context["logical_date"]` |
| **`relativedelta`** | Go back N months from a date | `logical_date - relativedelta(months=2)` |
| **XCom push** | Save small data for the next task | `context["ti"].xcom_push(key="x", value=v)` |
| **XCom pull** | Read data the previous task saved | `ti.xcom_pull(key="x")` |
| **`>>` operator** | Set task order / dependencies | `[read_train, read_val] >> create_features` |
| **`catchup=True`** | Auto-run all past missed months | Set in `DAG(...)` |
| **`max_active_runs`** | Don't overlap two months | Set in `DAG(...)` |
| **`retries`** | Retry a failed task twice | In `default_args` or per task |