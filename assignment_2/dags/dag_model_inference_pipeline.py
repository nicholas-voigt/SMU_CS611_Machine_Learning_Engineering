from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor

from datetime import datetime, timedelta
import os

from configs.data import model_registry_dir


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_model_inference_pipeline',
    default_args=default_args,
    description='inference pipeline run once a month after data pipeline',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:
 
    # --- wait for data pipeline to complete ---

    label_store_completed = ExternalTaskSensor(
        task_id='wait_for_label_store_completed',
        external_dag_id='dag_data_pipeline',
        external_task_id='label_store_completed',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='reschedule',
        timeout=60 * 1,  # Wait for 1 minute
    )

    feature_store_completed = ExternalTaskSensor(
        task_id='wait_for_feature_store_completed',
        external_dag_id='dag_data_pipeline',
        external_task_id='feature_store_completed',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='reschedule',
        timeout=60 * 1,  # Wait for 1 minute
    )

    # --- check if trained models are available ---

    dep_check_trained_models = PythonOperator(
        task_id='dep_check_trained_models',
        python_callable=lambda: os.path.exists(os.path.join(model_registry_dir, "best_model"))
    )

    # --- model inference ---

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow/scripts/ml_processing && '
            'python model_inference.py '
            '--date "{{ ds }}"'
        )
    )

    # --- model monitoring ---

    model_monitoring = BashOperator(
        task_id='model_monitoring',
        bash_command=(
            'cd /opt/airflow/scripts/ml_processing && '
            'python model_monitoring.py '
            '--date "{{ ds }}"'
        )
    )

    # --- Task Dependencies ---

    label_store_completed >> model_inference # type: ignore
    feature_store_completed >> model_inference # type: ignore
    dep_check_trained_models >> model_inference # type: ignore
    model_inference >> model_monitoring # type: ignore
