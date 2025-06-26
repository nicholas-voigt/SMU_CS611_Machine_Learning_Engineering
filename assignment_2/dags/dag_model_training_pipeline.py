from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

from datetime import datetime, timedelta

from utils.data import check_partition_availability
from configs.data import gold_data_dirs


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_model_training_pipeline',
    default_args=default_args,
    description='training pipeline runs after manual trigger',
    schedule='0 15 1 * *',  # At 00:15 on day-of-month 1 --> 1 hour after midnight since data pipeline has to run first
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 2),
    catchup=True,
) as dag:
 
    # --- check availability of data in feature store and label store ---

    check_label_store_availability = PythonOperator(
        task_id='check_label_store_availability',
        python_callable=lambda: check_partition_availability(
            store_dir=gold_data_dirs['label_store'], 
            start_date=datetime(2023, 1, 1),  # Adjust as needed
            end_date=datetime(2024, 4, 1)  # Adjust as needed
        )
    )

    check_feature_store_availability = PythonOperator(
        task_id='check_label_store_availability',
        python_callable=lambda: check_partition_availability(
            store_dir=gold_data_dirs['feature_store'], 
            start_date=datetime(2023, 1, 1),  # Adjust as needed
            end_date=datetime(2024, 4, 1)  # Adjust as needed
        )
    )

    # --- prepare data for model training ---

    prepare_data = BashOperator(
        task_id='prepare_data',
        bash_command=(
            'cd /opt/airflow/scripts/ml_processing && '
            'python prepare_data.py --date "{{ ds }}"'
        ),
    )

    # --- model training ---

    model_automl_start = EmptyOperator(task_id="model_inference") # necessary?

    model_xgb_train = BashOperator(
        task_id='model_xgb_train',
        bash_command=(
            'cd /opt/airflow/scripts/ml_processing && '
            'python model_trainer.py --model_type gbt'
        ),
    )

    model_logreg_train = BashOperator(
        task_id='model_logreg_train',
        bash_command=(
            'cd /opt/airflow/scripts/ml_processing && '
            'python model_trainer.py --model_type logreg'
        ),
    )

    model_automl_completed = EmptyOperator(task_id="model_automl_completed")

    # --- Task Dependencies ---

    check_label_store_availability >> prepare_data # type: ignore
    check_feature_store_availability >> prepare_data # type: ignore

    model_automl_start >> model_xgb_train >> model_automl_completed # type: ignore
    model_automl_start >> model_logreg_train >> model_automl_completed # type: ignore
