from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 2,  # Increased retries for robustness
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_data_pipeline',
    default_args=default_args,
    description='data pipeline to run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # --- label store ---

    # check if source label data is available
    dep_check_source_label_data = PythonOperator(
        task_id='dep_check_source_label_data',
        python_callable=lambda: print("Checking if source label data is available...")
        # This is a placeholder; replace with actual check logic if needed
    )

    # run bronze label store script
    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python bronze_processing.py '
            '--date "{{ ds }}" --input_path ../../data/lms_loan_daily.csv --bronze_name label_store'
        ),
    )

    # run silver label store script
    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python silver_processing.py '
            '--type loan_data --date "{{ ds }}" --bronze_directory ../../datamart/bronze/label_store --silver_directory ../../datamart/silver/label_store'
        ),
    )

    # run gold label store script
    gold_label_store = BashOperator(
        task_id="run_gold_label_store",
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python gold_processing_label.py '
            '--date "{{ ds }}" --silver_directory ../../datamart/silver/label_store --gold_directory ../../datamart/gold/label_store'
        ),
    )

    label_store_completed = PythonOperator(
        task_id="label_store_completed",
        python_callable=lambda: print(f"\n\n---Label store processing completed for date: {{ ds }}---\n\n")
        # This is a placeholder; replace with actual completion logic if needed
    )

    # DAG task dependencies
    dep_check_source_label_data.set_downstream(bronze_label_store)
    bronze_label_store.set_downstream(silver_label_store)
    silver_label_store.set_downstream(gold_label_store)
    gold_label_store.set_downstream(label_store_completed)
 
 
    # --- feature store ---
    dep_check_source_data_bronze_1 = EmptyOperator(task_id="dep_check_source_data_bronze_1")

    dep_check_source_data_bronze_2 = EmptyOperator(task_id="dep_check_source_data_bronze_2")

    dep_check_source_data_bronze_3 = EmptyOperator(task_id="dep_check_source_data_bronze_3")

    bronze_table_1 = EmptyOperator(task_id="bronze_table_1")
    
    bronze_table_2 = EmptyOperator(task_id="bronze_table_2")

    bronze_table_3 = EmptyOperator(task_id="bronze_table_3")

    silver_table_1 = EmptyOperator(task_id="silver_table_1")

    silver_table_2 = EmptyOperator(task_id="silver_table_2")

    gold_feature_store = EmptyOperator(task_id="gold_feature_store")

    feature_store_completed = EmptyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2 >> gold_feature_store
    gold_feature_store >> feature_store_completed
