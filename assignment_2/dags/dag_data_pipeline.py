from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

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
            '--date "{{ ds }}" --type loan_data'
        ),
    )

    # run silver label store script
    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python silver_processing.py '
            '--date "{{ ds }}" --type loan_data'
        ),
    )

    # run gold label store script
    gold_label_store = BashOperator(
        task_id="run_gold_label_store",
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python gold_processing_label.py '
            '--date "{{ ds }}"'
        ),
    )

    label_store_completed = PythonOperator(
        task_id="label_store_completed",
        python_callable=lambda: print(f"\n\n---Label store processing completed for date: {{ ds }}---\n\n")
        # This is a placeholder; replace with actual completion logic if needed
    )

    # DAG task dependencies
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed # type: ignore

 
    # --- feature store ---

    # check if source for clickstream data is available
    dep_check_source_clickstream_data = PythonOperator(
        task_id='dep_check_source_clickstream_data',
        python_callable=lambda: print("Checking if source clickstream data is available...")
        # This is a placeholder; replace with actual check logic if needed
    )

    # check if source for customer attributes data is available
    dep_check_source_customer_attributes_data = PythonOperator(
        task_id='dep_check_source_customer_attributes_data',
        python_callable=lambda: print("Checking if source customer attributes data is available...")
        # This is a placeholder; replace with actual check logic if needed
    )

    # check if source for customer financial data is available
    dep_check_source_customer_financial_data = PythonOperator(
        task_id='dep_check_source_customer_financial_data',
        python_callable=lambda: print("Checking if source customer financial data is available...")
        # This is a placeholder; replace with actual check logic if needed
    )

    # run bronze clickstream store script
    bronze_clickstream_store = BashOperator(
        task_id='run_bronze_clickstream_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python bronze_processing.py '
            '--date "{{ ds }}" --type clickstream_data'
        ),
    )
    
    # run bronze customer attributes store script
    bronze_customer_attributes_store = BashOperator(
        task_id='run_bronze_customer_attributes_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python bronze_processing.py '
            '--date "{{ ds }}" --type customer_attributes'
        ),
    )

    # run bronze customer financials store script
    bronze_customer_financials_store = BashOperator(
        task_id='run_bronze_customer_financials_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python bronze_processing.py '
            '--date "{{ ds }}" --type customer_financials'
        ),
    )

    # run silver clickstream store script
    silver_clickstream_store = BashOperator(
        task_id='run_silver_clickstream_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python silver_processing.py '
            '--date "{{ ds }}" --type clickstream_data'
        ),
    )

    # run silver customer attributes store script
    silver_customer_attributes_store = BashOperator(
        task_id='run_silver_customer_attributes_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python silver_processing.py '
            '--date "{{ ds }}" --type customer_attributes'
        ),
    )

    # run silver customer financials store script
    silver_customer_financials_store = BashOperator(
        task_id='run_silver_customer_financials_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python silver_processing.py '
            '--date "{{ ds }}" --type customer_financials'
        ),
    )

    # join silver feature data to gold feature store
    gold_feature_store_join = BashOperator(
        task_id='run_join_gold_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts/data_processing && '
            'python gold_processing_feature_join.py '
            '--date "{{ ds }}"'
        ),
    )

    # Gold Feature store data engineering
    gold_feature_store_engineered = EmptyOperator(task_id="run_engineer_gold_feature_store")

    feature_store_completed = EmptyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_clickstream_data >> bronze_clickstream_store >> silver_clickstream_store >> gold_feature_store_join # type: ignore
    dep_check_source_customer_attributes_data >> bronze_customer_attributes_store >> silver_customer_attributes_store >> gold_feature_store_join # type: ignore
    dep_check_source_customer_financial_data >> bronze_customer_financials_store >> silver_customer_financials_store >> gold_feature_store_join # type: ignore
    gold_feature_store_join >> gold_feature_store_engineered >> feature_store_completed # type: ignore