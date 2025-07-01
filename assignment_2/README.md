# CS611 Machine Learning Engineering Assignment 2

In this assignment, a complete ML workflow has been built with Airflow. The project is found in the respective GitHub repo for the CS611 course under: [GitHub-Repo](https://github.com/nicholas-voigt/SMU_CS611_Machine_Learning_Engineering/tree/main)


## Usage

1. build and run the docker container. enables webserver access on 0.0.0.0:8080
2. log in to airflow with user: admin pw: admin
3. enable data pipeline. will do automatic catchup
4. once data pipeline is fully done. model training pipeline can be triggered manually. The start and end date for which data will be used is stored in the configs module. The model training pipeline is kept manually on purpose.
5. Enable model inference pipeline. Pipeline checks if data pipeline already ran for this month and also checks if a trained model is available. When the inference pipeline is activated without a trained model, the DAG will throw an error message that a model is missing. The inference pipeline logs relevant monitoring metrics in json-logfiles under /opt/airflow/datamart/model_prediction/logs/.


## Structure

- /dags: contains the dag files with the execution logic for airflow (data pipeline, ml training pipeline and inference pipeline)
- /data: contains source data files in csv format. mounted to webserver and scheduler
- /datamart: contains processed data from bronze to gold level, training data, predictions and logs
- /scripts: contains the scripts which are executed from the airflow dag.
    - /scripts/data_processing: main processing logic for the data pipeline
    - /scripts/ml_processing: main processing logic for the ml pipeline
- /configs: configurable variables for directories, standard-timeframes, etc.
- /utils: helper functions

