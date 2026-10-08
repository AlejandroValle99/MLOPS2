"""
DAG: stroke_full_pipeline
Orquestador del pipeline de selección de modelo, en dos etapas:
1. stroke_data_cleaning (carga datos crudos y los particiona 60/20/20 a MinIO)
2. etl_train_models_process_taskflow (compara baseline, KNN, DT, XGBoost, RF y
   promueve al ganador por F2 como stroke-model@champion)

La comparación deja directamente el modelo champion que sirve la API: el preprocessing es
único (vive dentro del Pipeline de cada modelo) y el champion lo elige la comparación.

Usa TriggerDagRunOperator y espera su finalización con ExternalTaskSensor antes de
seguir con el siguiente.

Flujo:
trigger_cleaning > wait_cleaning > trigger_comparison > wait_comparison
> pipeline_completed  (EmptyOperator)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.state import DagRunState

# dags existentes a ejecutar
DAG_CLEANING = "stroke_data_cleaning"
DAG_COMPARISON = "etl_train_models_process_taskflow"


# dag
default_args = {
    "owner": "mlops-fiuba",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Tiempo espera por etapa
SENSOR_TIMEOUT_CLEANING = int(timedelta(hours=2).total_seconds())  # limpieza suele ser rápida
SENSOR_TIMEOUT_COMPARISON = int(timedelta(hours=4).total_seconds())  # Optuna puede tardar

POKE_INTERVAL = 60  # segundos entre comprobaciones del sensor


with DAG(
    dag_id="0_stroke_full_pipeline",
    description=("Orquestador de dags: limpieza, comparación de modelos, entrenamiento"),
    default_args=default_args,
    schedule=None,  # manual
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["stroke", "mlops", "orchestrator"],
    doc_md=__doc__,
) as dag:
    # inicio
    start = EmptyOperator(task_id="pipeline_start")

    # ETAPA 1. Limpieza y preprocesamiento
    trigger_cleaning = TriggerDagRunOperator(
        task_id="trigger_cleaning",
        trigger_dag_id=DAG_CLEANING,
        wait_for_completion=False,  # el sensor de abajo se encarga de la espera
        reset_dag_run=True,  # fuerza un DAG run nuevo aunque ya exista uno para la fecha
        poke_interval=POKE_INTERVAL,
        # Propagar la ejecucion lógica para que el sensor la encuentre
        execution_date="{{ logical_date }}",
        allowed_states=["success"],
        failed_states=["failed"],
    )

    wait_cleaning = ExternalTaskSensor(
        task_id="wait_cleaning",
        external_dag_id=DAG_CLEANING,
        external_task_id=None,  # None. espera a que el DAG entero termine
        execution_date_fn=lambda dt: dt,  # misma fecha que disparó el trigger
        allowed_states=[DagRunState.SUCCESS],
        failed_states=[DagRunState.FAILED],
        mode="poke",
        poke_interval=POKE_INTERVAL,
        timeout=SENSOR_TIMEOUT_CLEANING,
        soft_fail=False,
    )

    # ETAPA 2.Comparación de modelos
    trigger_comparison = TriggerDagRunOperator(
        task_id="trigger_comparison",
        trigger_dag_id=DAG_COMPARISON,
        wait_for_completion=False,
        reset_dag_run=True,
        poke_interval=POKE_INTERVAL,
        execution_date="{{ logical_date }}",
        allowed_states=["success"],
        failed_states=["failed"],
    )

    wait_comparison = ExternalTaskSensor(
        task_id="wait_comparison",
        external_dag_id=DAG_COMPARISON,
        external_task_id=None,
        execution_date_fn=lambda dt: dt,
        allowed_states=[DagRunState.SUCCESS],
        failed_states=[DagRunState.FAILED],
        mode="poke",
        poke_interval=POKE_INTERVAL,
        timeout=SENSOR_TIMEOUT_COMPARISON,
        soft_fail=False,
    )

    # fin — la comparación ya dejó el modelo champion que sirve la API
    end = EmptyOperator(task_id="pipeline_completed")

    # ejecución
    (start >> trigger_cleaning >> wait_cleaning >> trigger_comparison >> wait_comparison >> end)
