"""
DAG: 0_stroke_full_pipeline
Orquestador del ciclo de entrenamiento: dispara etl_train_models_process_taskflow
(verifica el CSV crudo, entrena y calibra los candidatos sobre el split congelado,
elige un finalista por F2 out-of-fold y lo pasa por el gate estadístico, que solo lo
promueve a stroke-model@champion si supera al campeón actual) y espera a que termine.

Ya no dispara stroke_data_cleaning: el entrenamiento reconstruye el split congelado
directamente desde el CSV crudo verificado, y no lee los CSV de processed/final/.

Usa TriggerDagRunOperator y espera su finalización con ExternalTaskSensor.

Flujo:
pipeline_start > trigger_comparison > wait_comparison > pipeline_completed
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.state import DagRunState

# dag existente a ejecutar
DAG_COMPARISON = "etl_train_models_process_taskflow"


# dag
default_args = {
    "owner": "mlops-fiuba",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Tiempo espera por etapa
SENSOR_TIMEOUT_COMPARISON = int(timedelta(hours=4).total_seconds())  # Optuna puede tardar

POKE_INTERVAL = 60  # segundos entre comprobaciones del sensor


with DAG(
    dag_id="0_stroke_full_pipeline",
    description="Orquestador: entrenamiento, comparación de modelos y gate",
    default_args=default_args,
    schedule=None,  # manual
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["stroke", "mlops", "orchestrator"],
    doc_md=__doc__,
) as dag:
    # inicio
    start = EmptyOperator(task_id="pipeline_start")

    # Comparación de modelos y gate
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

    # fin — el gate ya decidió si el finalista pasa a ser el champion que sirve la API
    end = EmptyOperator(task_id="pipeline_completed")

    # ejecución
    start >> trigger_comparison >> wait_comparison >> end
