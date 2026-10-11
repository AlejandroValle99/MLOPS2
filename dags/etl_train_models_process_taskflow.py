# lgv
import datetime
import logging
import os

import mlflow
import pandas as pd
from airflow.decorators import dag, task
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from utils.model_utils import BETA, METRIC_NAME, cv_f2
from utils.s3_utils import get_s3_client, s3_file_exists, s3_to_dataframe

from model.preprocess import build_feature_pipeline
from stroke.data import data_version_for

RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH", "/opt/airflow/data/raw/healthcare-dataset-stroke-data.csv"
)

# Clases custom del preprocessing/baseline que skops debe confiar al (de)serializar
PREPROCESS_TRUSTED_TYPES = [
    "model.preprocess.StrokeCleaner",
    "model.preprocess.BMIGroupImputer",
]
# Nombre del modelo registrado que sirve la API (models:/stroke-model@champion)
SERVING_MODEL_NAME = "stroke-model"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DIR_DATA_PROCESSED = os.getenv("DIR_DATA_PROCESSED", "processed/final")
BUCKET = "mlflow-artifacts"

default_args = {
    "owner": "mlops-fiuba",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}


@dag(
    dag_id="etl_train_models_process_taskflow",
    description="ETL para preparar datos de entrenamiento usando TaskFlow API",
    default_args=default_args,
    catchup=False,
    schedule=None,
    start_date=datetime.datetime(2025, 1, 1),
    tags=["ETL", "TaskFlow"],
)
def process_etl_taskflow():
    def connect_mlflow():

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logging.info("Conectado a MLflow Tracking Server")
        return mlflow

    def create_experiment(experiment_name, model):
        """Create a new MLFlow experiment with a specified name.
        Save artifacts to the specified S3 bucket."""
        mlflow = connect_mlflow()

        if not mlflow.get_experiment_by_name(experiment_name):
            logging.info(f"Creando experimento MLflow: {experiment_name}")
            mlflow.create_experiment(
                name=experiment_name,
                tags={
                    "project": "healthcare-stroke-prediction",
                    "model": model,
                    "team": "mlops1-fiuba",
                },
            )

        experiment = mlflow.get_experiment_by_name(experiment_name)

        return experiment.experiment_id

    @task.python(task_id="check_data_to_process", multiple_outputs=True)
    def check_data_to_process():
        """
        Verificamos si el dataset ya fue procesado, si no, lo descargamos
        y preparamos para el siguiente nodo.
        """
        s3 = get_s3_client()

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_train.csv"):
            logging.warning("Dataset no encontrado localmente: X_train. Verificando en S3/MinIO...")
            raise FileNotFoundError("Dataset no encontrado: X_train.csv")

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_train.csv"):
            logging.warning("Dataset no encontrado localmente: y_train. Verificando en S3/MinIO...")
            raise FileNotFoundError("Dataset no encontrado: y_train.csv")

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_test.csv"):
            logging.warning("Dataset no encontrado localmente: X_test. Verificando en S3/MinIO...")
            raise FileNotFoundError("Dataset no encontrado: X_test.csv")

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_test.csv"):
            logging.warning("Dataset no encontrado localmente: y_test. Verificando en S3/MinIO...")
            raise FileNotFoundError("Dataset no encontrado: y_test.csv")

        logging.info("✅ Dataset encontrado localmente. Continuando con el proceso.")

        X_train = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_train.csv")
        y_train = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_train.csv")
        X_val = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_val.csv")
        y_val = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_val.csv")

        if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
            logging.error("❌ Uno o más datasets están vacíos. Verifique los archivos en S3/MinIO.")
            raise ValueError("Dataset vacío encontrado")

        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            logging.error("❌ El número de muestras no coincide.")
            raise ValueError("Desbalance entre X e y")

        if len(X_val) < 300 or len(y_val) < 300:
            logging.error(
                "❌ El número de muestras en X_val o y_val es menor a 300. "
                "Verifique los archivos en S3/MinIO."
            )
            raise ValueError("Dataset de validación demasiado pequeño")

        X_train_path = "./X_train.csv"
        y_train_path = "./y_train.csv"
        X_val_path = "./X_val.csv"
        y_val_path = "./y_val.csv"

        X_train.to_csv(X_train_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        X_val.to_csv(X_val_path, index=False)
        y_val.to_csv(y_val_path, index=False)

        logging.info("✅ Dataset descargado y preparado localmente. Continuando con el proceso.")
        return {
            "X_train_path": X_train_path,
            "y_train_path": y_train_path,
            "X_val_path": X_val_path,
            "y_val_path": y_val_path,
        }

    @task.python(task_id="create_base_model", multiple_outputs=True)
    def create_base_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo base simple (baseline) para comparar con el modelo
        entrenado posteriormente. Clasificador que solo usa la edad como predictor.
        Se envuelve en un Pipeline con el preprocessing unificado para que el
        artefacto registrado reciba datos crudos (igual que los demás modelos).
        """
        from utils.model_utils import AgeBaselineClassifier
        from utils.plots import plot_confusion_matrix, plot_precision_recall, plot_roc_curve

        logging.info("Iniciando tarea modelo base (baseline) usando solo la edad como predictor...")

        experiment_name = "baseline_model_evaluation"
        registered_model_name = "baseline_model"

        experiment_id = create_experiment(experiment_name, model="baseline")

        logging.info("Creando modelo base...")
        X_train = pd.read_csv(X_train_path)
        y_train_arr = pd.read_csv(y_train_path).to_numpy().ravel()
        baseline_model = Pipeline(
            [
                ("preprocess", build_feature_pipeline()),
                ("model", AgeBaselineClassifier(threshold=0.3)),
            ]
        ).fit(X_train, y_train_arr)

        logging.info("Evaluando modelo base con datos de validación...")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()
        y_pred_baseline_val = baseline_model.predict(X_val)
        y_prob_baseline_val = baseline_model.predict_proba(X_val)

        bl_metrics = {
            "AUC-ROC": roc_auc_score(y_val_arr, y_prob_baseline_val[:, 1]),
            "PR-AUC": average_precision_score(y_val_arr, y_prob_baseline_val[:, 1]),
            "F1": f1_score(y_val_arr, y_pred_baseline_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_baseline_val, beta=BETA),
            "Recall": recall_score(y_val_arr, y_pred_baseline_val),
            "Precision": precision_score(y_val_arr, y_pred_baseline_val, zero_division=0),
        }

        logging.info(f"Registrando en MLflow: {experiment_name} (ID: {experiment_id})")
        run_name_base = "base_model_exp"

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_baseline_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_baseline_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_baseline_val[:, 1], save_path=None)

        baseline_params = baseline_model.named_steps["model"].get_params()

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name_base,
            tags={"model": "baseline", "type": "evaluation"},
        ):
            logging.info("✅ Log de modelos del modelo base en MLflow")
            model_info = mlflow.sklearn.log_model(
                sk_model=baseline_model,
                name="baseline",
                registered_model_name=registered_model_name,
                skops_trusted_types=[
                    *PREPROCESS_TRUSTED_TYPES,
                    "utils.model_utils.AgeBaselineClassifier",
                ],
            )

            logging.info("✅ Parámetros del modelo base:")
            mlflow.log_params(baseline_params)

            logging.info("✅ Métricas del modelo base registradas en MLflow")
            mlflow.log_metrics(bl_metrics)

            logging.info("✅ Plots de evaluación del modelo base generados.")
            mlflow.log_figure(matrix_plot, artifact_file="matrix_plot.png")
            mlflow.log_figure(roc_plots[0], artifact_file="roc_curve_1_plot.png")
            mlflow.log_figure(roc_plots[1], artifact_file="roc_curve_2_plot.png")
            mlflow.log_figure(pr_plots, artifact_file="precision_recall_plot.png")

        logging.info("✅ Tarea modelo base finalizada.")
        return {
            "metrics": bl_metrics,
            "params": baseline_params,
            "experiment_id": experiment_id,
            "run_name": run_name_base,
            "registered_model_name": registered_model_name,
            "model_id": model_info.model_id,
        }

    @task.python(task_id="create_knn_model", multiple_outputs=True)
    def create_knn_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo KNN para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        import optuna
        from sklearn.neighbors import KNeighborsClassifier
        from utils.plots import plot_confusion_matrix, plot_precision_recall, plot_roc_curve

        logging.info("Iniciando tarea modelo KNN...")

        experiment_id = create_experiment("knn_model_evaluation", model="knn")

        logging.info(f"Usando experimento MLflow: knn_model_evaluation (ID: {experiment_id})")
        run_name_base = "knn_model_exp"
        registered_model_name = "knn_model"

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        def knn_objective(trial):
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_categorical("p", [1, 2]),
            }
            model = Pipeline(
                [
                    ("preprocess", build_feature_pipeline()),
                    ("model", KNeighborsClassifier(n_jobs=-1, **params)),
                ]
            )
            return cv_f2(model, X_train, y_train_arr)

        knn_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        knn_study.optimize(knn_objective, n_trials=50, show_progress_bar=True)

        knn_best_params = knn_study.best_params
        knn_best = Pipeline(
            [
                ("preprocess", build_feature_pipeline()),
                ("model", KNeighborsClassifier(n_jobs=-1, **knn_best_params)),
            ]
        ).fit(X_train, y_train_arr)

        logging.info("✅ Modelo KNN creado.")
        logging.info("✅ Parámetros del modelo KNN:")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_knn_val = knn_best.predict_proba(X_val)
        y_pred_knn_val = knn_best.predict(X_val)

        bl_metrics = {
            "AUC-ROC": roc_auc_score(y_val_arr, y_prob_knn_val[:, 1]),
            "PR-AUC": average_precision_score(y_val_arr, y_prob_knn_val[:, 1]),
            "F1": f1_score(y_val_arr, y_pred_knn_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_knn_val, beta=BETA),
            "Recall": recall_score(y_val_arr, y_pred_knn_val),
            "Precision": precision_score(y_val_arr, y_pred_knn_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_knn_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_knn_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_knn_val[:, 1], save_path=None)

        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name_base,
            tags={"model": "knn", "type": "evaluation"},
        ):
            logging.info("✅ Log de modelo KNN en MLflow")
            mlflow.log_params(knn_best_params)
            model_info = mlflow.sklearn.log_model(
                sk_model=knn_best,
                name="knn",
                registered_model_name=registered_model_name,
                skops_trusted_types=[
                    *PREPROCESS_TRUSTED_TYPES,
                    "sklearn.metrics._dist_metrics.ManhattanDistance64",
                    "sklearn.metrics._dist_metrics.EuclideanDistance64",
                    "sklearn.neighbors._kd_tree.KDTree",
                    "sklearn.neighbors._ball_tree.BallTree",
                ],
            )

            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelos y métricas KNN registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="knn_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="knn_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="knn_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="knn_precision_recall_curve.png")

        logging.info("✅ Tarea modelo KNN finalizada.")
        return {
            "metrics": bl_metrics,
            "params": knn_best_params,
            "experiment_id": experiment_id,
            "run_name": run_name_base,
            "registered_model_name": registered_model_name,
            "model_id": model_info.model_id,
        }

    @task.python(task_id="create_decision_tree_model", multiple_outputs=True)
    def create_decision_tree_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo de árbol de decisión para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        import optuna
        from sklearn.tree import DecisionTreeClassifier
        from utils.plots import plot_confusion_matrix, plot_precision_recall, plot_roc_curve

        logging.info("Iniciando tarea modelo de árbol de decisión...")

        experiment_id = create_experiment("decision_tree_model_evaluation", model="decision_tree")

        logging.info(
            f"Usando experimento MLflow: decision_tree_model_evaluation (ID: {experiment_id})"
        )
        run_name_base = "decision_tree_model_exp"
        registered_model_name = "decision_tree_model"

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        def dt_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            }
            model = Pipeline(
                [
                    ("preprocess", build_feature_pipeline()),
                    (
                        "model",
                        DecisionTreeClassifier(class_weight="balanced", random_state=42, **params),
                    ),
                ]
            )
            return cv_f2(model, X_train, y_train_arr)

        dt_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        dt_study.optimize(dt_objective, n_trials=50, show_progress_bar=True)

        dt_best_params = dt_study.best_params
        dt_best = Pipeline(
            [
                ("preprocess", build_feature_pipeline()),
                (
                    "model",
                    DecisionTreeClassifier(
                        class_weight="balanced", random_state=42, **dt_best_params
                    ),
                ),
            ]
        ).fit(X_train, y_train_arr)

        logging.info("✅ Modelo de árbol de decisión creado.")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_dt_val = dt_best.predict_proba(X_val)
        y_pred_dt_val = dt_best.predict(X_val)
        logging.info("y_val_arr: " + str(y_val_arr[:5]))
        logging.info("y_prob_dt_val: " + str(y_prob_dt_val[:5]))
        logging.info("y_pred_dt_val: " + str(y_pred_dt_val[:5]))

        bl_metrics = {
            "AUC-ROC": roc_auc_score(y_val_arr, y_prob_dt_val[:, 1]),
            "PR-AUC": average_precision_score(y_val_arr, y_prob_dt_val[:, 1]),
            "F1": f1_score(y_val_arr, y_pred_dt_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_dt_val, beta=BETA),
            "Recall": recall_score(y_val_arr, y_pred_dt_val),
            "Precision": precision_score(y_val_arr, y_pred_dt_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_dt_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_dt_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_dt_val[:, 1], save_path=None)

        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name_base,
            tags={"model": "decision_tree", "type": "evaluation"},
        ):
            logging.info("✅ Log de modelo de árbol de decisión en MLflow")
            mlflow.log_params(dt_best_params)
            model_info = mlflow.sklearn.log_model(
                sk_model=dt_best,
                name="decision_tree",
                registered_model_name=registered_model_name,
                skops_trusted_types=[
                    *PREPROCESS_TRUSTED_TYPES,
                    "sklearn.tree._classes.DecisionTreeClassifier",
                    "sklearn.tree._tree.Tree",
                ],
            )
            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelo de árbol de decisión y métricas registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="dt_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="dt_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="dt_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="dt_precision_recall_curve.png")

        logging.info("✅ Tarea modelo de árbol de decisión finalizada.")
        return {
            "metrics": bl_metrics,
            "params": dt_best_params,
            "experiment_id": experiment_id,
            "run_name": run_name_base,
            "registered_model_name": registered_model_name,
            "model_id": model_info.model_id,
        }

    @task.python(task_id="create_xgboost_model", multiple_outputs=True)
    def create_xgboost_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo de XGBoost para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        import numpy as np
        import optuna
        from utils.plots import plot_confusion_matrix, plot_precision_recall, plot_roc_curve
        from xgboost import XGBClassifier

        logging.info("Iniciando tarea modelo de XGBoost...")

        experiment_id = create_experiment("xgboost_model_evaluation", model="xgboost")

        logging.info(f"Usando experimento MLflow: xgboost_model_evaluation (ID: {experiment_id})")
        run_name_base = "xgboost_model_exp"
        registered_model_name = "xgboost_model"

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        neg, pos = np.bincount(y_train_arr)
        scale_pos = neg / pos

        def xgb_objective(trial):
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = Pipeline(
                [
                    ("preprocess", build_feature_pipeline()),
                    (
                        "model",
                        XGBClassifier(
                            scale_pos_weight=scale_pos,
                            eval_metric="aucpr",
                            random_state=42,
                            use_label_encoder=False,
                            n_jobs=1,
                            **params,
                        ),
                    ),
                ]
            )
            return cv_f2(model, X_train, y_train_arr)

        xgb_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

        xgb_best_params = xgb_study.best_params
        xgb_best = Pipeline(
            [
                ("preprocess", build_feature_pipeline()),
                (
                    "model",
                    XGBClassifier(
                        scale_pos_weight=scale_pos,
                        eval_metric="aucpr",
                        random_state=42,
                        use_label_encoder=False,
                        n_jobs=1,
                        **xgb_best_params,
                    ),
                ),
            ]
        ).fit(X_train, y_train_arr)

        logging.info("✅ Modelo de XGBoost creado.")
        logging.info("✅ Parámetros del modelo de XGBoost:")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_xgb_val = xgb_best.predict_proba(X_val)
        y_pred_xgb_val = xgb_best.predict(X_val)

        bl_metrics = {
            "AUC-ROC": roc_auc_score(y_val_arr, y_prob_xgb_val[:, 1]),
            "PR-AUC": average_precision_score(y_val_arr, y_prob_xgb_val[:, 1]),
            "F1": f1_score(y_val_arr, y_pred_xgb_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_xgb_val, beta=BETA),
            "Recall": recall_score(y_val_arr, y_pred_xgb_val),
            "Precision": precision_score(y_val_arr, y_pred_xgb_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_xgb_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_xgb_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_xgb_val[:, 1], save_path=None)

        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name_base,
            tags={"model": "xgboost", "type": "evaluation"},
        ):
            logging.info("✅ Log de modelo XGBoost en MLflow")
            mlflow.log_params(xgb_best_params)
            # Se loguea el Pipeline completo con flavor sklearn para que la API
            # pueda cargarlo con mlflow.sklearn.load_model si resulta campeón.
            model_info = mlflow.sklearn.log_model(
                sk_model=xgb_best,
                name="xgboost",
                registered_model_name=registered_model_name,
                skops_trusted_types=[
                    *PREPROCESS_TRUSTED_TYPES,
                    "xgboost.sklearn.XGBClassifier",
                    "xgboost.core.Booster",
                ],
            )
            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelo XGBoost y métricas registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="xgb_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="xgb_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="xgb_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="xgb_precision_recall_curve.png")

        logging.info("✅ Tarea modelo XGBoost finalizada.")
        return {
            "metrics": bl_metrics,
            "params": xgb_best_params,
            "experiment_id": experiment_id,
            "run_name": run_name_base,
            "registered_model_name": registered_model_name,
            "model_id": model_info.model_id,
        }

    @task.python(task_id="create_random_forest_model", multiple_outputs=True)
    def create_random_forest_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo de bosque aleatorio para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        import optuna
        from sklearn.ensemble import RandomForestClassifier
        from utils.plots import plot_confusion_matrix, plot_precision_recall, plot_roc_curve

        logging.info("Iniciando tarea modelo de random forest...")

        experiment_id = create_experiment("random_forest_model_evaluation", model="random_forest")

        logging.info(
            f"Usando experimento MLflow: random_forest_model_evaluation (ID: {experiment_id})"
        )
        run_name_base = "random_forest_model_exp"
        registered_model_name = "random_forest_model"

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        def rf_objective(trial):
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, None]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            }
            model = Pipeline(
                [
                    ("preprocess", build_feature_pipeline()),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced", random_state=42, n_jobs=-1, **params
                        ),
                    ),
                ]
            )
            return cv_f2(model, X_train, y_train_arr)

        rf_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        rf_study.optimize(rf_objective, n_trials=50, show_progress_bar=True)

        rf_best_params = rf_study.best_params
        rf_best = Pipeline(
            [
                ("preprocess", build_feature_pipeline()),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced", random_state=42, n_jobs=-1, **rf_best_params
                    ),
                ),
            ]
        ).fit(X_train, y_train_arr)

        logging.info("✅ Modelo de bosque aleatorio creado.")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_rf_val = rf_best.predict_proba(X_val)
        y_pred_rf_val = rf_best.predict(X_val)

        bl_metrics = {
            "AUC-ROC": roc_auc_score(y_val_arr, y_prob_rf_val[:, 1]),
            "PR-AUC": average_precision_score(y_val_arr, y_prob_rf_val[:, 1]),
            "F1": f1_score(y_val_arr, y_pred_rf_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_rf_val, beta=BETA),
            "Recall": recall_score(y_val_arr, y_pred_rf_val),
            "Precision": precision_score(y_val_arr, y_pred_rf_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_rf_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_rf_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_rf_val[:, 1], save_path=None)

        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name_base,
            tags={"model": "random_forest", "type": "evaluation"},
        ):
            logging.info("✅ Log de modelo de random forest en MLflow")
            mlflow.log_params(rf_best_params)
            model_info = mlflow.sklearn.log_model(
                sk_model=rf_best,
                name="random_forest",
                registered_model_name=registered_model_name,
                skops_trusted_types=[
                    *PREPROCESS_TRUSTED_TYPES,
                    "sklearn.ensemble._forest.RandomForestClassifier",
                    "sklearn.tree._classes.DecisionTreeClassifier",
                    "sklearn.tree._tree.Tree",
                ],
            )
            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelo de random forest y métricas registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="rf_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="rf_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="rf_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="rf_precision_recall_curve.png")

        logging.info("✅ Tarea modelo de random forest finalizada.")
        return {
            "metrics": bl_metrics,
            "params": rf_best_params,
            "experiment_id": experiment_id,
            "run_name": run_name_base,
            "registered_model_name": registered_model_name,
            "model_id": model_info.model_id,
        }

    @task.python(task_id="set_champion_model", multiple_outputs=True)
    def set_champion_model(metrics_base, metrics_knn, metrics_dt, metrics_xgb, metrics_rf):
        """
        Cierra el ciclo: compara los cinco modelos por F2 y promueve al ganador
        como `champion` del modelo que sirve la API (stroke-model).

        Cada modelo se entrenó como Pipeline completo (preprocessing + estimador),
        así que el artefacto ganador es autónomo: la API lo carga con
        mlflow.sklearn.load_model y recibe datos crudos sin cambios.
        """
        mlflow = connect_mlflow()

        logging.info("Iniciando tarea de selección del modelo campeón...")

        models = [
            ("baseline", metrics_base),
            ("knn", metrics_knn),
            ("decision_tree", metrics_dt),
            ("xgboost", metrics_xgb),
            ("random_forest", metrics_rf),
        ]

        logging.info(f"Comparando modelos según la métrica principal ({METRIC_NAME})...")
        for name, m in models:
            logging.info(f"  {name}: {METRIC_NAME}={m['metrics'][METRIC_NAME]:.4f}")

        champion_name, champion = max(models, key=lambda x: x[1]["metrics"][METRIC_NAME])
        champion_f2 = champion["metrics"][METRIC_NAME]
        winner_uri = f"models:/{champion['model_id']}"

        logging.info(
            f"✅ Modelo campeón: {champion_name} ({METRIC_NAME}={champion_f2:.4f}) → {winner_uri}"
        )

        # Registrar el Pipeline ganador como nueva versión del modelo que sirve la API
        client = mlflow.MlflowClient()
        mv = mlflow.register_model(winner_uri, SERVING_MODEL_NAME)
        client.set_registered_model_alias(SERVING_MODEL_NAME, "champion", mv.version)

        # D5 (docs/PLAN.md): tag model_family and data_version on the version that
        # actually serves the API. `threshold` is deliberately not tagged here — it
        # depends on Phase 5's calibration + out-of-fold threshold selection, which
        # this comparison DAG doesn't do yet (it still picks a winner at the default
        # 0.5 cutoff, baked into METRIC_NAME's F2 computation upstream).
        client.set_model_version_tag(SERVING_MODEL_NAME, mv.version, "model_family", champion_name)
        version = data_version_for(RAW_DATA_PATH)
        if version is not None:
            client.set_model_version_tag(SERVING_MODEL_NAME, mv.version, "data_version", version)

        logging.info(
            f"'{SERVING_MODEL_NAME}' v{mv.version} promovido a alias 'champion' "
            f"(algoritmo ganador: {champion_name})"
        )
        logging.info("✅ Tarea de selección del modelo campeón finalizada.")

        return {
            "champion_algorithm": champion_name,
            "champion_f2": champion_f2,
            "serving_model": SERVING_MODEL_NAME,
            "version": mv.version,
        }

    # 🧩 Encadenamiento
    paths = check_data_to_process()
    metrics_base = create_base_model(
        paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"]
    )
    metrics_knn = create_knn_model(
        paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"]
    )
    metrics_dt = create_decision_tree_model(
        paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"]
    )
    metrics_xgb = create_xgboost_model(
        paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"]
    )
    metrics_rf = create_random_forest_model(
        paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"]
    )
    set_champion_model(metrics_base, metrics_knn, metrics_dt, metrics_xgb, metrics_rf)


dag = process_etl_taskflow()
