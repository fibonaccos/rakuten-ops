"""
DAG de détection de drift : exécute ton script Evidently existant sur le
batch le plus récent, dans un conteneur basé sur l'image d'entraînement
(ou une image dédiée si evidently n'y est pas déjà). Le script :
  - logue les métriques de drift comme une run MLflow (déjà fait chez toi) ;
  - pousse les métriques "métier" (part de colonnes drifted, etc.) vers le
    Prometheus Pushgateway, que Prometheus scrape ensuite normalement, et
    que Grafana lit via le datasource Prometheus existant.

Convention imposée au script scripts/drift_detection.py : la DERNIÈRE ligne
de son stdout doit être "DRIFT_SHARE=<float>" pour que ce DAG puisse décider
de déclencher ou non un réentraînement. À adapter si tu préfères un autre
mécanisme (fichier de sortie JSON, tag MLflow relu via l'API, etc.).

Placeholders à remplacer :
  <TRAINING_IMAGE_NAME>          image contenant le script evidently + ses deps
  <SHARED_DOCKER_NETWORK_NAME>
  <MLFLOW_CONTAINER_URL>
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

from airflow.datasets import Dataset
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

DATA_ROOT = os.environ.get("DATA_ROOT", "/opt/airflow/data")
BATCH_DIR = os.path.join(DATA_ROOT, "batches")
DRIFT_OUTPUT_DIR = os.path.join(DATA_ROOT, "drift_reports")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.3"))

batch_dataset = Dataset(f"file://{BATCH_DIR}")


@dag(
    dag_id="drift_detection_pipeline",
    schedule=[batch_dataset],  # se déclenche automatiquement à chaque nouveau batch
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1},
    tags=["mlops", "drift"],
)
def drift_detection_pipeline():

    @task
    def find_latest_batch() -> str:
        # Le nommage YYYYMMDD_HHMMSS_batch_<uuid>.parquet est trié
        # lexicographiquement = trié chronologiquement, donc glob + sort suffit.
        files = sorted(glob.glob(os.path.join(BATCH_DIR, "*.parquet")))
        if not files:
            raise FileNotFoundError("Aucun batch trouvé dans le répertoire.")
        return files[-1]

    latest_batch = find_latest_batch()

    run_drift = DockerOperator(
        task_id="run_drift_detection",
        image="<TRAINING_IMAGE_NAME>",
        command=[
            "python", "scripts/drift_detection.py",
            "--batch-path", "{{ ti.xcom_pull(task_ids='find_latest_batch') }}",
            "--output-dir", "/data/drift_reports",
        ],
        mounts=[Mount(source=DATA_ROOT, target="/data", type="bind")],
        environment={
            "MLFLOW_TRACKING_URI": "<MLFLOW_CONTAINER_URL>",
            "PROMETHEUS_PUSHGATEWAY_URL": os.environ.get(
                "PROMETHEUS_PUSHGATEWAY_URL", "http://pushgateway:9091"
            ),
        },
        network_mode="<SHARED_DOCKER_NETWORK_NAME>",
        auto_remove="success",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )

    @task.short_circuit
    def drift_gate(docker_logs: str) -> bool:
        share = None
        for line in docker_logs.splitlines():
            if line.startswith("DRIFT_SHARE="):
                share = float(line.split("=", 1)[1])
        if share is None:
            raise ValueError("Le script de drift n'a pas produit de ligne DRIFT_SHARE=...")
        return share >= DRIFT_THRESHOLD

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining_on_drift",
        trigger_dag_id="retraining_pipeline",
        conf={"triggered_by": "drift_detection_pipeline"},
    )

    latest_batch >> run_drift
    gate = drift_gate(run_drift.output)
    gate >> trigger_retraining


drift_detection_pipeline()
