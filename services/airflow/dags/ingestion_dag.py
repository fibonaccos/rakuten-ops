"""
DAG d'ingestion : extrait les nouvelles lignes de la base applicative dès
qu'il y en a au moins BATCH_SIZE depuis la dernière extraction, les valide,
les fige dans un fichier parquet horodaté, et alimente le dataset
d'entraînement en cours de constitution (next_training_dataset.parquet).

Placeholders à remplacer :
  <APP_TABLE_NAME>    table Postgres source (ex: "predictions")
  <ID_COLUMN>          colonne clé primaire / auto-incrémentée, utilisée
                       comme watermark
  <REQUIRED_COLUMNS>   liste des colonnes obligatoires pour la validation
                       (remplacer la liste ci-dessous)
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from airflow.datasets import Dataset
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.providers.postgres.hooks.postgres import PostgresHook


BATCH_SIZE = int(os.environ.get("BATCH_SIZE", -1))
DATA_ROOT = os.environ.get("DATA_ROOT", "/opt/airflow/data")
BATCH_DIR = os.path.join(DATA_ROOT, "batches")
REFERENCE_DATASET_PATH = os.path.join(DATA_ROOT, "references", "raw.csv")
TRAIN_DATASET_PATH = os.path.join(DATA_ROOT, "train", "train.csv")
WATERMARK_TABLE = "ml_ingestion_watermark"
CRON_EXPR = os.environ.get("CRON_EXPR", "0 * * * *")

APP_TABLE_NAME = "inference"
ID_COLUMN = "inference_id"
REQUIRED_COLUMNS = [
    "inference_id",
    "designation",
    "description"
]

# Dataset Airflow : le DAG de drift se déclenche automatiquement quand
# un nouveau batch parquet est écrit (data-aware scheduling, pas de polling).
batch_dataset = Dataset(f"file://{BATCH_DIR}")

DAG_ID="ingestion_pipeline"


@dag(
    dag_id=DAG_ID,
    schedule=CRON_EXPR,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1},
    tags=["mlops", "ingestion"],
)
def ingestion_pipeline():
    @task
    def ensure_watermark_table():
        hook = PostgresHook(postgres_conn_id="app_postgres")
        hook.run(f"""
            CREATE TABLE IF NOT EXISTS {WATERMARK_TABLE} (
                id SERIAL PRIMARY KEY,
                last_extracted_id BIGINT NOT NULL DEFAULT 0,
                last_extracted_at TIMESTAMPTZ,
                batch_file TEXT
            );
            INSERT INTO {WATERMARK_TABLE} (last_extracted_id)
            SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM {WATERMARK_TABLE});
        """)

    @task
    def get_last_watermark() -> int:
        hook = PostgresHook(postgres_conn_id="app_postgres")
        row = hook.get_first(
            f"SELECT last_extracted_id FROM {WATERMARK_TABLE} ORDER BY id DESC LIMIT 1;"
        )
        return int(row[0]) if row else 0

    @task
    def extract_batch(last_id: int) -> dict:
        hook = PostgresHook(postgres_conn_id="app_postgres")
        count = hook.get_first(
            f"SELECT COUNT(*) FROM {APP_TABLE_NAME} WHERE {ID_COLUMN} > %s;",
            parameters=(last_id,),
        )[0]

        if count < BATCH_SIZE:
            raise AirflowSkipException(
                f"Dag '{DAG_ID}' ended early as only {count} new samples has been extracted ({BATCH_SIZE} required)."
            )
        df = hook.get_pandas_df(
            f"""
            SELECT * FROM {APP_TABLE_NAME}
            WHERE {ID_COLUMN} > %s
            ORDER BY {ID_COLUMN} ASC
            LIMIT %s;
            """,
            parameters=(last_id, BATCH_SIZE),
        )
        df = df[REQUIRED_COLUMNS]
        os.makedirs(BATCH_DIR, exist_ok=True)
        return {"n_rows": len(df), "_df_json": df.to_json(orient="split")}

    @task
    def validate_batch(batch: dict) -> dict:
        df = pd.read_json(batch["_df_json"], orient="split")
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns found : {missing_cols}")
        return batch

    @task(outlets=[batch_dataset])
    def write_parquet(batch: dict) -> str:
        df = pd.read_json(batch["_df_json"], orient="split")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_batch_{uuid.uuid4().hex[:8]}.parquet"
        path = os.path.join(BATCH_DIR, filename)
        df.to_parquet(path, index=False)
        return path

    @task
    def append_to_training_dataset(batch_path: str) -> str:
        new_df = pd.read_parquet(batch_path)
        new_df = new_df.drop(columns=["inference_id"])
        os.makedirs(os.path.dirname(TRAIN_DATASET_PATH), exist_ok=True)

        if os.path.exists(REFERENCE_DATASET_PATH):
            base_df = pd.read_csv(REFERENCE_DATASET_PATH)
            last_productid = base_df["productid"].max()
            last_imageid = base_df["imageid"].max()
            new_productid = list(range(last_productid + 1, last_productid + 1 + len(base_df)))
            new_imageid = list(range(last_imageid + 1, last_imageid + 1 + len(base_df)))
            new_df["productid"] = new_productid
            new_df["imageid"] = new_imageid
            new_df["prdtypecode"] = None
            combined = pd.concat([base_df, new_df]).drop_duplicates(subset=[ID_COLUMN])
        else:
            new_productid = list(range(1, len(new_df) + 1))
            new_imageid = list(range(1, len(new_df) + 1))
            new_df["productid"] = new_productid
            new_df["imageid"] = new_imageid
            new_df["prdtypecode"] = None
            combined = new_df

        combined.to_parquet(TRAIN_DATASET_PATH, index=False)
        return TRAIN_DATASET_PATH

    @task
    def update_watermark(batch_path: str):
        df = pd.read_csv(batch_path)
        new_max_id = int(df[ID_COLUMN].max())
        hook = PostgresHook(postgres_conn_id="app_postgres")
        hook.run(
            f"""
            INSERT INTO {WATERMARK_TABLE} (last_extracted_id, last_extracted_at, batch_file)
            VALUES (%s, now(), %s);
            """,
            parameters=(new_max_id, os.path.basename(batch_path)),
        )

    ensure = ensure_watermark_table()
    last_id = get_last_watermark()
    batch = extract_batch(last_id)
    validated = validate_batch(batch)
    parquet_path = write_parquet(validated)
    training_path = append_to_training_dataset(parquet_path)
    watermark = update_watermark(parquet_path)

    ensure >> last_id >> batch >> validated >> parquet_path >> training_path >> watermark


ingestion_pipeline()
