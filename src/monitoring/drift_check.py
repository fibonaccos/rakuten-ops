"""
Pour chaque lot simulé (data/stream/batch_XX.csv, voir simulate_stream.py),
compare la distribution des données à la référence d'entraînement
(data/train/raw.csv) avec Evidently, et évalue la performance du modèle en
production sur un échantillon du lot (via l'API réelle). Logge tout dans
MLflow, un run par lot, dans la même série de steps pour suivre l'évolution
dans le temps.

Usage (depuis la racine du repo) :
    export MONITORING_API_USERNAME=...
    export MONITORING_API_PASSWORD=...
    python3 src/monitoring/drift_check.py

Nécessite l'API (localhost:8000) et MLflow (localhost:5001) démarrés.
Si aucun modèle n'est en production, le drift est quand même calculé -- seule
la partie performance est marquée comme indisponible pour ce lot.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import requests
from evidently import Dataset, DataDefinition, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset

REFERENCE_PATH = Path("data/train/raw.csv")
STREAM_DIR = Path("data/stream")
REPORTS_DIR = Path("data/monitoring_reports")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# How many rows per batch to actually send to the model for performance
# evaluation. Drift itself is computed on the full batch (no API calls
# needed); only the performance check calls the API, so we sample to keep
# runtime reasonable.
PERFORMANCE_SAMPLE_SIZE = 200

TEXT_COLUMNS = ["designation", "description"]
CATEGORICAL_COLUMNS = ["prdtypecode"]


def load_reference() -> pd.DataFrame:
    """Load the real training data used as the drift reference."""
    df = pd.read_csv(REFERENCE_PATH, sep=",", dtype={"prdtypecode": str})
    df = df.drop(columns=["productid", "imageid"])
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("")
    return df


def load_batches() -> list[Path]:
    """List simulated batch files in order (batch_01.csv, batch_02.csv, ...)."""
    return sorted(STREAM_DIR.glob("batch_*.csv"))


def compute_drift(reference: pd.DataFrame, batch: pd.DataFrame) -> tuple[dict[str, float], Snapshot]:
    """Run Evidently's DataDriftPreset and return a flat dict of scalar metrics."""
    definition = DataDefinition(
        categorical_columns=CATEGORICAL_COLUMNS, text_columns=TEXT_COLUMNS
    )
    reference_ds = Dataset.from_pandas(reference, data_definition=definition)
    batch_ds = Dataset.from_pandas(batch, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=batch_ds, reference_data=reference_ds)

    metrics: dict[str, float] = {}
    for entry in snapshot.dict()["metrics"]:
        name = entry["metric_name"]
        value = entry["value"]
        if name.startswith("DriftedColumnsCount"):
            metrics["drift_share"] = float(value["share"])
        elif name.startswith("ValueDrift(column=designation"):
            metrics["drift_designation"] = float(value)
        elif name.startswith("ValueDrift(column=description"):
            metrics["drift_description"] = float(value)
        elif name.startswith("ValueDrift(column=prdtypecode"):
            metrics["drift_prdtypecode"] = float(value)

    return metrics, snapshot


def get_api_token() -> str | None:
    """Log in to the API and return a bearer token, or None if it fails."""
    username = os.environ.get("MONITORING_API_USERNAME")
    password = os.environ.get("MONITORING_API_PASSWORD")
    if not username or not password:
        print("MONITORING_API_USERNAME / MONITORING_API_PASSWORD non définis -- "
              "performance non évaluée.")
        return None

    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Impossible de se connecter à l'API ({e}) -- performance non évaluée.")
        return None


def compute_accuracy(batch: pd.DataFrame, token: str) -> float | None:
    """Sample the batch, call /predict/batch, and return accuracy vs true labels.

    Returns None if the API/model isn't available (e.g. no model in production).
    """
    sample = batch.sample(min(PERFORMANCE_SAMPLE_SIZE, len(batch)), random_state=42)
    payload = [
        {"designation": row.designation, "description": row.description or None}
        for row in sample.itertuples()
    ]

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  /predict/batch a échoué ({e}) -- pas de modèle en production ?")
        return None

    predictions = [item["category"] for item in response.json()["output"]]
    true_labels = sample["prdtypecode"].tolist()
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    return correct / len(true_labels)


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("data-drift-monitoring")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    batches = load_batches()
    if not batches:
        print(f"Aucun lot trouvé dans {STREAM_DIR}/ -- lance simulate_stream.py d'abord.")
        return

    token = get_api_token()
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(reference),
        "batches": [],
    }

    with mlflow.start_run(run_name="drift-monitoring-run"):
        mlflow.log_param("n_batches", len(batches))
        mlflow.log_param("reference_rows", len(reference))
        mlflow.log_param("performance_sample_size", PERFORMANCE_SAMPLE_SIZE)

        for step, batch_path in enumerate(batches, start=1):
            print(f"--- Lot {step}/{len(batches)} : {batch_path.name} ---")
            batch = pd.read_csv(batch_path, sep=",", dtype={"prdtypecode": str})
            for col in TEXT_COLUMNS:
                batch[col] = batch[col].fillna("")

            drift_metrics, snapshot = compute_drift(reference, batch)
            report_path = REPORTS_DIR / f"drift_{batch_path.stem}.html"
            snapshot.save_html(str(report_path))

            metrics = dict(drift_metrics)
            if token:
                accuracy = compute_accuracy(batch, token)
                if accuracy is not None:
                    metrics["accuracy"] = accuracy

            print(f"  {metrics}")
            mlflow.log_metrics(metrics, step=step)
            mlflow.log_artifact(str(report_path), artifact_path="drift_reports")

            summary["batches"].append({
                "batch": batch_path.name,
                "n_rows": len(batch),
                "metrics": metrics,
                "report": str(report_path),
            })

    summary_path = REPORTS_DIR / "drift_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nTerminé. Résultats visibles sur {MLFLOW_TRACKING_URI}, "
          f"expérience 'data-drift-monitoring'. Résumé JSON : {summary_path}")


if __name__ == "__main__":
    main()
