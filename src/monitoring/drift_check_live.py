"""
Version "trafic réel" du monitoring de drift : au lieu de lire des lots
fabriqués à l'avance (data/stream/batch_XX.csv), interroge directement la
table `inference` -- alimentée par le trafic Locust (ou par de vrais
utilisateurs) -- sur une fenêtre de temps glissante depuis la dernière
vérification, compare cette fenêtre à la référence d'entraînement, et logge
le résultat dans MLflow.

Peut être lancé à la main (vérification manuelle) ou planifié (cron, tâche
récurrente) pour une surveillance automatique.

Usage (depuis la racine du repo) :
    export DATABASE_URL=postgresql://user:pass@localhost:5432/rakuten
    python3 src/monitoring/drift_check_live.py

Sans état préalable (premier lancement), la fenêtre par défaut est les
dernières LOOKBACK_HOURS heures. Les lancements suivants ne reprennent que ce
qui est arrivé depuis la dernière vérification (data/monitoring_reports/
last_check.json), pour ne jamais compter deux fois la même ligne.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import pandas as pd
import psycopg2

from drift_check import MLFLOW_TRACKING_URI, TEXT_COLUMNS, compute_drift, load_reference

STATE_PATH = Path("data/monitoring_reports/last_check.json")
LOOKBACK_HOURS = int(os.environ.get("MONITORING_LOOKBACK_HOURS", "1"))
MIN_ROWS_FOR_DRIFT = 30  # en dessous, le test statistique n'est pas fiable


def get_since() -> datetime:
    """Horodatage du dernier contrôle, ou "il y a LOOKBACK_HOURS" au premier lancement."""
    if STATE_PATH.exists():
        with STATE_PATH.open(encoding="utf-8") as f:
            return datetime.fromisoformat(json.load(f)["last_check"])
    return datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)


def save_last_check(timestamp: datetime) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump({"last_check": timestamp.isoformat()}, f)


def load_current_batch_from_db(since: datetime) -> pd.DataFrame:
    """Charge les prédictions réelles loggées depuis `since` (table `inference`).

    predicted_category est renommé en prdtypecode pour rester cohérent avec
    le reste du pipeline (data/train/raw.csv, simulate_stream.py, ...).
    """
    query = """
        SELECT designation, description, predicted_category AS prdtypecode,
               labeled_category, queried_at
        FROM inference
        WHERE queried_at >= %s
        ORDER BY queried_at
    """
    conn = psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=5)
    try:
        df = pd.read_sql(query, conn, params=(since,))
    finally:
        conn.close()

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("")
    return df


def compute_live_accuracy(batch: pd.DataFrame) -> tuple[float | None, float]:
    """Accuracy calculée directement depuis les lignes déjà labellisées en base
    (labeled_category), sans rappeler l'API -- et le taux de couverture (quelle
    part du lot a effectivement une vraie catégorie confirmée).
    """
    labeled = batch[batch["labeled_category"].notna()]
    coverage = len(labeled) / len(batch) if len(batch) else 0.0
    if labeled.empty:
        return None, coverage
    accuracy = (labeled["prdtypecode"] == labeled["labeled_category"]).mean()
    return float(accuracy), coverage


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("data-drift-monitoring-live")

    since = get_since()
    now = datetime.now(timezone.utc)
    print(f"Fenêtre observée : {since.isoformat()} -> {now.isoformat()}")

    batch = load_current_batch_from_db(since)
    print(f"{len(batch)} prédictions trouvées sur cette fenêtre.")

    if len(batch) < MIN_ROWS_FOR_DRIFT:
        print(
            f"Moins de {MIN_ROWS_FOR_DRIFT} lignes -- pas assez pour un test de "
            "drift fiable, on ne logge rien cette fois."
        )
        save_last_check(now)
        return

    reference = load_reference()
    drift_metrics, snapshot = compute_drift(reference, batch[["designation", "description", "prdtypecode"]])

    accuracy, coverage = compute_live_accuracy(batch)

    metrics = dict(drift_metrics)
    metrics["n_predictions"] = len(batch)
    metrics["labeled_coverage"] = coverage
    if accuracy is not None:
        metrics["accuracy"] = accuracy

    reports_dir = Path("data/monitoring_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"drift_live_{now.strftime('%Y%m%dT%H%M%S')}.html"
    snapshot.save_html(str(report_path))

    with mlflow.start_run(run_name=f"live-check-{now.strftime('%Y%m%dT%H%M%S')}"):
        mlflow.log_params({"since": since.isoformat(), "until": now.isoformat()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))

    summary_path = reports_dir / "drift_summary_live.jsonl"
    entry = {
        "run_at": now.isoformat(),
        "since": since.isoformat(),
        "metrics": metrics,
        "report": str(report_path),
    }
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(metrics)
    save_last_check(now)
    print(f"\nTerminé. Résultats dans MLflow, expérience 'data-drift-monitoring-live'. "
          f"Historique JSON : {summary_path}")


if __name__ == "__main__":
    main()
