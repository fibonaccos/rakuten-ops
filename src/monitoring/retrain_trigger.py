"""
Boucle de monitoring + ré-entraînement automatique.

Pour chaque lot simulé (data/stream/batch_XX.csv, voir simulate_stream.py) :
  1. calcule le drift vs data/train/raw.csv (comme drift_check.py)
  2. évalue la performance du modèle en prod sur un échantillon du lot
  3. si le drift ou la perf dépasse un seuil, déclenche un ré-entraînement :
       - ajoute le lot aux données d'entraînement (data/train/raw.csv)
       - relance clean_data -> build_features -> train -> compare_promote

Rien n'est réinventé côté ré-entraînement : compare_promote.py fait déjà
l'auto-promotion champion/challenger (champion_challenger.auto_promote=true
dans training/params.yaml) -- ce script se contente d'enchaîner le pipeline
existant quand le drift le justifie.

Usage (depuis la racine du repo) :
    export MONITORING_API_USERNAME=...
    export MONITORING_API_PASSWORD=...
    python3 src/monitoring/retrain_trigger.py
"""

import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

from drift_check import (
    MLFLOW_TRACKING_URI,
    TEXT_COLUMNS,
    compute_accuracy,
    compute_drift,
    get_api_token,
    load_batches,
    load_reference,
)

TRAIN_PATH = Path("data/train/raw.csv")

# Seuils de déclenchement -- valeurs de départ raisonnables, à recalibrer une
# fois qu'on a de vrais chiffres de référence (voir le premier run MLflow
# sans ré-entraînement, celui de drift_check.py, pour choisir des seuils
# réalistes plutôt qu'arbitraires).
DRIFT_SHARE_THRESHOLD = 0.3  # part des colonnes surveillées en drift significatif
ACCURACY_THRESHOLD = 0.70  # accuracy sous ce seuil sur l'échantillon du lot

# Quand DRY_RUN=true, la détection et la décision tournent normalement, mais
# rien n'est réellement écrit ni lancé -- utile pour valider la mécanique
# (append + orchestration) sans dépendre de tensorflow/torch ni risquer
# d'écraser data/train/raw.csv.
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"


def should_retrain(metrics: dict) -> tuple[bool, str]:
    """Décide si un lot justifie un ré-entraînement, et pourquoi."""
    drift_share = metrics.get("drift_share", 0.0)
    if drift_share > DRIFT_SHARE_THRESHOLD:
        return True, f"drift_share={drift_share:.2f} > seuil {DRIFT_SHARE_THRESHOLD}"

    accuracy = metrics.get("accuracy")
    if accuracy is not None and accuracy < ACCURACY_THRESHOLD:
        return True, f"accuracy={accuracy:.2f} < seuil {ACCURACY_THRESHOLD}"

    return False, ""


def append_batch_to_training_data(batch: pd.DataFrame) -> None:
    """Ajoute un lot aux données d'entraînement, avant de relancer le pipeline.

    Sans cette étape, relancer l'entraînement n'apporterait rien : le modèle
    reverrait exactement les mêmes données qu'avant, pas celles qui ont drifté.
    """
    columns = ["productid", "designation", "description", "prdtypecode"]
    existing = pd.read_csv(TRAIN_PATH, sep=",", dtype={"prdtypecode": str})
    combined = pd.concat([existing, batch[columns]], ignore_index=True)

    if DRY_RUN:
        preview_path = TRAIN_PATH.with_name("raw_dryrun_preview.csv")
        combined.to_csv(preview_path, index=False)
        print(
            f"  [DRY RUN] {len(batch)} lignes AURAIENT été ajoutées à {TRAIN_PATH} "
            f"(total simulé : {len(combined)}). Aperçu écrit dans {preview_path}, "
            f"{TRAIN_PATH} n'a pas été touché."
        )
        return

    combined.to_csv(TRAIN_PATH, sep=",", index=False)
    print(f"  {len(batch)} lignes ajoutées à {TRAIN_PATH} (total : {len(combined)})")


def run_pipeline_step(module: str) -> None:
    """Lance une étape du pipeline existant via `python -m`, comme le fait l'équipe."""
    if DRY_RUN:
        print(f"  [DRY RUN] aurait lancé : python -m {module}")
        return
    print(f"  -> python -m {module}")
    result = subprocess.run([sys.executable, "-m", module])
    if result.returncode != 0:
        raise RuntimeError(f"{module} a échoué (code {result.returncode})")


def retrain() -> None:
    """Enchaîne le pipeline existant de l'équipe : rien de nouveau, juste orchestré."""
    run_pipeline_step("src.rakuten.data.clean_data")
    run_pipeline_step("src.rakuten.data.build_features")
    run_pipeline_step("training.train")
    run_pipeline_step("training.compare_promote")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("data-drift-monitoring")

    reference = load_reference()
    batches = load_batches()
    if not batches:
        print("Aucun lot trouvé -- lance simulate_stream.py d'abord.")
        return

    token = get_api_token()

    with mlflow.start_run(run_name="drift-monitoring-with-retrain"):
        for step, batch_path in enumerate(batches, start=1):
            print(f"--- Lot {step}/{len(batches)} : {batch_path.name} ---")
            batch = pd.read_csv(batch_path, sep=",", dtype={"prdtypecode": str})
            for col in TEXT_COLUMNS:
                batch[col] = batch[col].fillna("")

            drift_metrics, _snapshot = compute_drift(reference, batch)
            metrics = dict(drift_metrics)
            if token:
                accuracy = compute_accuracy(batch, token)
                if accuracy is not None:
                    metrics["accuracy"] = accuracy

            print(f"  {metrics}")
            mlflow.log_metrics(metrics, step=step)

            trigger, reason = should_retrain(metrics)
            mlflow.log_metric("retrain_triggered", int(trigger), step=step)

            if trigger:
                print(f"  ⚠️  Ré-entraînement déclenché : {reason}")
                append_batch_to_training_data(batch)
                retrain()
                print(
                    "  Ré-entraînement terminé (voir l'expérience MLflow "
                    "du pipeline pour le détail champion/challenger)."
                )
            else:
                print("  Pas de dérive significative, pas de ré-entraînement.")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
