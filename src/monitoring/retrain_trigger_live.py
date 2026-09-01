"""
Version "trafic réel" du déclencheur de ré-entraînement : même mécanique que
retrain_trigger.py, mais la détection de drift et la décision s'appuient sur
le vrai trafic loggé dans la table `inference` (via drift_check_live.py) au
lieu de lots CSV fabriqués à l'avance.

Point important : la détection de drift regarde predicted_category (tout le
trafic, ça reste utile même sans vérité terrain), mais le RÉ-ENTRAÎNEMENT
n'utilise que les lignes où labeled_category est renseigné -- c'est la seule
vraie catégorie confirmée. Réentraîner sur predicted_category reviendrait à
apprendre au modèle à répéter ses propres erreurs.

Usage (depuis la racine du repo) :
    export DATABASE_URL=postgresql://user:pass@localhost:5432/rakuten
    export DRY_RUN=true   # recommandé pour un premier test
    python3 src/monitoring/retrain_trigger_live.py

Nécessite training.train (tensorflow/torch) pour un vrai ré-entraînement --
voir DRY_RUN dans retrain_trigger.py pour tester la mécanique sans ça.
"""

from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

from drift_check import MLFLOW_TRACKING_URI
from drift_check_live import compute_drift, compute_live_accuracy, get_since, load_current_batch_from_db, load_reference, save_last_check, MIN_ROWS_FOR_DRIFT
from retrain_trigger import DRY_RUN, retrain, run_pipeline_step, should_retrain

TRAIN_PATH = Path("data/train/raw.csv")
MIN_LABELED_ROWS_FOR_RETRAIN = 20  # en dessous, pas assez de vraies étiquettes pour ré-entraîner utilement


def append_labeled_rows_to_training_data(batch: pd.DataFrame) -> int:
    """Ajoute UNIQUEMENT les lignes avec une vraie catégorie confirmée
    (labeled_category) aux données d'entraînement -- jamais predicted_category.

    Retourne le nombre de lignes effectivement ajoutées (0 si aucune ligne
    labellisée dans le lot).
    """
    labeled = batch[batch["labeled_category"].notna()].copy()
    if len(labeled) < MIN_LABELED_ROWS_FOR_RETRAIN:
        print(
            f"  Seulement {len(labeled)} lignes avec une vraie catégorie confirmée "
            f"(minimum {MIN_LABELED_ROWS_FOR_RETRAIN}) -- pas assez pour ré-entraîner, "
            "même si le drift est réel."
        )
        return 0

    # inference_id sert de productid de substitution (la table inference n'en
    # a pas) -- non utilisé comme feature par le pipeline, juste un identifiant.
    to_append = pd.DataFrame({
        "productid": labeled["inference_id"] if "inference_id" in labeled.columns else range(len(labeled)),
        "designation": labeled["designation"],
        "description": labeled["description"],
        "prdtypecode": labeled["labeled_category"],  # la vraie cible, jamais predicted_category
    })

    existing = pd.read_csv(TRAIN_PATH, dtype={"prdtypecode": str})
    combined = pd.concat([existing, to_append], ignore_index=True)

    if DRY_RUN:
        preview_path = TRAIN_PATH.with_name("raw_dryrun_preview.csv")
        combined.to_csv(preview_path, index=False)
        print(
            f"  [DRY RUN] {len(to_append)} lignes labellisées AURAIENT été ajoutées "
            f"à {TRAIN_PATH} (total simulé : {len(combined)}). Aperçu dans {preview_path}."
        )
    else:
        combined.to_csv(TRAIN_PATH, index=False)
        print(f"  {len(to_append)} lignes labellisées ajoutées à {TRAIN_PATH} (total : {len(combined)})")

    return len(to_append)


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("data-drift-monitoring-live")

    since = get_since()
    now = datetime.now(timezone.utc)
    print(f"Fenêtre observée : {since.isoformat()} -> {now.isoformat()}")

    batch = load_current_batch_from_db(since)
    print(f"{len(batch)} prédictions trouvées sur cette fenêtre.")

    if len(batch) < MIN_ROWS_FOR_DRIFT:
        print(f"Moins de {MIN_ROWS_FOR_DRIFT} lignes -- pas assez pour un test fiable.")
        save_last_check(now)
        return

    reference = load_reference()
    drift_metrics, _snapshot = compute_drift(
        reference, batch[["designation", "description", "prdtypecode"]]
    )
    accuracy, coverage = compute_live_accuracy(batch)

    metrics = dict(drift_metrics)
    metrics["n_predictions"] = len(batch)
    metrics["labeled_coverage"] = coverage
    if accuracy is not None:
        metrics["accuracy"] = accuracy

    print(f"  {metrics}")

    trigger, reason = should_retrain(metrics)

    with mlflow.start_run(run_name=f"live-retrain-check-{now.strftime('%Y%m%dT%H%M%S')}"):
        mlflow.log_metrics(metrics)
        mlflow.log_metric("retrain_triggered", int(trigger))

        if trigger:
            print(f"  \u26a0\ufe0f  D\u00e9rive d\u00e9tect\u00e9e : {reason}")
            n_added = append_labeled_rows_to_training_data(batch)
            if n_added > 0:
                run_pipeline_step("src.rakuten.data.clean_data")
                run_pipeline_step("src.rakuten.data.build_features")
                run_pipeline_step("training.train")
                run_pipeline_step("training.compare_promote")
                mlflow.log_metric("retrain_executed", 1)
            else:
                print("  Drift confirmé mais pas assez de vraies étiquettes pour ré-entraîner.")
                mlflow.log_metric("retrain_executed", 0)
        else:
            print("  Pas de dérive significative, pas de ré-entraînement.")
            mlflow.log_metric("retrain_executed", 0)

    save_last_check(now)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
