import json

import mlflow

from box import Box
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Any
from yaml import safe_load

from training.package_model import require_tracking_uri, EXPERIMENT


PARAMS_FILEPATH: Path = Path(__file__).parent / "params.yaml"
ROOT_DIR: Path = Path(__file__).parent.parent

with open(PARAMS_FILEPATH, "r") as f:
    _PARAMS: dict[str, Any] = safe_load(f)


def get_test_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    """
    Récupère les métriques de test (préfixées "test_") déjà loggées par
    `training/train.py` pour un run donné -- sans rien recalculer. Chaque run de
    `training/train.py` évalue son modèle sur le même jeu de test gelé
    (data/features/x_test.parquet, figé tant que la configuration du split dans
    `training/params.yaml` ne change pas) juste avant de packager le modèle : ces
    métriques sont donc directement comparables d'un run à l'autre.

    Args:
        client (MlflowClient): Le client MLflow.
        run_id (str): L'identifiant du run à interroger.

    Returns:
        dict[str, float]: Les métriques de test, sans le préfixe "test_" (ex.
            "f1_macro" plutôt que "test_f1_macro").
    """

    run = client.get_run(run_id)
    metrics = {
        key.removeprefix("test_"): value
        for key, value in run.data.metrics.items()
        if key.startswith("test_") and key != "test_support"
    }
    if not metrics:
        raise RuntimeError(
            f"Aucune métrique 'test_*' trouvée pour le run {run_id}")
    return metrics


def decide_promotion(
        champion_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
        cfg: Box
    ) -> tuple[bool, list[str]]:
    """
    Règle de promotion à 2 conditions :
      1. amélioration de la métrique principale >= seuil minimal
      2. pas de régression significative sur les métriques annexes.

    Args:
        champion_metrics (dict[str, float]): Métriques du Champion.
        challenger_metrics (dict[str, float]): Métriques du Challenger.
        cfg (Box): La section `champion_challenger` de training/params.yaml.

    Returns:
        tuple: (promote, reasons), la décision et la liste des justifications.
    """

    reasons: list[str] = []

    primary = cfg.primary_metric
    delta_primary = challenger_metrics[primary] - champion_metrics[primary]
    ok_improvement = delta_primary >= cfg.min_improvement
    reasons.append(
        f"[{'OK' if ok_improvement else 'KO'}] delta {primary} = {delta_primary:+.4f} "
        f"(seuil minimal requis : {cfg.min_improvement:+.4f})"
    )

    ok_regression = True
    for metric, tolerance in cfg.regression_tolerance.items():
        delta = challenger_metrics[metric] - champion_metrics[metric]
        passed = delta >= -tolerance
        ok_regression = ok_regression and passed
        reasons.append(
            f"[{'OK' if passed else 'KO'}] delta {metric} = {delta:+.4f} "
            f"(régression tolérée : -{tolerance:.4f})"
        )

    promote = ok_improvement and ok_regression
    return promote, reasons


def compare_and_promote() -> dict:
    """
    Point d'entrée principal : compare le Challenger (dernière version enregistrée du
    modèle) au Champion actuel (alias @champion), en relisant les métriques de test
    déjà loggées par training/train.py pour chacun.
    Promeut le Challenger si la règle de decide_promotion est satisfaite, sinon conserve le Champion.
    Si aucun Champion n'existe encore, le premier candidat est promu directement

    La décision complète (métriques des deux modèles, verdict et justifications) est
    loggée dans le run actif et comme artefact JSON lisible.

    Returns:
        dict: Le rapport de décision.
    """

    params_all: Box = Box(_PARAMS)
    cfg: Box = params_all.champion_challenger

    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    registered_name = cfg.registered_model_name
    champion_alias = cfg.champion_alias
    challenger_alias = cfg.challenger_alias

    all_versions = client.search_model_versions(f"name='{registered_name}'")
    if not all_versions:
        raise RuntimeError(
            f"Aucune version enregistrée pour '{registered_name}' "
        )
    challenger_mv = max(all_versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(registered_name, challenger_alias, challenger_mv.version)

    challenger_metrics = get_test_metrics(client, challenger_mv.run_id)

    try:
        champion_mv = client.get_model_version_by_alias(registered_name, champion_alias)
        has_champion = True
    except MlflowException:
        has_champion = False

    mlflow.log_param("challenger_version", challenger_mv.version)
    mlflow.log_metrics({f"challenger_{k}": v for k, v in challenger_metrics.items()})

    if not has_champion:
        client.set_registered_model_alias(registered_name, champion_alias, challenger_mv.version)
        client.delete_registered_model_alias(registered_name, challenger_alias)
        mlflow.log_param("champion_challenger_decision", "Promotion_auto")
        report = {
            "decision": "Promotion_auto",
            "reason": "Aucun champion existant : première version promue directement.",
            "champion_version": challenger_mv.version,
            "challenger_metrics": challenger_metrics,
        }
        print(f"Aucun Champion existant : la version {challenger_mv.version} devient Champion.")
        report_path = ROOT_DIR / "artifacts" / "champion_challenger_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))
        return report

    champion_metrics = get_test_metrics(client, champion_mv.run_id)
    mlflow.log_param("champion_version", champion_mv.version)
    mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})

    promote, reasons = decide_promotion(champion_metrics, challenger_metrics, cfg)
    decision = "promoted" if promote else "kept_champion"
    mlflow.log_param("champion_challenger_decision", decision)

    report = {
        "champion_version": champion_mv.version,
        "challenger_version": challenger_mv.version,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "decision": decision,
        "reasons": reasons,
    }
    report_path = ROOT_DIR / "artifacts" / "champion_challenger_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(str(report_path))

    print("\n".join(reasons))
    if promote:
        client.set_registered_model_alias(registered_name, champion_alias, challenger_mv.version)
        print(f"Challenger (v{challenger_mv.version}) promu Champion.")
    else:
        print(f"Champion (v{champion_mv.version}) gardé.")

    client.delete_registered_model_alias(registered_name, challenger_alias)
    return report


if __name__ == "__main__":
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="champion_challenger_comparison"):
        compare_and_promote()