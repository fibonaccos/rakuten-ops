"""
Découpe le jeu "prod" (le vrai held-out mis de côté par
src/rakuten/data/split_train_prod.py, jamais vu à l'entraînement) en 3 lots,
avec des proportions de catégories volontairement DIFFÉRENTES d'un lot à
l'autre -- pour simuler des données qui changent de tendance dans le temps,
pas juste 3 échantillons aléatoires de la même distribution.

  - batch_01 : proportions proches de l'original (période "calme")
  - batch_02 : dérive modérée (certaines catégories sur-représentées)
  - batch_03 : dérive forte, dominé par une poignée de catégories seulement

Usage (depuis la racine du repo, une fois split_train_prod.py déjà lancé) :
    python3 src/monitoring/simulate_stream.py
"""

import random
from pathlib import Path

import pandas as pd

SOURCE_PATH = Path("data/prod/raw.csv")
OUTPUT_DIR = Path("data/stream")
BATCH_SIZE = 600
RANDOM_STATE = 42

# Un batch par profil de dérive. `weight_top_n` = combien de catégories sont
# sur-pondérées, `weight_factor` = à quel point (1.0 = pas de biais du tout).
BATCH_PROFILES = [
    {"name": "batch_01", "weight_top_n": 0, "weight_factor": 1.0},   # calme
    {"name": "batch_02", "weight_top_n": 10, "weight_factor": 3.0},  # dérive modérée
    {"name": "batch_03", "weight_top_n": 3, "weight_factor": 12.0},  # dérive forte
]


def build_skewed_batch(df: pd.DataFrame, profile: dict, rng: random.Random) -> pd.DataFrame:
    """Échantillonne un lot en sur-pondérant volontairement quelques catégories."""
    categories = sorted(df["prdtypecode"].unique())

    weights_by_category = {cat: 1.0 for cat in categories}
    if profile["weight_top_n"] > 0:
        boosted = rng.sample(categories, min(profile["weight_top_n"], len(categories)))
        for cat in boosted:
            weights_by_category[cat] = profile["weight_factor"]

    row_weights = df["prdtypecode"].map(weights_by_category)
    return df.sample(n=BATCH_SIZE, weights=row_weights, random_state=RANDOM_STATE)


def main() -> None:
    df = pd.read_csv(SOURCE_PATH, dtype={"prdtypecode": str})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Nettoie les lots d'un essai précédent (ex: l'ancienne version à 10 lots)
    # pour que drift_check.py ne ramasse pas de fichiers périmés.
    for stale_batch in OUTPUT_DIR.glob("batch_*.csv"):
        stale_batch.unlink()

    for profile in BATCH_PROFILES:
        rng = random.Random(RANDOM_STATE)
        batch = build_skewed_batch(df, profile, rng)

        out_path = OUTPUT_DIR / f"{profile['name']}.csv"
        batch.to_csv(out_path, index=False)

        top_categories = batch["prdtypecode"].value_counts().head(5)
        print(f"{out_path} : {len(batch)} lignes -- top catégories :")
        print(top_categories.to_string())
        print()

    print(f"{len(BATCH_PROFILES)} lots écrits dans {OUTPUT_DIR}/ (source : {SOURCE_PATH})")


if __name__ == "__main__":
    main()
