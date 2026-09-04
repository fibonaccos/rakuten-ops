"""
Génère 3 fichiers d'échantillons pour Locust (benchmark/locustfile.py), avec
des proportions de catégories volontairement différentes -- comme
simulate_stream.py, mais au format attendu par load_samples_pool() (parquet,
colonnes designation/description uniquement, pas de catégorie : Locust les
envoie à l'API sans connaître la vraie réponse).

Ne modifie PAS locustfile.py : chaque "vague" se lance en pointant DATA_FILE
vers un fichier différent.

Usage (depuis la racine du repo, une fois split_train_prod.py déjà lancé) :
    python3 src/monitoring/generate_locust_samples.py

Puis, pour chaque vague :
    export DATA_FILE=benchmark/data/samples_wave1.parquet
    # relancer le conteneur/service locust normalement
"""

import random
from pathlib import Path

import pandas as pd

SOURCE_PATH = Path("data/prod/raw.csv")
OUTPUT_DIR = Path("benchmark/data")
SAMPLE_SIZE = 2000
RANDOM_STATE = 42

WAVE_PROFILES = [
    {"name": "samples_wave1", "weight_top_n": 0, "weight_factor": 1.0},   # calme
    {"name": "samples_wave2", "weight_top_n": 10, "weight_factor": 3.0},  # dérive modérée
    {"name": "samples_wave3", "weight_top_n": 3, "weight_factor": 12.0},  # dérive forte
]


def build_skewed_sample(df: pd.DataFrame, profile: dict, rng: random.Random) -> pd.DataFrame:
    categories = sorted(df["prdtypecode"].unique())
    weights_by_category = {cat: 1.0 for cat in categories}
    if profile["weight_top_n"] > 0:
        boosted = rng.sample(categories, min(profile["weight_top_n"], len(categories)))
        for cat in boosted:
            weights_by_category[cat] = profile["weight_factor"]

    row_weights = df["prdtypecode"].map(weights_by_category)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), weights=row_weights, random_state=RANDOM_STATE)
    # load_samples_pool() ne lit que designation/description -- inutile de
    # garder prdtypecode dans le fichier livré à Locust.
    return sample[["designation", "description"]]


def main() -> None:
    df = pd.read_csv(SOURCE_PATH, dtype={"prdtypecode": str})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for profile in WAVE_PROFILES:
        rng = random.Random(RANDOM_STATE)
        sample = build_skewed_sample(df, profile, rng)
        out_path = OUTPUT_DIR / f"{profile['name']}.parquet"
        sample.to_parquet(out_path, index=False)
        print(f"{out_path} : {len(sample)} lignes")

    print(f"\n{len(WAVE_PROFILES)} fichiers écrits dans {OUTPUT_DIR}/")
    print("\nPour lancer une vague :")
    for profile in WAVE_PROFILES:
        print(f"  export DATA_FILE={OUTPUT_DIR}/{profile['name']}.parquet")


if __name__ == "__main__":
    main()
