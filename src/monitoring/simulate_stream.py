"""
Découpe le dataset complet (85k lignes, hors data/raw/data.csv utilisé pour
l'entraînement) en lots séquentiels, pour simuler des données qui "arrivent"
dans le temps (un lot = un jour), en vue de tester la détection de drift.

Usage (depuis la racine du repo) :
    python3 src/monitoring/simulate_stream.py
"""

from pathlib import Path

import pandas as pd

SOURCE_PATH = Path("data/full_dataset_85k.csv")
OUTPUT_DIR = Path("data/stream")
N_BATCHES = 10
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(SOURCE_PATH, sep=";", dtype={"prdtypecode": str})

    # Shuffle once, then slice into N_BATCHES equal chunks. A random shuffle
    # (rather than the original row order) avoids batches that would just
    # reflect however the raw file happened to be sorted.
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = len(df) // N_BATCHES

    for i in range(N_BATCHES):
        start = i * batch_size
        end = None if i == N_BATCHES - 1 else (i + 1) * batch_size
        batch = df.iloc[start:end]
        out_path = OUTPUT_DIR / f"batch_{i + 1:02d}.csv"
        batch.to_csv(out_path, sep=";", index=False)
        print(f"{out_path} : {len(batch)} lignes")

    print(f"\n{N_BATCHES} lots écrits dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
