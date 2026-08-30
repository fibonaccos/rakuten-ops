"""
Copie uniquement les images utilisées par data/raw/demo_products.csv depuis le
dossier complet d'images vers data/images (le dossier léger utilisé par Streamlit).

Usage (depuis la racine du repo) :
    python3 src/rakuten/data/extraire_images_demo.py
"""

import csv
import shutil
from pathlib import Path

# Dossier où se trouvent TOUTES les images (le dossier complet).
SOURCE_DIR = Path("data/images/image_train")

# Sous-dossier allégé, utilisé par services/streamlit/views/accueil.py.
TARGET_DIR = Path("data/images/demo_images")

CSV_PATH = Path("data/raw/demo_products.csv")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    found, missing = 0, 0
    for row in rows:
        match = next(SOURCE_DIR.glob(f"{row['image_base']}.*"), None)
        if match is None:
            missing += 1
            continue
        shutil.copy2(match, TARGET_DIR / match.name)
        found += 1

    print(f"{found} images copiées dans {TARGET_DIR}")
    print(f"{missing} images introuvables dans {SOURCE_DIR}")


if __name__ == "__main__":
    main()
