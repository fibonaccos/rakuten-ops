"""
Copie uniquement les images utilisées par demo_products.csv depuis le dossier
complet d'images vers un dossier "demo_images" allégé.

Usage (depuis services/streamlit/) :
    python3 extraire_images_demo.py
"""

import csv
import shutil
from pathlib import Path

# Dossier où se trouvent TOUTES tes images (le dossier complet, déjà sur ta machine).
SOURCE_DIR = Path("data/images/images_train")

# Dossier allégé, créé par ce script, qui ne contiendra que les images du catalogue de démo.
TARGET_DIR = Path("data/images/demo_images")

CSV_PATH = Path("data/demo_products.csv")


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
