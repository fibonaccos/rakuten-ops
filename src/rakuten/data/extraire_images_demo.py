"""
Copie uniquement les images utilisées par services/streamlit/data/demo_products.csv
depuis le dossier complet d'images vers services/streamlit/data/images/demo_images
(le dossier léger utilisé par services/catalog.py).

Écrire dans services/streamlit/data/ (et pas à la racine du repo) est important :
c'est le seul moyen pour que ces fichiers soient inclus dans l'image Docker du
front-end (Dockerfile.app fait `COPY services/streamlit .`, donc rien en dehors
de ce dossier n'atterrit dans le conteneur).

Usage (depuis la racine du repo) :
    python3 src/rakuten/data/extraire_images_demo.py
"""

import csv
import shutil
from pathlib import Path

# Dossier où se trouvent TOUTES les images (le dossier complet, hors du repo Git).
SOURCE_DIR = Path("data/images")

# Sous-dossier allégé, utilisé par services/streamlit/services/catalog.py.
TARGET_DIR = Path("services/streamlit/data/images/demo_images")

CSV_PATH = Path("services/streamlit/data/demo_products.csv")


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
