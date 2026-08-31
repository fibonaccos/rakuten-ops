"""
The demo catalogue, and the one place that turns a model output into French.

The classifier answers with a `prdtypecode` such as "2583". The catalogue built
from the training data (`data/demo_products.csv`) already carries the readable
name for every code, so the same file serves three purposes: the listings shown
on the home page, the category filter, and the label put on a prediction.

Paths resolve from this file rather than from the working directory, so the app
runs the same whether it is started from the repository root or from `/app`
inside the container.
"""

import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent
CATALOG_FILE: Path = PACKAGE_ROOT / "data" / "demo_products.csv"
IMAGES_DIR: Path = PACKAGE_ROOT / "data" / "images" / "demo_images"
STYLESHEET: Path = PACKAGE_ROOT / "static" / "style.css"

UNKNOWN_CATEGORY = "Catégorie inconnue"


@lru_cache(maxsize=1)
def load_products() -> tuple[dict[str, str], ...]:
    """
    Read the demo catalogue.

    Returns:
        tuple: One dict per listing, in file order. Immutable so the cached value
            cannot be mutated by a caller.
    """
    with CATALOG_FILE.open(encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


@lru_cache(maxsize=1)
def _labels_by_code() -> dict[str, str]:
    return {product["prdtypecode"]: product["category"] for product in load_products()}


@lru_cache(maxsize=1)
def categories_by_popularity() -> tuple[str, ...]:
    """Category names, most represented in the catalogue first."""
    counts = Counter(product["category"] for product in load_products())
    return tuple(name for name, _ in counts.most_common())


def label_for(prdtypecode: str | int) -> str:
    """
    Translate a model output into the category name a seller would recognise.

    Args:
        prdtypecode: The class the classifier returned, e.g. "2583".

    Returns:
        str: The French category name, or a readable fallback when the code is
            not one of the 27 the catalogue covers.
    """
    return _labels_by_code().get(str(prdtypecode), UNKNOWN_CATEGORY)


def products_in_category(category: str) -> list[dict[str, str]]:
    """Every listing filed under one category name."""
    return [product for product in load_products() if product["category"] == category]


def search(query: str) -> list[dict[str, str]]:
    """
    Listings whose designation contains every keyword of the query.

    Args:
        query: Free text typed in the search box; empty returns nothing.
    """
    keywords = query.lower().split()
    if not keywords:
        return []
    return [
        product
        for product in load_products()
        if all(keyword in product["designation"].lower() for keyword in keywords)
    ]


def image_for(product: dict[str, str]) -> Path | None:
    """The extracted photo of a listing, when `extraire_images_demo.py` has run."""
    if not IMAGES_DIR.is_dir():
        return None
    return next(IMAGES_DIR.glob(f"{product['image_base']}.*"), None)
