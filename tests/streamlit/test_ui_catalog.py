"""The demo catalogue, and the mapping from a model output to a readable label."""

import pytest


@pytest.fixture
def catalog(service):
    import services.catalog as module

    return module


def test_the_catalogue_loads_every_listing(catalog) -> None:
    products = catalog.load_products()

    assert len(products) > 500
    assert set(products[0]) >= {"designation", "prdtypecode", "category", "price", "emoji"}


def test_the_catalogue_resolves_from_the_package_not_the_working_directory(catalog) -> None:
    """The container runs from /app, a developer runs from the repository root."""
    assert catalog.CATALOG_FILE.is_absolute()
    assert catalog.CATALOG_FILE.is_file()
    assert catalog.STYLESHEET.is_file()


def test_a_prdtypecode_maps_to_its_french_category(catalog) -> None:
    products = catalog.load_products()
    sample = products[0]

    assert catalog.label_for(sample["prdtypecode"]) == sample["category"]


def test_a_prdtypecode_maps_the_same_whether_it_is_text_or_a_number(catalog) -> None:
    code = catalog.load_products()[0]["prdtypecode"]

    assert catalog.label_for(int(code)) == catalog.label_for(code)


def test_an_unknown_code_falls_back_to_a_readable_label(catalog) -> None:
    """A model trained on new classes must not print a bare code to a seller."""
    assert catalog.label_for("999999") == catalog.UNKNOWN_CATEGORY


def test_every_category_of_the_catalogue_is_reachable_from_a_code(catalog) -> None:
    categories = {product["category"] for product in catalog.load_products()}
    codes = {product["prdtypecode"] for product in catalog.load_products()}

    assert {catalog.label_for(code) for code in codes} == categories


def test_categories_are_ordered_by_how_many_listings_they_hold(catalog) -> None:
    ordered = catalog.categories_by_popularity()

    sizes = [len(catalog.products_in_category(name)) for name in ordered]
    assert sizes == sorted(sizes, reverse=True)
    assert len(ordered) == 27


def test_filtering_by_category_keeps_only_that_category(catalog) -> None:
    name = catalog.categories_by_popularity()[0]

    assert {p["category"] for p in catalog.products_in_category(name)} == {name}


def test_search_matches_every_keyword(catalog) -> None:
    results = catalog.search("de la")

    assert results
    for product in results:
        assert "de" in product["designation"].lower()
        assert "la" in product["designation"].lower()


def test_search_is_case_insensitive(catalog) -> None:
    assert catalog.search("LIVRE") == catalog.search("livre")


def test_an_empty_search_returns_nothing(catalog) -> None:
    assert catalog.search("   ") == []


def test_a_listing_without_an_extracted_photo_has_no_image(catalog) -> None:
    """Images are optional: extraire_images_demo.py has usually not been run."""
    product = dict(catalog.load_products()[0], image_base="pas-une-image")

    assert catalog.image_for(product) is None
