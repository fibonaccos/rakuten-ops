"""Home page: search bar + hero banner + a catalog of listings.

Category filtering is driven by the sidebar (st.session_state.category_filter,
see services/ui.py) rather than a widget on this page.

The catalog (`../../data/raw/demo_products.csv`) is a small sample built from the real
training data (X_train.csv / Y_train.csv) — it becomes a live API call once a
"browse listings" endpoint exists.
"""

import csv
from math import ceil
from pathlib import Path

import streamlit as st

from services.categories import CATEGORY_ORDER

PRODUCTS_PER_PAGE = 10

IMAGES_DIR = Path("../../data/images/demo_images")


def load_products() -> list[dict]:
    """Read the demo product catalog from ../../data/raw/demo_products.csv."""
    path = Path("../../data/raw/demo_products.csv")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def search_products(products: list[dict], query: str) -> list[dict]:
    """Keep products whose designation contains every keyword in the query."""
    keywords = query.lower().split()
    return [
        p for p in products
        if all(keyword in p["designation"].lower() for keyword in keywords)
    ]


def render_product_grid(products: list[dict]) -> None:
    """Display products in a 3-column grid of cards.

    Shows the real photo (../../data/images/demo_images/<image_base>.*, any
    extension) when it has been extracted (see extraire_images_demo.py),
    otherwise falls back to the emoji placeholder.
    """
    columns = st.columns(3)
    for index, product in enumerate(products):
        with columns[index % 3]:
            with st.container(border=True):
                image_path = next(IMAGES_DIR.glob(f"{product['image_base']}.*"), None)
                if image_path:
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.markdown(
                        f"<div class='product-thumb'>{product['emoji']}</div>",
                        unsafe_allow_html=True,
                    )
                st.write(f"**{product['designation']}**")
                st.write(product["price"])


def render_pagination(total_items: int) -> None:
    """Show Previous / page count / Next controls, backed by st.session_state.page."""
    total_pages = max(1, ceil(total_items / PRODUCTS_PER_PAGE))
    st.session_state.page = min(st.session_state.page, total_pages)

    col_prev, col_label, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Précédent", disabled=st.session_state.page <= 1):
            st.session_state.page -= 1
            st.rerun()
    with col_label:
        st.markdown(
            f"<p style='text-align:center;'>Page {st.session_state.page} / {total_pages}</p>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Suivant →", disabled=st.session_state.page >= total_pages):
            st.session_state.page += 1
            st.rerun()


# --- Session state defaults ---
if "page" not in st.session_state:
    st.session_state.page = 1

# --- Search bar (top of the page, styled like rakuten.fr) ---
with st.container(key="search-bar"):
    col_input, col_btn = st.columns([9, 1])
    with col_input:
        search_query = st.text_input(
            "Rechercher un produit",
            placeholder="Rechercher un produit",
            label_visibility="collapsed",
        )
    with col_btn:
        st.button("🔍")

# Reset to page 1 whenever the search query changes.
if search_query != st.session_state.get("last_query"):
    st.session_state.page = 1
    st.session_state.last_query = search_query

st.markdown(
    """
    <div class="hero">
        <h1>Donnez une seconde vie à vos objets</h1>
        <p>Déposez une annonce en quelques secondes : la classification automatique
        choisit la bonne catégorie à votre place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

products = load_products()
selected_category = st.session_state.get("category_filter")

if search_query:
    results = search_products(products, search_query)
    start = (st.session_state.page - 1) * PRODUCTS_PER_PAGE
    page_items = results[start : start + PRODUCTS_PER_PAGE]

    st.subheader(f"Résultats pour « {search_query} » ({len(results)})")
    if results:
        render_product_grid(page_items)
        render_pagination(len(results))
    else:
        st.info("Aucun produit ne correspond à cette recherche.")

elif selected_category:
    # Reset to page 1 whenever the category filter changes.
    if selected_category != st.session_state.get("last_category"):
        st.session_state.page = 1
        st.session_state.last_category = selected_category

    filtered = [p for p in products if p["category"] == selected_category]
    start = (st.session_state.page - 1) * PRODUCTS_PER_PAGE
    page_items = filtered[start : start + PRODUCTS_PER_PAGE]

    st.subheader(f"Produits — {selected_category} ({len(filtered)})")
    render_product_grid(page_items)
    render_pagination(len(filtered))

else:
    st.subheader("Top des produits")
    # One featured item per top category, to keep the homepage preview short.
    featured = [next(p for p in products if p["category"] == c) for c in CATEGORY_ORDER[:9]]
    render_product_grid(featured)
