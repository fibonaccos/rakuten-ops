"""Home page: search bar + hero banner + a catalog of listings.

Category filtering is driven by the sidebar (st.session_state.category_filter,
see services/ui.py) rather than a widget on this page.

Reading the catalogue, and the ordering of the category chips, both live in
services/catalog.py so the sell page can reuse them.
"""

from math import ceil

import streamlit as st

from services.catalog import (
    categories_by_popularity,
    image_for,
    load_products,
    products_in_category,
    search,
)

PRODUCTS_PER_PAGE = 10
FEATURED_CATEGORIES = 9


def render_product_grid(products: list[dict]) -> None:
    """Display products in a 3-column grid of cards.

    Shows the real photo (data/images/demo_images/<image_base>.*, any
    extension) when it has been extracted (see extraire_images_demo.py),
    otherwise falls back to the emoji placeholder.
    """
    columns = st.columns(3)
    for index, product in enumerate(products):
        with columns[index % 3], st.container(border=True):
            image_path = image_for(product)
            if image_path:
                st.image(str(image_path), width="stretch")
            else:
                st.markdown(
                    f"<div class='product-thumb'>{product['emoji']}</div>",
                    unsafe_allow_html=True,
                )
            st.write(f"**{product['designation']}**")
            st.write(product["price"])


def render_page(items: list[dict]) -> list[dict]:
    """Return the slice of `items` belonging to the current page."""
    start = (st.session_state.page - 1) * PRODUCTS_PER_PAGE
    return items[start : start + PRODUCTS_PER_PAGE]


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


def article_count(n: int) -> str:
    """Pluralize "article" correctly: '1 article', '2 articles'."""
    return f"{n} article" if n <= 1 else f"{n} articles"


def reset_page_on_change(key: str, value) -> None:
    """Send the reader back to page 1 whenever the search or the filter changes."""
    if value != st.session_state.get(key):
        st.session_state.page = 1
        st.session_state[key] = value


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
reset_page_on_change("last_query", search_query)

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

if search_query:
    results = search(search_query)

    st.subheader(f"Résultats pour « {search_query} » ({article_count(len(results))})")
    if results:
        render_product_grid(render_page(results))
        render_pagination(len(results))
    else:
        st.info("Aucun produit ne correspond à cette recherche.")

else:
    selected_category = st.session_state.get("category_filter")
    reset_page_on_change("last_category", selected_category)

    if selected_category:
        filtered = products_in_category(selected_category)

        st.subheader(f"Produits — {selected_category} ({article_count(len(filtered))})")
        render_product_grid(render_page(filtered))
        render_pagination(len(filtered))
    else:
        categories = categories_by_popularity()
        st.subheader("Top des produits")
        st.caption(f"{len(load_products())} annonces réparties sur {len(categories)} catégories.")
        # One featured item per top category, to keep the homepage preview short.
        featured = [
            products_in_category(category)[0]
            for category in categories[:FEATURED_CATEGORIES]
        ]
        render_product_grid(featured)
