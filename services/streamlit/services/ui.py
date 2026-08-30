"""Shared UI helpers used by every Streamlit page (home + pages/)."""

from pathlib import Path

import streamlit as st

from services.categories import CATEGORY_ORDER


def load_css(path: str) -> str:
    """Read a CSS file from disk as plain text."""
    return Path(path).read_text(encoding="utf-8")


def inject_css(path: str = "static/style.css") -> None:
    """Inject the shared stylesheet into the current page."""
    st.markdown(f"<style>{load_css(path)}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Display the Rakuten-style header: logo on the left, sell link + account menu on the right."""
    col_logo, col_spacer, col_sell, col_account = st.columns([3, 4, 1, 1])

    with col_logo:
        st.markdown('<div class="logo">Rakuten</div>', unsafe_allow_html=True)

    with col_sell:
        with st.container(key="header-sell"):
            st.page_link("views/vendre.py", label="Vendre", icon="💶")

    with col_account:
        with st.container(key="account-menu"):
            logged_in = bool(st.session_state.get("token"))
            trigger_label = st.session_state.username if logged_in else "Se connecter"
            st.markdown(f'<div class="account-trigger">👤 {trigger_label}</div>', unsafe_allow_html=True)

            with st.container(key="account-dropdown"):
                if logged_in:
                    st.page_link("views/vendre.py", label="Vendre un produit", icon="🛒")
                    if st.button("Se déconnecter", key="header-logout"):
                        st.session_state.token = None
                        st.session_state.username = None
                        st.session_state.role = None
                        st.rerun()
                else:
                    st.page_link("views/connexion.py", label="Se connecter", icon="🔑")
                    st.caption("Créer un compte — bientôt disponible")

    st.markdown('<hr class="header-rule">', unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render the app's full sidebar navigation: admin shortcut, main links,
    an expandable category list, and the logged-in user / logout section.
    """
    with st.sidebar:
        if st.session_state.get("role") == "admin":
            with st.container(key="admin-highlight"):
                st.page_link("views/administration.py", label="Administration", icon="🛠️")

        st.page_link("views/accueil.py", label="Accueil", icon="🏠")
        st.page_link("views/vendre.py", label="Vendre un produit", icon="🛒")

        with st.expander("📂 Catégories"):
            if st.session_state.category_filter:
                if st.button("✕ Retirer le filtre", key="clear-category-filter"):
                    st.session_state.category_filter = None
                    st.switch_page("views/accueil.py")
            for category in CATEGORY_ORDER:
                active = st.session_state.category_filter == category
                label = f"● {category}" if active else category
                if st.button(label, key=f"sidebar-cat-{category}", use_container_width=True):
                    st.session_state.category_filter = category
                    st.switch_page("views/accueil.py")

        if st.session_state.get("token"):
            st.divider()
            st.write(f"Connecté : **{st.session_state.username}** ({st.session_state.role})")
            if st.button("Se déconnecter", key="sidebar-logout"):
                st.session_state.token = None
                st.session_state.username = None
                st.session_state.role = None
                st.session_state.category_filter = None
                st.rerun()


def render_footer() -> None:
    """Display a 3-column footer: app links, help placeholders, and technical
    resources (API docs, MLflow, Grafana) useful for the team.
    """
    st.divider()
    with st.container(key="footer"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="footer-col-title">Liens utiles</div>', unsafe_allow_html=True)
            st.page_link("views/accueil.py", label="Accueil")
            st.page_link("views/vendre.py", label="Vendre un produit")
            if st.session_state.get("token"):
                st.page_link("views/vendre.py", label="Mon compte")
            else:
                st.page_link("views/connexion.py", label="Se connecter")

        with col2:
            st.markdown('<div class="footer-col-title">Aide</div>', unsafe_allow_html=True)
            st.caption("Centre d'aide — bientôt disponible")
            st.caption("Nous contacter — bientôt disponible")
            st.caption("Vendre en toute confiance — bientôt disponible")

        with col3:
            st.markdown('<div class="footer-col-title">Ressources techniques</div>', unsafe_allow_html=True)
            st.markdown("[Documentation API](http://localhost:8000/docs)")
            st.markdown("[MLflow](http://localhost:5001)")
            st.markdown("[Grafana](http://localhost:3000)")
