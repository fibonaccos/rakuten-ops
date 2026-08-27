"""Shared UI helpers used by every Streamlit page (home + pages/)."""

from pathlib import Path

import streamlit as st

from services.catalog import STYLESHEET, label_for


def load_css(path: str) -> str:
    """Read a CSS file from disk as plain text."""
    return Path(path).read_text(encoding="utf-8")


def inject_css(path: str | Path = STYLESHEET) -> None:
    """Inject the shared stylesheet into the current page."""
    st.markdown(f"<style>{load_css(str(path))}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Display the Rakuten-style logo header at the top of the page."""
    st.markdown(
        """
        <div class="app-header">
            <div class="logo">Rakuten</div>
            <div class="tagline">Vendez, achetez, simplement.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_user() -> None:
    """Show the logged-in user and a logout button in the sidebar, if connected."""
    if not st.session_state.get("token"):
        return

    with st.sidebar:
        st.write(f"Connecté : **{st.session_state.username}** ({st.session_state.role})")
        if st.button("Se déconnecter"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()


def render_prediction(category_code: str, confidence: float) -> None:
    """
    Announce the predicted category, with the raw code kept in sight.

    Args:
        category_code: The prdtypecode the classifier returned, e.g. "2583".
        confidence: Probability of that class, between 0 and 1.
    """
    st.markdown(
        f"""
        <div class="prediction">
            <div class="eyebrow">Catégorie suggérée</div>
            <div class="category">{label_for(category_code)}</div>
            <div class="code">code {category_code} &middot; confiance {confidence:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_distribution(distribution: dict[str, float], top: int = 5) -> None:
    """
    Draw the most likely categories as ranked bars.

    The runner-up matters as much as the winner: a 41 % / 39 % split is a very
    different answer from 96 % / 1 %, and a plain list hides that.

    Args:
        distribution: Probability per prdtypecode, as returned by the API.
        top: How many categories to draw.
    """
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:top]
    if not ranked:
        return

    rows = []
    for rank, (code, probability) in enumerate(ranked):
        css_class = "dist-row" if rank == 0 else "dist-row runner-up"
        width = max(1.0, 100 * probability)
        rows.append(
            f"<div class='{css_class}'>"
            f"<span class='name' title='code {code}'>{label_for(code)}</span>"
            f"<span class='track'><span class='fill' style='width:{width:.1f}%'></span></span>"
            f"<span class='value'>{probability:.1%}</span>"
            f"</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_status(label: str, state: str) -> None:
    """
    Show one backend as an up/down pill.

    Args:
        label: Human name of the service, e.g. "Base de données".
        state: The value the API reported; anything but "ready" reads as down.
    """
    is_up = state == "ready"
    css_class = "status up" if is_up else "status down"
    text = "Disponible" if is_up else "Indisponible"
    st.markdown(
        f"<div class='status-label'>{label}</div>"
        f"<span class='{css_class}'>{text}</span>",
        unsafe_allow_html=True,
    )
