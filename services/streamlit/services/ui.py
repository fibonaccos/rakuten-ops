"""Shared UI helpers used by every Streamlit page (home + pages/)."""

from pathlib import Path

import streamlit as st


def load_css(path: str) -> str:
    """Read a CSS file from disk as plain text."""
    return Path(path).read_text(encoding="utf-8")


def inject_css(path: str = "static/style.css") -> None:
    """Inject the shared stylesheet into the current page."""
    st.markdown(f"<style>{load_css(path)}</style>", unsafe_allow_html=True)


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
