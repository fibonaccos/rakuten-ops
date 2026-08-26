"""Entry point: shared chrome (CSS, header, sidebar) + role-aware navigation."""

import streamlit as st

from services.ui import inject_css, render_header, render_sidebar_user

st.set_page_config(page_title="Rakuten", page_icon="🛍️", layout="centered")
inject_css()

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.role = None

render_header()
render_sidebar_user()

pages = [
    st.Page("views/connexion.py", title="Connexion", icon="🔑", default=True),
    st.Page("views/accueil.py", title="Accueil", icon="🏠"),
    st.Page("views/vendre.py", title="Vendre un produit", icon="🛒"),
]
if st.session_state.role == "admin":
    pages.append(st.Page("views/administration.py", title="Administration", icon="🛠️"))

st.navigation(pages).run()
