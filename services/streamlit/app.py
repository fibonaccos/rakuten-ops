"""Entry point: shared chrome (CSS, header, sidebar) + role-aware navigation."""

import streamlit as st

from services.ui import inject_css, render_header, render_sidebar, render_footer

st.set_page_config(page_title="Rakuten", page_icon="🏷️", layout="centered")
inject_css()

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.role = None
if "category_filter" not in st.session_state:
    st.session_state.category_filter = None

pages = [
    st.Page("views/accueil.py", title="Accueil", icon="🏠", default=True),
    st.Page("views/connexion.py", title="Connexion", icon="🔑"),
    st.Page("views/vendre.py", title="Vendre un produit", icon="🛒"),
]
if st.session_state.role == "admin":
    pages.append(st.Page("views/administration.py", title="Administration", icon="🛠️"))

# position="hidden": we build our own sidebar instead of Streamlit's automatic
# page list. st.navigation() must still run before any st.page_link() call
# (used in render_header/render_sidebar below) so Streamlit knows the app's
# page structure and can resolve links correctly.
navigation = st.navigation(pages, position="hidden")

render_header()
render_sidebar()

navigation.run()

render_footer()
