"""Login page. On success, stores the access token and role in st.session_state."""

import streamlit as st

from services.api_client import ApiError, get_me, login

st.title("Se connecter")

if st.session_state.get("token"):
    st.success(f"Déjà connecté en tant que **{st.session_state.username}**")
    st.page_link("views/accueil.py", label="Aller à l'accueil", icon="🏠")
    st.stop()

with st.form("login_form"):
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    submitted = st.form_submit_button("Se connecter")

if submitted:
    try:
        token = login(username, password)
        me = get_me(token)
    except ApiError as e:
        st.error(str(e))
    else:
        st.session_state.token = token
        st.session_state.username = me["username"]
        st.session_state.role = me["role"]
        st.switch_page("views/accueil.py")
