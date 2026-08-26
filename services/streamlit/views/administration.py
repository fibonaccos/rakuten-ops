"""Administration page. Hidden from the sidebar unless role == "admin" (see app.py)."""

import streamlit as st

st.title("Administration")

# Defense in depth: app.py already hides this page from non-admins, but a role
# could change mid-session, so we re-check here too.
if st.session_state.role != "admin":
    st.error("Cette page est réservée aux administrateurs.")
    st.stop()

st.success(f"Connecté en tant qu'admin : **{st.session_state.username}**")

st.subheader("Supervision")
st.caption(
    "Réutilise les outils déjà en place dans la stack plutôt que de "
    "dupliquer le monitoring dans Streamlit."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Santé API & infrastructure**")
    st.write("Dashboards Prometheus/Grafana de l'équipe (gateway, inference, infra).")
    st.link_button("Ouvrir Grafana", "http://localhost:3000")

with col2:
    st.markdown("**Performance des modèles**")
    st.write("Historique des métriques de chaque run d'entraînement.")
    st.link_button("Ouvrir MLflow", "http://localhost:5001")

st.divider()
st.write(
    "Espace réservé à l'administration (contenu à définir : "
    "modération des annonces, gestion des comptes...)."
)
