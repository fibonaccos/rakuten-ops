"""Administration page. Hidden from the sidebar unless role == "admin" (see app.py)."""

from datetime import datetime

import streamlit as st

from services.api_client import ApiError, get_current_model, get_models, get_readiness
from services.ui import render_status

GRAFANA_URL = "http://localhost:3000"
MLFLOW_URL = "http://localhost:5001"


def format_date(value: str | None) -> str:
    """Render an ISO timestamp as a date, leaving anything unparseable alone."""
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%d/%m/%Y")
    except ValueError:
        return value


st.title("Administration")

# Defense in depth: app.py already hides this page from non-admins, but a role
# could change mid-session, so we re-check here too.
if st.session_state.get("role") != "admin":
    st.error("Cette page est réservée aux administrateurs.")
    st.stop()

token = st.session_state.token
st.success(f"Connecté en tant qu'admin : **{st.session_state.username}**")

# ── Backend readiness ────────────────────────────────────────────────────────
st.subheader("État de la plateforme")

if st.button("Rafraîchir"):
    st.rerun()

try:
    readiness = get_readiness(token)
except ApiError as error:
    st.error(f"Impossible de lire l'état des services : {error}")
else:
    labels = {
        "database": "Base de données",
        "model_registry": "Registre de modèles",
        "inference": "Service d'inférence",
    }
    for column, (key, label) in zip(st.columns(len(labels)), labels.items()):
        with column:
            render_status(label, readiness.get(key, "unknown"))

st.divider()

# ── Model registry ───────────────────────────────────────────────────────────
st.subheader("Modèles")

try:
    current = get_current_model(token)
except ApiError:
    current = None
    st.info("Aucun modèle n'est actuellement servi par le service d'inférence.")
else:
    st.markdown(f"**En production :** {current['name']} v{current['version']}")

try:
    models = get_models(token)
except ApiError as error:
    st.warning(f"Registre injoignable : {error}")
else:
    if not models:
        st.caption("Le registre ne contient encore aucun modèle enregistré.")
    else:
        in_production = (current["name"], current["version"]) if current else (None, None)
        st.table(
            [
                {
                    "Modèle": model["name"],
                    "Version": model["version"] or "—",
                    "Publié le": format_date(model.get("published_at")),
                    "En production": (
                        "oui" if (model["name"], model["version"]) == in_production else ""
                    ),
                }
                for model in models
            ]
        )

st.divider()

# ── Deeper tooling ───────────────────────────────────────────────────────────
st.subheader("Supervision détaillée")
st.caption(
    "Réutilise les outils déjà en place dans la stack plutôt que de "
    "dupliquer le monitoring dans Streamlit."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Santé API & infrastructure**")
    st.write("Dashboards Prometheus/Grafana de l'équipe (gateway, inference, infra).")
    st.link_button("Ouvrir Grafana", GRAFANA_URL)

with col2:
    st.markdown("**Performance des modèles**")
    st.write("Historique des métriques de chaque run d'entraînement.")
    st.link_button("Ouvrir MLflow", MLFLOW_URL)
