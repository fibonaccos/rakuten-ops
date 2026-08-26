"""Sell page: submit a product and get its category suggested automatically."""

import streamlit as st

from services.api_client import ApiError, get_current_model, predict

st.title("Déposer une annonce")

if not st.session_state.get("token"):
    st.warning("Connecte-toi d'abord pour déposer une annonce.")
    st.page_link("views/connexion.py", label="Se connecter", icon="🔑")
    st.stop()

# Surface clearly whether a trained model is available, instead of letting the
# predict call fail with a confusing error later. TODO équipe : ce bandeau
# disparaît une fois qu'un modèle est entraîné et enregistré dans MLflow.
try:
    model_info = get_current_model(st.session_state.token)
    st.caption(f"Modèle en production : **{model_info['name']}** v{model_info['version']}")
except ApiError:
    st.warning(
        "⚠️ Aucun modèle en production pour l'instant — la classification "
        "automatique ne fonctionnera pas tant qu'un modèle n'a pas été "
        "entraîné et publié (voir l'équipe ML)."
    )

designation = st.text_input(
    "Titre de l'annonce", placeholder="Ex : Manteau en laine femme, taille M"
)
description = st.text_area("Description", placeholder="État, matière, marque, dimensions...")

if st.button("Analyser automatiquement"):
    if not designation:
        st.warning("Le titre est obligatoire pour lancer la classification.")
    else:
        try:
            result = predict(st.session_state.token, designation, description)
        except ApiError as e:
            st.error(str(e))
        else:
            output = result["output"]
            st.success(f"Catégorie suggérée : **{output['category']}**")
            st.progress(output["confidence"], text=f"Confiance : {output['confidence']:.0%}")

            with st.expander("Voir la distribution complète des probabilités"):
                sorted_dist = sorted(
                    output["distribution"].items(), key=lambda kv: kv[1], reverse=True
                )
                for category, proba in sorted_dist[:10]:
                    st.write(f"{category} — {proba:.1%}")
