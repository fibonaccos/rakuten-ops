"""Sell page: submit a product and get its category suggested automatically."""

import streamlit as st

from services.api_client import ApiError, get_current_model, predict
from services.catalog import image_for, label_for, products_in_category
from services.ui import render_distribution, render_prediction

SIMILAR_SHOWN = 3

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

with st.form("annonce"):
    designation = st.text_input(
        "Titre de l'annonce", placeholder="Ex : Manteau en laine femme, taille M"
    )
    description = st.text_area(
        "Description",
        placeholder="État, matière, marque, dimensions...",
        help="Facultatif, mais le modèle s'appuie dessus autant que sur le titre.",
    )
    submitted = st.form_submit_button("Analyser automatiquement")

if submitted and not designation:
    st.warning("Le titre est obligatoire pour lancer la classification.")

elif submitted:
    result = None
    with st.spinner("Classification en cours..."):
        try:
            result = predict(st.session_state.token, designation, description)
        except ApiError as error:
            st.error(str(error))

    if result is not None:
        output = result["output"]
        render_prediction(output["category"], output["confidence"])

        st.markdown("**Les autres catégories envisagées**")
        render_distribution(output["distribution"])

        metadata = result.get("metadata", {})
        model = metadata.get("model_info", {})
        if model:
            st.caption(
                f"Classé par {model.get('name', 'le modèle')} v{model.get('version', '?')} "
                f"en {metadata.get('inference_time_ms', 0):.0f} ms."
            )

        # Showing what is already filed under the suggested category is the
        # quickest way for a seller to tell whether the answer is plausible.
        similar = products_in_category(label_for(output["category"]))[:SIMILAR_SHOWN]
        if similar:
            st.divider()
            st.markdown("**Déjà en ligne dans cette catégorie**")
            for column, product in zip(st.columns(len(similar)), similar):
                with column, st.container(border=True):
                    photo = image_for(product)
                    if photo:
                        st.image(str(photo), use_container_width=True)
                    else:
                        st.markdown(
                            f"<div class='product-thumb'>{product['emoji']}</div>",
                            unsafe_allow_html=True,
                        )
                    st.write(f"**{product['designation'][:60]}**")
                    st.write(product["price"])
