"""Sell page: submit a product and get its category suggested automatically."""

import streamlit as st

from services.api_client import ApiError, get_current_model, label_prediction, predict
from services.catalog import image_for, label_for, load_products, products_in_category
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
    with st.spinner("Classification en cours..."):
        try:
            # Gardé en session_state (pas juste une variable locale) : les
            # boutons Confirmer/Corriger plus bas provoquent eux-mêmes un
            # rechargement de la page, et le résultat doit survivre à ça.
            st.session_state.last_prediction = predict(st.session_state.token, designation, description)
        except ApiError as error:
            st.error(str(error))

result = st.session_state.get("last_prediction")

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

    # --- Confirmer / corriger la catégorie -----------------------------------
    # C'est ce qui alimente labeled_category en base : sans cette étape, le
    # monitoring de drift ne trouve jamais de vraie étiquette à utiliser pour
    # le ré-entraînement (voir retrain_trigger_live.py).
    inference_id = metadata.get("inference_id")
    if inference_id is None:
        st.caption("Confirmation indisponible pour cette prédiction.")
    else:
        label_key = f"labeled_{inference_id}"
        if st.session_state.get(label_key):
            confirmed_category = st.session_state[label_key]
            st.success(f"Merci ! Catégorie enregistrée : **{label_for(confirmed_category)}**")
        else:
            st.markdown("**Cette catégorie est-elle la bonne ?**")
            col_confirm, col_correct = st.columns([1, 2])

            with col_confirm:
                if st.button("✅ Confirmer", key=f"confirm-{inference_id}"):
                    try:
                        label_prediction(st.session_state.token, inference_id, output["category"])
                        st.session_state[label_key] = output["category"]
                        st.rerun()
                    except ApiError as error:
                        st.error(str(error))

            with col_correct:
                catalogue = {p["prdtypecode"]: p["category"] for p in load_products()}
                labels_sorted = sorted(catalogue.items(), key=lambda kv: kv[1])
                chosen_label = st.selectbox(
                    "Ou choisis la bonne catégorie",
                    options=[name for _, name in labels_sorted],
                    index=None,
                    placeholder="Sélectionner une catégorie...",
                    key=f"correct-select-{inference_id}",
                    label_visibility="collapsed",
                )
                if chosen_label and st.button("Valider la correction", key=f"correct-btn-{inference_id}"):
                    code = next(c for c, name in labels_sorted if name == chosen_label)
                    try:
                        label_prediction(st.session_state.token, inference_id, code)
                        st.session_state[label_key] = code
                        st.rerun()
                    except ApiError as error:
                        st.error(str(error))

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
                    st.image(str(photo), width="stretch")
                else:
                    st.markdown(
                        f"<div class='product-thumb'>{product['emoji']}</div>",
                        unsafe_allow_html=True,
                    )
                st.write(f"**{product['designation'][:60]}**")
                st.write(product["price"])
