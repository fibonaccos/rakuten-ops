from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from db.models import Inference, User
from db.session import get_session
from services.auth import get_current_user, ensure_user_is_admin_from_token
from services.predict import request_predict_single, request_predict_batch, feed_inference_db
from services.db import InferenceRepository
from schemas.inference import (
    BatchPredictionMetadata,
    BatchPredictionResponse,
    LabelPredictionRequest,
    LabelPredictionResponse,
    PredictionInput,
    PredictionOutput,
    SinglePredictionMetadata,
    SinglePredictionResponse
)


router = APIRouter(prefix="/predict", tags=["Predict"])


@router.post(
    path="/single",
    summary="single",
    description="Given an input, returns a prediction response using the current " \
    "model. Includes the predicted category, the confidence and the probability " \
    "density of each category.",
    responses={
        500: {"description": "Inference server unabled to provide response"}
    },
    response_model=SinglePredictionResponse
)
async def predict_single(
    inputs: PredictionInput,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> SinglePredictionResponse:
    query_id: str = uuid4().hex
    result = await request_predict_single(inputs.model_dump(mode="json"))
    try:
        output = PredictionOutput(**(result["output"]))
        metadata = SinglePredictionMetadata(**(result["metadata"]))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e))
    inference = Inference(
        user_id=user.user_id,
        model_name=metadata.model_info.name,
        model_version=metadata.model_info.version,
        query_id=query_id,
        batch=False,
        designation=inputs.designation,
        description=inputs.description,
        predicted_category=output.category,
        labeled_category=None,
        confidence=output.confidence,
        queried_at=metadata.timestamp
    )
    created = await feed_inference_db(inference, session)
    # inference_id n'existe qu'après l'écriture en base (auto-incrément) --
    # on le rattache aux métadonnées seulement maintenant, pour que le client
    # puisse confirmer/corriger cette prédiction via /predict/{id}/label.
    metadata.inference_id = created.inference_id
    return SinglePredictionResponse(output=output, metadata=metadata)


@router.post(
    path="/batch",
    summary="batch",
    description="Given a batch of inputs, returns a prediction response using the " \
    "current model. Includes the predicted category for each input in the " \
    "batch, the confidence and the probability density of each category.",
    responses={
        500: {"description": "Inference server unabled to provide response"}
    },
    response_model=BatchPredictionResponse
)
async def predict_batch(
    inputs: list[PredictionInput],
    user: User = Depends(ensure_user_is_admin_from_token),
    session: AsyncSession = Depends(get_session)
) -> BatchPredictionResponse:
    query_id: str = uuid4().hex
    result = await request_predict_batch([x.model_dump(mode="json") for x in inputs])
    try:
        output = [PredictionOutput(**r) for r in result["output"]]
        metadata = BatchPredictionMetadata(**(result["metadata"]))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e))
    inferences = []
    for inp, out in zip(inputs, output):
        inferences.append(
            Inference(
                user_id=user.user_id,
                model_name=metadata.model_info.name,
                model_version=metadata.model_info.version,
                query_id=query_id,
                batch=True,
                designation=inp.designation,
                description=inp.description,
                predicted_category=out.category,
                labeled_category=None,
                confidence=out.confidence,
                queried_at=metadata.timestamp
            )
        )
    await feed_inference_db(inferences, session)
    return BatchPredictionResponse(output=output, metadata=metadata)


@router.patch(
    path="/{inference_id}/label",
    summary="label",
    description="Confirm or correct the category of a previous single prediction. "
    "Send the same category to confirm the model was right, or a different one "
    "to correct it. Only the user who made the original prediction can label it -- "
    "this is what lets the drift monitoring pipeline retrain on genuinely "
    "confirmed labels instead of the model's own (possibly wrong) predictions.",
    responses={
        404: {"description": "Inference not found, or it doesn't belong to the current user"}
    },
    response_model=LabelPredictionResponse
)
async def label_prediction(
    inference_id: int,
    body: LabelPredictionRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> LabelPredictionResponse:
    repository = InferenceRepository(session)
    inference = await repository.update_label(inference_id, user.user_id, body.labeled_category)
    if inference is None:
        raise HTTPException(
            status_code=404,
            detail="Inference not found for this user."
        )
    return LabelPredictionResponse(
        inference_id=inference.inference_id,
        labeled_category=inference.labeled_category,
        matched_prediction=(inference.labeled_category == inference.predicted_category)
    )
