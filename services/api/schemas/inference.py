from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """information about a model."""

    name: str = Field(..., description="Name of the model.")
    version: str = Field(..., description="Version of the model.")
    published_at: datetime = Field(
        ...,
        description="Timestamp of the model when it reached the production stage."
    )


class PredictionInput(BaseModel):
    """Inputs of the model."""

    designation: str = Field(..., description="Main information about the product.")
    description: str | None = Field(
        default=None,
        description="Optional description of the product."
    )


class PredictionOutput(BaseModel):
    """
    Output of the model which contains :
    - the predicted category
    - the confidence (the prob of the predicted category)
    - the distribution of prob of each category
    """

    category: str = Field(..., description="The predicted category (argmax).")
    confidence: float = Field(
        ...,
        description="The confidence for the predicted category."
    )
    distribution: dict[str, float] = Field(
        ...,
        description="The probability density of the categories."
    )


class SinglePredictionMetadata(BaseModel):
    """Metadata of a single prediction."""

    inference_id: int | None = Field(
        default=None,
        description="Identifier of this prediction in the inference log. Set after "
        "the prediction is persisted; use it to confirm or correct the category "
        "afterward via PATCH /predict/{inference_id}/label."
    )
    model_info: ModelInfo = Field(
        ...,
        description="Information about the model used for prediction."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the instant when prediction started."
    )
    inference_time_ms: float = Field(
        ...,
        description="Time taken by the model to make the prediction."
    )
    on_gpu: bool = Field(
        ...,
        description="Whether a gpu was enabled for inference."
    )


class BatchPredictionMetadata(BaseModel):
    """Metadata of a batch prediction."""

    model_info: ModelInfo = Field(
        ...,
        description="Information about the model used for prediction."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the instant when prediction started."
    )
    mean_inference_time_ms: float = Field(
        ...,
        description="Average time taken by the model to make a single prediction."
    )
    total_inference_time_ms: float = Field(
        ...,
        description="Time taken by the model to make the whole batch prediction."
    )
    on_gpu: bool = Field(
        ...,
        description="Whether a gpu was enabled for inference."
    )


class SinglePredictionResponse(BaseModel):
    """
    Full response of an API call to the prediction service for a single inference job :
    - a single output (PredictionOutput) matching the result of the model
    - the metadata (SinglePredictionMetadata) of the result
    """

    output: PredictionOutput = Field(
        ...,
        description="Full output of a single inference job."
    )
    metadata: SinglePredictionMetadata = Field(
        ...,
        description="Metadata of a single inference job."
    )


class BatchPredictionResponse(BaseModel):
    """
    Full response of an API call to the prediction service for a batch inference job :
    - a list of outputs (list[PredictionOutput]) matching the results of the model
    - the metadata (BatchPredictionMetadata) of the results
    """
    output: list[PredictionOutput] = Field(
        ...,
        description="Full output of a batch inference job."
    )
    metadata: BatchPredictionMetadata = Field(
        ...,
        description="Metadata of a batch inference job."
    )


class LabelPredictionRequest(BaseModel):
    """Body sent by the client to confirm or correct a previous prediction."""

    labeled_category: str = Field(
        ...,
        description="The category confirmed (same as predicted) or corrected "
        "(chosen by the user) by whoever made the original request."
    )


class LabelPredictionResponse(BaseModel):
    """Result of confirming or correcting a previous prediction."""

    inference_id: int = Field(..., description="Identifier of the labeled prediction.")
    labeled_category: str = Field(..., description="The category that was recorded.")
    matched_prediction: bool = Field(
        ...,
        description="True if the user confirmed the model's prediction, False if "
        "they corrected it to a different category."
    )
