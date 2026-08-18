import json 
import numpy as np
import sys

from mlflow.pyfunc.model import PythonModel
from mlflow.models import set_model

from model.pipeline import InferenceBatchResponse, InferencePipeline, InferenceSingleResponse


# TODO: All over this file, there are hard coded conversion of windows paths to posix.
#       This is a risky way to ensure artifacts paths are correctly read, thus a
#       future version will set the training pipeline into a linux container, and none
#       of the conversions below will remain useful.


def _convert_path_to_posix(artifact_path: str) -> str:
    return artifact_path.replace("\\", "/")


class RakutenModel(PythonModel):
    """
    Model stored by `mlflow` after a successful training. It is the only object that
    should be loaded by the inference service in the API lifespan.
    """

    def load_context(self, context) -> None:
        cfg = dict(context.model_config or {})
        if sys.platform in ("linux", "darwin"):
            ctx_model = _convert_path_to_posix(context.artifacts["keras_model"])
            ctx_labels_map = _convert_path_to_posix(context.artifacts["labels_map"])
            ctx_embedder_uri = _convert_path_to_posix(context.artifacts.get("embedder") or cfg["embedder_name"])
            ctx_scaler = _convert_path_to_posix(context.artifacts["scaler"])
            ctx_pca = _convert_path_to_posix(context.artifacts["pca"])
        else:
            ctx_model = context.artifacts["keras_model"]
            ctx_labels_map = context.artifacts["labels_map"]
            ctx_embedder_uri = context.artifacts.get("embedder") or cfg["embedder_name"]
            ctx_scaler = context.artifacts["scaler"]
            ctx_pca = context.artifacts["pca"]

        with open(ctx_labels_map, "r") as f:
            labels_map: dict[str, int] = json.load(f)

        self.pipeline = InferencePipeline(
            model_uri=ctx_model,
            embedder_uri=ctx_embedder_uri,
            scaler_uri=ctx_scaler,
            reducer_uri=ctx_pca,
            chunk_size=int(cfg["chunk_size"]),
            overlap=int(cfg["overlap"]),
            labels_map=labels_map,
        )
        return None

    def predict(  # type: ignore
        self,
        context,
        model_input: np.ndarray,
        params=None
    ) -> InferenceSingleResponse | InferenceBatchResponse:
        if model_input.shape[0] == 1:
            return self.pipeline.predict_single(
                tuple(model_input[0])
            )
        else:
            return self.pipeline.predict_batch(
                [tuple(x) for x in model_input]
            )


set_model(RakutenModel())
