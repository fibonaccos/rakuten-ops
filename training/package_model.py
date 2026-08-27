import mlflow
import os
import pandas as pd
import yaml

from dotenv import load_dotenv
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModelContext
from pathlib import Path
from sentence_transformers import SentenceTransformer

from model.pyfunc_model import RakutenModel


EMBED_SENTENCE_TRANSFORMER = True
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
REGISTERED_NAME = "rakuten-naive"
EXPERIMENT = "rakuten"

ROOT_DIR = Path(__file__).parent.parent


def require_tracking_uri() -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        if not load_dotenv(ROOT_DIR / ".env"):
            raise FileNotFoundError("Unable to find a .env file.")
        return None
    raise RuntimeError("MLFLOW_TRACKING_URI must be defined.")


def log_pipeline(register: bool = True) -> str:
    """
    Logge le pyfunc dans la run MLflow ACTIVE. Ne cree pas de run.
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_pipeline doit etre appele dans une run active")

    params = yaml.safe_load(open(ROOT_DIR / "training/params.yaml"))["features"]["embedding"]
    model_config = {
        "embedder_name": EMBEDDER_NAME,
        "chunk_size": int(params["chunk_size"]),
        "overlap": int(params["overlap"]),
    }

    clean = pd.read_parquet(ROOT_DIR / "data/clean/data.parquet").astype("str")
    input_example = clean[["designation", "description"]].head(2).to_numpy()

    artifacts = {
        "keras_model": str(ROOT_DIR / "artifacts/model.keras"),
        "scaler": str(ROOT_DIR / "artifacts/scaler.joblib"),
        "pca": str(ROOT_DIR / "artifacts/pca.joblib"),
        "labels_map": str(ROOT_DIR / "artifacts/labels_map.json")
    }

    if EMBED_SENTENCE_TRANSFORMER:
        SentenceTransformer(EMBEDDER_NAME).save(str(ROOT_DIR / "artifacts" / EMBEDDER_NAME))
        artifacts["embedder"] = str(ROOT_DIR / "artifacts" / EMBEDDER_NAME)

    probe = RakutenModel()
    probe.load_context(
        PythonModelContext(artifacts=artifacts, model_config=model_config)
    )
    signature = infer_signature(input_example, probe.predict(None, input_example))

    mlflow.log_params(model_config)
    mlflow.pyfunc.log_model(
        name="rakuten_pipeline",
        python_model=RakutenModel(),
        code_paths=[str(ROOT_DIR / "model")],
        artifacts=artifacts,
        model_config=model_config,
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_NAME if register else None,
        pip_requirements=[
            "keras>=3.15.1", "sentence-transformers>=5.7.0", "scikit-learn>=1.9.0",
            "pandas<3.0.0", "lxml>=6.1.1", "joblib>=1.5.3", "numpy>=2.5.2"
        ]
    )

    uri = f"runs:/{run.info.run_id}/rakuten_pipeline"
    print("MODEL_URI =", uri)
    return uri


if __name__ == "__main__":
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="repackaging"):
        log_pipeline()
