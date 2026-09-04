import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras as ks
import mlflow
import numpy as np
import pandas as pd
import random

from box import Box
from collections import Counter
from keras.callbacks import ModelCheckpoint
from keras.metrics import AUC
from pathlib import Path
from typing import Any
from yaml import safe_load

from training.package_model import log_pipeline, require_tracking_uri, EXPERIMENT
from training.compare_promote import compare_and_promote
from src.rakuten.models.evaluate import make_report, make_confusion_matrix


PARAMS_FILEPATH: Path = Path(__file__).parent / "params.yaml"
ROOT_DIR: Path = Path(__file__).parent.parent

with open(PARAMS_FILEPATH, "r") as f:
    _PARAMS: dict[str, Any] = safe_load(f)


def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility across python, numpy and keras/TF

    Args:
        seed (int): the seed value to use everywhere
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    ks.utils.set_random_seed(seed)


def create_model(n_features: int, n_classes: int) -> ks.models.Model:
    """
    Create a Keras Model instance with input shape `n_features` and output shape
    `n_classes`.

    Args:
        n_features (int): The input dimension of the neural network.
        n_classes (int): The output dimension of the neural network.

    Returns:
        ks.models.Model: A neural network with the corresponding input and output
            shape.
    """

    inputs = ks.layers.Input(shape=(n_features,), name="input")

    x = ks.layers.Dense(units=256, activation="relu", name="dense1")(inputs)
    x = ks.layers.Dropout(rate=0.4, name="drop1")(x)

    output = ks.layers.Dense(units=n_classes, activation="softmax", name="output")(x)

    model = ks.models.Model(inputs=inputs, outputs=output)
    return model


def compute_class_weight(labels: pd.DataFrame) -> dict[str, float]:
    """
    Compute weights for each classes to rebalance the importance of loss on each class.
    Weights are computed on the train set. The less important a class is in the
    dataset, the higher its weight is, resulting to a bigger loss fore less
    representative classes.

    Args:
        labels (pd.DataFrame): The labels of the train set.

    Returns:
        dict[str, float]: A dict mapping each label to its corresponding weight.
    """

    counts = Counter(labels.to_numpy()[:, 0])
    N = labels.shape[0]
    K: int = labels.nunique().to_numpy()[0]
    weights = {str(cls): float(N / (K * count)) for cls, count in counts.items()}
    return weights


def train_model() -> None:
    """
    Main function to train the model built using `create_model`. It uses the parameters
    loaded from the `params.yaml` file at the root of the `core` directory.

    Stateful artefacts and some metadata are saved in a dedicated folder.

    The fields used for training the model can be found under the `train` key of the
    `params.yaml` file.
    """

    params_all: Box = Box(_PARAMS)
    params: Box = params_all.train
 
    seed: int = params.get("random_state", 42)
    set_seed(seed)
 
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT)
 
    X_train = pd.read_parquet(ROOT_DIR / params.input.x_train)
    X_val = pd.read_parquet(ROOT_DIR / params.input.x_val)
    X_test = pd.read_parquet(ROOT_DIR / params.input.x_test)
    y_train = pd.read_parquet(ROOT_DIR / params.input.y_train)
    y_val = pd.read_parquet(ROOT_DIR / params.input.y_val)
    y_test = pd.read_parquet(ROOT_DIR / params.input.y_test)
 
    X = X_train.drop(columns=["productid"]).to_numpy()
    Xv = X_val.drop(columns=["productid"]).to_numpy()
    Xt = X_test.drop(columns=["productid"]).to_numpy()
 
    all_codes = sorted(
        set(y_train[params.target].tolist())
        | set(y_val[params.target].tolist())
        | set(y_test[params.target].tolist())
    )
    code_to_index = {int(code): i for i, code in enumerate(all_codes)}
 
    class_weights = compute_class_weight(y_train)
 
    y = np.array([code_to_index[code] for code in y_train[params.target].to_numpy()])
    yv = np.array([code_to_index[code] for code in y_val[params.target].to_numpy()])
    yt = np.array([code_to_index[code] for code in y_test[params.target].to_numpy()])
 
    y_onehot = ks.utils.to_categorical(y, num_classes=params.n_classes)
    yv_onehot = ks.utils.to_categorical(yv, num_classes=params.n_classes)
 
    model = create_model(n_features=X.shape[1], n_classes=params.n_classes)
 
    checkpoint_cb = ModelCheckpoint(
        ROOT_DIR / params.output.model,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=0,
    )
 
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[AUC(multi_label=True, name="auc"), "accuracy"],
    )
 
    with mlflow.start_run(run_name="train"):
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_classes", params.n_classes)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))
 
        history = model.fit(
            X,
            y_onehot,
            epochs=5,
            batch_size=256,
            class_weight=class_weights,
            validation_data=(Xv, yv_onehot), 
            callbacks=[checkpoint_cb],
        )
 
        with open(ROOT_DIR / params.output.history, "w") as f:
            json.dump(history.history, f, indent=2)
 
        with open(ROOT_DIR / params.output.labels_map, "w") as f:
            json.dump(code_to_index, f, indent=2)
 
        with open(ROOT_DIR / params.output.class_weights, "w") as f:
            json.dump(class_weights, f, indent=2)
 
        mlflow.log_metric("accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("auc", history.history["auc"][-1])
        mlflow.log_metric("loss", history.history["loss"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
        mlflow.log_metric("val_auc", history.history["val_auc"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
 
        mlflow.log_metric("best_val_auc", max(history.history["val_auc"]))
        mlflow.log_metric("best_val_accuracy", max(history.history["val_accuracy"]))
 
        mlflow.log_artifact(str(ROOT_DIR / params.output.history))
        mlflow.log_artifact(str(ROOT_DIR / params.output.labels_map))
        mlflow.log_artifact(str(ROOT_DIR / params.output.class_weights))

        mlflow.log_artifact(str(ROOT_DIR / "artifacts/embedder.joblib"))
        mlflow.log_artifact(str(ROOT_DIR / "artifacts/scaler.joblib"))
        mlflow.log_artifact(str(ROOT_DIR / "artifacts/pca.joblib"))
        mlflow.log_artifact(str(ROOT_DIR / "artifacts/model.keras"))
 
        best_model: Any = ks.models.load_model(ROOT_DIR / params.output.model)
        y_pred = best_model.predict(Xt)
 
        report = make_report(
            yt, y_pred, code_to_index, str(ROOT_DIR / params.output.metrics)
        )
        make_confusion_matrix(
            yt, y_pred, code_to_index, str(ROOT_DIR / params.output.confusion_matrix)
        )
 
        test_metrics = {
            f"test_{k}": v for k, v in report["global"].items() if k != "support"
        }
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("test_support", report["global"]["support"])
 
        mlflow.log_artifact(str(ROOT_DIR / params.output.metrics))
        mlflow.log_artifact(str(ROOT_DIR / params.output.confusion_matrix))
 
        log_pipeline()
 
        print("Training Done")
        print("Métriques test :")
        for k, v in test_metrics.items():
            print(f"  {k} = {v:.4f}")
 
        if params_all.champion_challenger.get("auto_promote", True):
            print("Comparaison Champion/Challenger automatique")
            compare_and_promote()
        else:
            print("auto_promote désactivé : comparaison non lancée.")
 
 
if __name__ == "__main__":
    train_model()
