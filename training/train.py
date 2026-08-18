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

    params: Box = Box(_PARAMS).train

    seed: int = params.get("random_state", 42)
    set_seed(seed)

    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT)

    X_train = pd.read_parquet(ROOT_DIR / params.input.x_train)
    X_test = pd.read_parquet(ROOT_DIR / params.input.x_test)
    y_train = pd.read_parquet(ROOT_DIR / params.input.y_train)
    y_test = pd.read_parquet(ROOT_DIR / params.input.y_test)[params.target]

    X = X_train.drop(columns=["productid"]).to_numpy()
    Xt = X_test.drop(columns=["productid"]).to_numpy()
    y_base = y_train[params.target].to_numpy()

    class_weights = compute_class_weight(y_train)

    codes_uniques = [int(c) for c in np.unique(y_base)]
    code_to_index = {code: i for i, code in enumerate(codes_uniques)}
    y = np.array([code_to_index[code] for code in y_base])
    yt = np.array([code_to_index[code] for code in y_test.to_numpy()])

    y_onehot = ks.utils.to_categorical(y)
    yt_onehot = ks.utils.to_categorical(yt)

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
        #à remplir avec ce que l'on juge utile        

        history = model.fit(
            X,
            y_onehot,
            epochs=15,
            batch_size=128,
            class_weight=class_weights,
            validation_data=(Xt, yt_onehot),
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

        mlflow.log_artifact(str(ROOT_DIR / "artifacts/scaler.joblib"))
        mlflow.log_artifact(str(ROOT_DIR / "artifacts/pca.joblib"))
        mlflow.log_artifact(str(ROOT_DIR / "artifacts/model.keras"))

        log_pipeline()

        print("Training complete !")


if __name__ == "__main__":
    train_model()
