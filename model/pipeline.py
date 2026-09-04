import os

os.environ["KERAS_BACKEND"] = "torch"

import joblib
import keras as ks
import numpy as np
import re
import unicodedata

from lxml import html
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from time import perf_counter
from typing import Any


InferenceSingleResponse = tuple[str, float, dict[str, float], float]
InferenceBatchResponse = tuple[list[tuple[str, float, dict[str, float]]], float, float]


def _compute_statistics(designation: str, description: str) -> np.ndarray:
    desi_words = [len(w) for w in designation.split()]
    desi_mean_word_length = float(np.mean(desi_words)) if desi_words and len(desi_words) > 0 else 0.
    desi_max_word_length = float(max(desi_words)) if desi_words and len(desi_words) > 0 else 0.
    desi_num_digits = sum([unicodedata.category(c).startswith("N") for c in designation])
    desi_num_punctuation = sum([unicodedata.category(c).startswith("P") for c in designation])

    desc_words = [len(w) for w in description.split()]
    desc_mean_word_length = float(np.mean(desc_words)) if desc_words and len(desc_words) > 0 else 0.
    desc_max_word_length = float(max(desc_words)) if desc_words and len(desc_words) > 0 else 0.
    desc_num_digits = sum([unicodedata.category(c).startswith("N") for c in description])
    desc_num_punctuation = sum([unicodedata.category(c).startswith("P") for c in description])

    return np.array([
        float(len(designation)),
        float(len(desi_words)),
        desi_mean_word_length,
        desi_max_word_length,
        desi_num_digits,
        desi_num_punctuation,
        float(len(description)),
        float(len(desc_words)),
        desc_mean_word_length,
        desc_max_word_length,
        desc_num_digits,
        desc_num_punctuation
    ])


class Cleaner:
    def __init__(self) -> None:
        return None

    def __call__(
        self,
        inputs: tuple[str, str | None] | list[tuple[str, str | None]],
        /
    ) -> tuple[str, str] | list[tuple[str, str]]:
        if isinstance(inputs, tuple):
            return self._call_single(inputs)
        return self._call_batch(inputs)

    def _call_single(
        self,
        inputs: tuple[str, str | None],
        /
    ) -> tuple[str, str]:
        outputs = self._remove_html(inputs)
        outputs = self._remove_patterns(outputs)
        outputs = self._keep_characters(outputs)
        return outputs

    def _call_batch(
        self,
        inputs: list[tuple[str, str | None]],
        /
    ) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        for x in inputs:
            y = self._remove_html(x)
            y = self._remove_patterns(y)
            y = self._keep_characters(y)
            outputs.append(y)
        return outputs

    def _remove_html(self, /, inputs: tuple[str, str | None]) -> tuple[str, str]:
        outputs: list[str] = []
        for t in inputs:
            if not isinstance(t, str) or len(t) == 0:
                outputs.append("NULL")
                continue
            tree = html.fromstring(t)
            outputs.append(tree.text_content())
        return (outputs[0], outputs[1])

    def _remove_patterns(self, /, inputs: tuple[str, str]) -> tuple[str, str]:
        patterns = {
            "URL": r'https?://\S+|www\.\S+',
            "MAIL": r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
        }
        outputs: list[str] = []
        for t in inputs:
            # Both patterns apply to the same text, one after the other, and each
            # text yields exactly one output, as in clean_data.remove_patterns
            # which produced the data the model was trained on. Appending inside
            # the pattern loop returned the designation twice and dropped the
            # description before it ever reached the model.
            for name, pattern in patterns.items():
                t = re.sub(pattern, name, t)
            outputs.append(t)
        return (outputs[0], outputs[1])

    def _keep_characters(self, /, inputs: tuple[str, str]) -> tuple[str, str]:
        outputs: list[str] = []
        for t in inputs:
            cleaned = []
            for char in t:
                cat = unicodedata.category(char)
                if cat.startswith("L") \
                or cat.startswith("N") \
                or cat.startswith("P") \
                or cat == "Sc" \
                or char.isspace():
                    cleaned.append(char)
            text = "".join(cleaned)
            text = re.sub(r"\s+", " ", text).strip()
            text = unicodedata.normalize("NFKC", text)
            outputs.append(text)
        return (outputs[0], outputs[1])


class Embedder:
    def __init__(
        self,
        /,
        model: TfidfVectorizer
    ) -> None:
        self._embedder_model: TfidfVectorizer = model

    def __call__(
        self,
        /,
        inputs: tuple[str, str] | list[tuple[str, str]]
    ) -> np.ndarray:
        if isinstance(inputs, tuple):
            return self._embed_single(inputs)
        return self._embed_batch(inputs)

    def _embed_single(self, /, inputs: tuple[str, str]) -> np.ndarray:
        x = inputs[0] + " " + inputs[1]
        return np.array(self._embedder_model.transform([x]).todense()).reshape((-1, ))

    def _embed_batch(self, /, inputs: list[tuple[str, str]]) -> np.ndarray:
        return np.array([self._embed_single(x) for x in inputs])


class Preprocessor:
    def __init__(
        self,
        /,
        embedder_uri: str,
        scaler_uri: str,
        reducer_uri: str
    ) -> None:
        self._embedder: Embedder = Embedder(joblib.load(embedder_uri))
        self._scaler: StandardScaler = joblib.load(scaler_uri)
        self._reducer: PCA = joblib.load(reducer_uri)
        return None

    def __call__(self, inputs: tuple[str, str] | list[tuple[str, str]]) -> np.ndarray:
        if isinstance(inputs, tuple):
            return self._call_single(inputs)
        return self._call_batch(inputs)

    def _call_single(self, inputs: tuple[str, str]) -> np.ndarray:
        embedded = self._embedder(inputs).reshape(1, -1)
        to_transform = np.hstack([
            _compute_statistics(inputs[0], inputs[1]).reshape(1, -1), embedded
            ]
        )
        return self._reducer.transform(self._scaler.transform(to_transform))

    def _call_batch(self, inputs: list[tuple[str, str]]) -> np.ndarray:
        embedded = self._embedder(inputs)
        computed = np.vstack([_compute_statistics(x, y) for (x, y) in inputs])
        to_transform = np.hstack([computed, embedded])
        return self._reducer.transform(self._scaler.transform(to_transform))


class _Model:
    def __init__(self, /, model_uri: str) -> None:
        self._model: Any = ks.saving.load_model(model_uri)
        return None

    def __call__(self, inputs: np.ndarray) -> list:
        if inputs.shape[0] == 1:
            return list(self._model.predict(inputs))[0]
        return list(self._model.predict(inputs))


class InferencePipeline:
    def __init__(
        self,
        /,
        model_uri: str,
        embedder_uri: str,
        scaler_uri: str,
        reducer_uri: str,
        labels_map: dict[str, int]
    ) -> None:
        self._cleaner: Cleaner = Cleaner()
        self._preprocessor: Preprocessor = Preprocessor(
            embedder_uri=embedder_uri,
            scaler_uri=scaler_uri,
            reducer_uri=reducer_uri
        )
        self._model: _Model = _Model(model_uri)
        self._labels_map: dict[int, str] = {k: v for v, k in labels_map.items()}
        return None

    def predict_single(
        self,
        /,
        inputs: tuple[str, str | None]
    ) -> InferenceSingleResponse:
        start: float = perf_counter()
        y = self._model(self._preprocessor(self._cleaner(inputs)).reshape(1, -1))
        y = [float(v) for v in y]
        category: str = self._labels_map[y.index(max(y))]
        confidence: float = max(y)
        density: dict[str, float] = {
            code: y[i] for i, code in self._labels_map.items()
        }
        elapsed_ms: float = 1000 * (perf_counter() - start)
        return (category, confidence, density, elapsed_ms)

    def predict_batch(
        self,
        /,
        inputs: list[tuple[str, str | None]]
    ) -> InferenceBatchResponse:
        start: float = perf_counter()
        print("cleaning : ", self._cleaner(inputs))
        y: list[list[float]] = self._model(self._preprocessor(self._cleaner(inputs)))
        y = [[float(v) for v in output] for output in y]
        categories: list[str] = [str(self._labels_map[c.index(max(c))]) for c in y]
        confidences: list[float] = [max(c) for c in y]
        densities: list[dict[str, float]] = [
            {code: c[i] for i, code in self._labels_map.items()} for c in y
        ]
        y_batch: list[tuple[str, float, dict[str, float]]] = [
            (cat, conf, dens)
            for (cat, conf, dens) in zip(categories, confidences, densities)
        ]
        elapsed_ms: float = 1000 * (perf_counter() - start)
        return (y_batch, elapsed_ms, elapsed_ms / len(y_batch))
