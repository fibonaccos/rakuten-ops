"""Loaders and I/O helpers shared by the training pipeline."""

import pandas as pd
import pytest

from rakuten.data.utils import load, save
from rakuten.shared.exceptions import APIClientError, APIError, InferenceServiceError
from rakuten.shared.loaders import load_params


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "productid": [1, 2],
            "designation": ["piscine gonflable", "carte dracaufeu"],
            "prdtypecode": [2583, 40],
        }
    )


def test_a_csv_round_trip_preserves_the_frame(tmp_path, frame) -> None:
    path = tmp_path / "data.csv"

    save(frame, str(path))

    pd.testing.assert_frame_equal(load(str(path), format="csv"), frame)


def test_a_parquet_round_trip_preserves_the_frame(tmp_path, frame) -> None:
    path = tmp_path / "data.parquet"

    save(frame, str(path))

    pd.testing.assert_frame_equal(load(str(path), format="parquet"), frame)


def test_saving_never_writes_the_index(tmp_path, frame) -> None:
    path = tmp_path / "data.csv"

    save(frame, str(path))

    assert path.read_text(encoding="utf-8").splitlines()[0] == "productid,designation,prdtypecode"


def test_params_are_loaded_as_a_plain_dict(tmp_path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("clean:\n  input: data/raw/data.csv\n  keep: [a, b]\n", encoding="utf-8")

    params = load_params(str(path))

    assert params == {"clean": {"input": "data/raw/data.csv", "keep": ["a", "b"]}}


def test_the_shipped_training_params_are_readable() -> None:
    """The file the whole training pipeline is configured from must parse."""
    from tests.conftest import ROOT

    params = load_params(str(ROOT / "training" / "params.yaml"))

    assert "clean" in params
    assert isinstance(params, dict)


def test_client_errors_are_api_errors() -> None:
    """Callers catch APIError; a client error must not slip past that."""
    assert issubclass(APIClientError, APIError)
    assert not issubclass(InferenceServiceError, APIError)
