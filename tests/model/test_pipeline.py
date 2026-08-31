"""
Pure parts of the inference pipeline: text cleaning, chunking, statistical features.

These are the pieces that must stay identical to the training pipeline in
`src/rakuten/data/`. Any drift here is training-serving skew, which degrades
predictions without raising anything.
"""

import numpy as np
import pytest

from model.pipeline import Cleaner, _chunk, _compute_statistics


@pytest.fixture
def cleaner() -> Cleaner:
    return Cleaner()


# ── Cleaner ───────────────────────────────────────────────────────────────────


def test_cleaning_keeps_designation_and_description_apart(cleaner) -> None:
    """
    The description must survive cleaning as its own text.

    Regression: the cleaner used to append one output per pattern instead of one
    per input, so the returned pair was the designation twice and the description
    never reached the model.
    """
    designation, description = cleaner(("Piscine gonflable", "Avec filtre et bache"))

    assert designation == "Piscine gonflable"
    assert description == "Avec filtre et bache"


def test_cleaning_applies_every_pattern_to_every_field(cleaner) -> None:
    designation, description = cleaner(
        ("Vu sur www.rakuten.fr", "Ecrire a vendeur@rakuten.fr ou https://rakuten.fr/x")
    )

    assert "www.rakuten.fr" not in designation
    assert "URL" in designation
    assert "vendeur@rakuten.fr" not in description
    assert "MAIL" in description
    assert "https://rakuten.fr/x" not in description
    assert "URL" in description


def test_cleaning_strips_html_markup(cleaner) -> None:
    designation, description = cleaner(
        ("<h1>Carte Dracaufeu</h1>", "<p>Edition <b>limitee</b></p>")
    )

    assert "<" not in designation and ">" not in designation
    assert designation == "Carte Dracaufeu"
    assert description == "Edition limitee"


def test_cleaning_replaces_a_missing_description(cleaner) -> None:
    _, description = cleaner(("Piscine gonflable", None))

    assert description == "NULL"


def test_cleaning_replaces_an_empty_description(cleaner) -> None:
    _, description = cleaner(("Piscine gonflable", ""))

    assert description == "NULL"


def test_cleaning_collapses_runs_of_whitespace(cleaner) -> None:
    designation, _ = cleaner(("Piscine     gonflable\n\n  ronde", "Rien"))

    assert designation == "Piscine gonflable ronde"


def test_cleaning_keeps_letters_digits_punctuation_and_currency(cleaner) -> None:
    designation, _ = cleaner(("Carte 9/10, estimee 138€ !", "Rien"))

    assert "9/10" in designation
    assert "138€" in designation


def test_cleaning_drops_symbols_that_are_not_currency(cleaner) -> None:
    designation, _ = cleaner(("Piscine ★ gonflable ✓", "Rien"))

    assert "★" not in designation
    assert "✓" not in designation


def test_cleaning_a_batch_returns_one_pair_per_product(cleaner) -> None:
    outputs = cleaner(
        [
            ("Piscine gonflable", "Avec filtre"),
            ("Carte Dracaufeu", "Edition limitee"),
            ("Odyssee de Homere", None),
        ]
    )

    assert len(outputs) == 3
    assert outputs[0] == ("Piscine gonflable", "Avec filtre")
    assert outputs[1] == ("Carte Dracaufeu", "Edition limitee")
    assert outputs[2][1] == "NULL"


# ── Chunking ──────────────────────────────────────────────────────────────────


def test_a_short_text_is_a_single_chunk() -> None:
    assert _chunk("piscine gonflable", chunk_size=150, overlap=30) == ["piscine gonflable"]


def test_a_long_text_is_split_into_overlapping_chunks() -> None:
    text = " ".join(str(i) for i in range(200))

    chunks = _chunk(text, chunk_size=150, overlap=30)

    assert len(chunks) > 1
    assert chunks[0].split()[0] == "0"
    # The second chunk restarts `overlap` words before the end of the first.
    assert chunks[1].split()[0] == "120"


def test_a_non_string_is_chunked_as_empty() -> None:
    assert _chunk(None, chunk_size=150, overlap=30) == [""]


# ── Statistical features ──────────────────────────────────────────────────────


def test_statistics_produce_twelve_features() -> None:
    features = _compute_statistics("piscine gonflable", "avec filtre")

    assert features.shape == (12,)
    assert features.dtype == np.float64


def test_statistics_measure_designation_then_description() -> None:
    features = _compute_statistics("abc de", "fghi")

    # designation: length, words, mean word length, max word length, digits, punctuation
    assert features[0] == 6.0
    assert features[1] == 2.0
    assert features[2] == pytest.approx(2.5)
    assert features[3] == 3.0
    # description
    assert features[6] == 4.0
    assert features[7] == 1.0


def test_statistics_count_digits_and_punctuation() -> None:
    features = _compute_statistics("carte 9/10, ex", "prix 138 euros")

    assert features[4] == 3.0  # 9, 1, 0
    assert features[5] == 2.0  # / and ,
    assert features[10] == 3.0  # 1, 3, 8


def test_statistics_of_an_empty_text_are_zero() -> None:
    features = _compute_statistics("", "")

    assert list(features) == [0.0] * 12
