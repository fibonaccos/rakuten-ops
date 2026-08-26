"""
The training cleaner and the serving cleaner must agree, character for character.

Training cleans a DataFrame column by column (`rakuten.data.clean_data`); serving
cleans one product at a time (`model.pipeline.Cleaner`). They are two separate
implementations of the same contract, so any divergence is training-serving skew:
the model would be fed text it has never seen in training, and accuracy would drop
without a single error being raised.
"""

import pandas as pd
import pytest

from model.pipeline import Cleaner
from rakuten.data.clean_data import keep_characters, remove_html, remove_patterns

PRODUCTS: list[tuple[str, str]] = [
    ("Piscine gonflable", "Avec filtre et bache de protection"),
    ("<h1>Carte Dracaufeu</h1>", "<p>Edition <b>limitee</b>, 9/10</p>"),
    ("Vu sur www.rakuten.fr", "Ecrire a vendeur@rakuten.fr ou https://rakuten.fr/x"),
    ("Piscine ★ gonflable ✓", "Prix : 138€ — negociable"),
    ("Odyssee    de   Homere", "Version\ncollector\t\toriginale"),
]


def _clean_as_training_does(designation: str, description: str) -> tuple[str, str]:
    """Run the training cleaning chain on a single product."""
    columns = ["designation", "description"]
    frame = pd.DataFrame({"designation": [designation], "description": [description]})
    frame = remove_html(frame, columns)
    frame = remove_patterns(frame, columns)
    frame = keep_characters(frame, columns)
    return (frame.loc[0, "designation"], frame.loc[0, "description"])


@pytest.mark.parametrize(("designation", "description"), PRODUCTS)
def test_serving_cleans_exactly_like_training(designation: str, description: str) -> None:
    served = Cleaner()((designation, description))

    assert served == _clean_as_training_does(designation, description)


def test_a_batch_is_cleaned_exactly_like_a_sequence_of_singles() -> None:
    cleaner = Cleaner()

    batch = cleaner(PRODUCTS)

    assert batch == [cleaner(product) for product in PRODUCTS]
