import pandas as pd

from box import Box
from pathlib import Path
from sklearn.model_selection import train_test_split

from ..shared.loaders import load_params
from .utils import load, save


def split_train_prod() -> None:
    """
    Sépare le dataset brut sans nettoyage en deux moitiés.

    Returns:
        None:
    """

    CONFIG_DIR = Path(__file__).parent.parent.parent.parent
    params: Box = Box(load_params(str(CONFIG_DIR / "training" / "params.yaml"))).train_prod

    df: pd.DataFrame = load(str(CONFIG_DIR / params.input), format="csv")

    train, prod = train_test_split(
        df,
        train_size=params.train_ratio,
        stratify=df[params.target],
        random_state=params.random_state,
    )

    train = train.reset_index(drop=True)
    prod = prod.reset_index(drop=True)

    save(train, str(CONFIG_DIR / params.output.train))
    save(prod, str(CONFIG_DIR / params.output.prod))

    print(f"Train : {len(train)} {params.output.train} lignes & prod :{len(prod)} {params.output.prod} lignes")

    return None


if __name__ == "__main__":
    split_train_prod()