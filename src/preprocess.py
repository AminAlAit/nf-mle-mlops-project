"""
Download, load and clean the data
"""

from pathlib import Path
from urllib.request import urlretrieve

import click
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

DEFAULT_YEAR = 2024
DEFAULT_MONTH = 1
DEFAULT_COLOR = "green"

CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID"]
NUMERICAL_FEATURES = ["trip_distance"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "duration"

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def data_path(year: int, month: int, color: str = DEFAULT_COLOR) -> Path:
    """Local path of the monthly parquet file."""
    return DATA_DIR / f"{color}_tripdata_{year}-{month:02d}.parquet"


def download_data(
    year: int, month: int, color: str = DEFAULT_COLOR, force: bool = False
) -> Path:
    """Download one month of trip data unless it is already present."""
    path = data_path(year, month, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        url = f"{BASE_URL}/{color}_tripdata_{year}-{month:02d}.parquet"
        urlretrieve(url, path)
    return path


def load_data(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Compute duration, drop outliers, and cast location IDs to categories.

    Note: trip_distance is the taximeter distance of a *completed* trip
    (historical replay), not an estimate available at pickup.
    """
    df = df.copy()
    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]

    df = df.copy()
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    return df[FEATURES + [TARGET]]


def to_feature_dicts(X: pd.DataFrame) -> list[dict]:
    """Convert features to the record format DictVectorizer expects.

    Location IDs are cast to str here too, so the API can pass ints safely.
    """
    X = X[FEATURES].copy()
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str)
    return X.to_dict(orient="records")


@click.command()
@click.option("--year", default=DEFAULT_YEAR, show_default=True, type=int)
@click.option("--month", default=DEFAULT_MONTH, show_default=True, type=int)
@click.option("--color", default=DEFAULT_COLOR, show_default=True)
@click.option("--force", is_flag=True, help="Re-download even if the file exists.")
def main(year: int, month: int, color: str, force: bool) -> None:
    """Download the raw data (used as the DVC `download_data` stage)."""
    path = download_data(year, month, color, force=force)
    click.echo(f"Data available at {path}")


if __name__ == "__main__":
    main()