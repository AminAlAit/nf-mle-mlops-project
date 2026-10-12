"""Generate an Evidently drift report: training data vs. logged API requests."""

from pathlib import Path

import click
import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERICAL_FEATURES,
    data_path,
    load_data,
    preprocess,
    to_feature_dicts,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
DEFAULT_CURRENT = ROOT_DIR / "monitoring" / "requests.csv"
DEFAULT_OUTPUT = ROOT_DIR / "monitoring" / "reports" / "data_drift_report.html"

COLUMNS = FEATURES + ["prediction"]

# Tell Evidently which columns are categories (zone IDs) vs. numbers,
# so it picks the right statistical test for each.
DATA_DEFINITION = DataDefinition(
    numerical_columns=NUMERICAL_FEATURES + ["prediction"],
    categorical_columns=CATEGORICAL_FEATURES,
)


def build_reference(year: int, month: int, sample_size: int, seed: int) -> pd.DataFrame:
    """Training-month trips plus the model's predictions on them."""
    df = preprocess(load_data(data_path(year, month)))
    df = df.sample(n=min(sample_size, len(df)), random_state=seed)
    model = joblib.load(MODEL_PATH)
    df["prediction"] = model.predict(to_feature_dicts(df))
    return df[COLUMNS]


def load_current(path: Path) -> pd.DataFrame:
    """Logged API requests (inputs + prediction)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Send requests first: uv run python -m src.send_requests"
        )
    df = pd.read_csv(path)
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} rows in {path}; send more requests first.")
    return df[COLUMNS]


def _as_dataset(df: pd.DataFrame) -> Dataset:
    df = df.copy()
    # Same type in both datasets: zone IDs as strings (categories).
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    return Dataset.from_pandas(df, data_definition=DATA_DEFINITION)


@click.command()
@click.option("--current", "current_path", default=str(DEFAULT_CURRENT), show_default=True,
              type=click.Path(path_type=Path), help="Request log to evaluate.")
@click.option("--output", "output_path", default=str(DEFAULT_OUTPUT), show_default=True,
              type=click.Path(path_type=Path), help="Where to write the HTML report.")
@click.option("--year", default=2024, show_default=True, type=int, help="Reference (training) year.")
@click.option("--month", default=1, show_default=True, type=int, help="Reference (training) month.")
@click.option("--sample-size", default=10_000, show_default=True, type=int,
              help="Reference rows to use (keeps the report fast).")
@click.option("--seed", default=42, show_default=True, type=int)
def main(current_path, output_path, year, month, sample_size, seed) -> None:
    """Compare reference (training) data with current (API request) data."""
    reference = _as_dataset(build_reference(year, month, sample_size, seed))
    current = _as_dataset(load_current(current_path))

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    click.echo(f"Wrote Evidently report to {output_path}")


if __name__ == "__main__":
    main()