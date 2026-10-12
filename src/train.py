"""
Train the model and track it with MLflow
"""

import json
import os
from pathlib import Path

import click
import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from src.preprocess import (
    DEFAULT_COLOR,
    DEFAULT_MONTH,
    DEFAULT_YEAR,
    FEATURES,
    TARGET,
    data_path,
    load_data,
    preprocess,
    to_feature_dicts,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "train_metrics.json"

EXPERIMENT_NAME = "green-taxi-monitoring"
REGISTERED_MODEL_NAME = "green-taxi-ride-duration"
MODEL_ALIAS = "production"


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_pipeline() -> Pipeline:
    """One-hot encode the zones and fit a linear regression."""
    return make_pipeline(DictVectorizer(), LinearRegression())


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the pipeline on the training data."""
    pipeline = build_pipeline()
    pipeline.fit(to_feature_dicts(X_train), y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Return the RMSE on the test set (in minutes)."""
    y_pred = model.predict(to_feature_dicts(X_test))
    return float(root_mean_squared_error(y_test, y_pred))


def save_artifacts(model: Pipeline, metadata: dict, metrics: dict) -> None:
    """Save the model, its metadata and metrics in predictable locations."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def setup_mlflow() -> None:
    """Point MLflow at the tracking store from .env and select the experiment."""
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def register_model(model_uri: str) -> str:
    """Register a new model version and point the production alias at it."""
    model_version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, MODEL_ALIAS, model_version.version
    )
    return model_version.version


def run_training(
    year: int, month: int, color: str, test_size: float, random_state: int,
    developer: str,
) -> dict:
    """Full training run: load → preprocess → split → train → evaluate → log."""
    path = data_path(year, month, color)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python -m src.preprocess` "
            "or `uv run dvc repro` first."
        )

    df = preprocess(load_data(path))
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

    setup_mlflow()
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "model": "linear regression",
            "developer": developer,
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": FEATURES,
            "target": TARGET,
        })
        mlflow.log_params({
            "year": year,
            "month": month,
            "color": color,
            "test_size": test_size,
            "random_state": random_state,
            "model_type": "DictVectorizer+LinearRegression",
        })

        model = train_model(X_train, y_train)
        rmse = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("rmse", rmse)

        model_info = mlflow.sklearn.log_model(
            sk_model=model, name="mlflow-model", serialization_format="skops"
        )
        version = register_model(model_info.model_uri)

        metrics = {
            "rmse": round(rmse, 4),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
        metadata = {
            "features": FEATURES,
            "target": TARGET,
            "year": year,
            "month": month,
            "mlflow_run_id": run.info.run_id,
            "registered_model": REGISTERED_MODEL_NAME,
            "model_version": version,
        }
        save_artifacts(model, metadata, metrics)
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="model")

    return metrics


@click.command()
@click.option("--year", default=DEFAULT_YEAR, show_default=True, type=int)
@click.option("--month", default=DEFAULT_MONTH, show_default=True, type=int)
@click.option("--color", default=DEFAULT_COLOR, show_default=True)
@click.option("--test-size", default=0.2, show_default=True, type=float)
@click.option("--random-state", default=42, show_default=True, type=int)
@click.option("--developer", default="Avi", show_default=True)
def main(year, month, color, test_size, random_state, developer) -> None:
    """Train the model and print its metrics."""
    metrics = run_training(year, month, color, test_size, random_state, developer)
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()