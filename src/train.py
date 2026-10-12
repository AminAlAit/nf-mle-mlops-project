"""Train the minimal reference taxi duration model and log the result with MLflow."""

import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.mlflow_gateway import MLflowGateway
from src.utils.preprocess import FEATURE_COLUMNS, TARGET_COLUMN, preprocess

ROOT_DIR             = Path(__file__).resolve().parents[1]
DATA_PATH            = ROOT_DIR / "data" / "raw" / "green_tripdata_2024-01.parquet"
MODEL_DIR            = ROOT_DIR / "models"
MODEL_PATH           = MODEL_DIR / "model.joblib"
MODEL_METADATA_PATH  = MODEL_DIR / "model_metadata.json"
REPORTS_DIR          = ROOT_DIR / "reports"
METRICS_PATH         = REPORTS_DIR / "train_metrics.json"
MLFLOW_DB_PATH       = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "mlartifacts"


def build_pipeline() -> Any:
    """Create a unified vectorizer and linear regression pipeline."""
    return make_pipeline(
        DictVectorizer(),
        LinearRegression()
    )


def train_model(
    data_path: Path = DATA_PATH,
    gateway: Optional[MLflowGateway] = None,
) -> dict:
    """Train the taxi duration model and track the run via MLflow."""
    if gateway is None:
        gateway = MLflowGateway(
            db_path=MLFLOW_DB_PATH,
            artifacts_dir=MLFLOW_ARTIFACTS_DIR,
        )

    # Load raw records and extract features
    df           = pd.read_parquet(data_path)
    df_processed = preprocess(df)

    y            = df_processed[TARGET_COLUMN]
    X            = df_processed[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    pipeline     = build_pipeline()

    with gateway.start_run(run_name="linear-regression-taxi"):
        tags = {
            "model": "linear regression",
            "developer": "Amin",
            "dataset": "green-taxi",
            "year": 2024,
            "month": 1,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
        }

        # Fit model and evaluate validation loss
        pipeline.fit(X_train.to_dict(orient="records"), y_train)
        predictions = pipeline.predict(X_test.to_dict(orient="records"))
        rmse        = root_mean_squared_error(y_test, predictions)

        # Persist local artifacts
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        metrics  = {
            "rmse": round(float(rmse), 4),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "model_type": "linear_regression",
        }
        metadata = {
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        }

        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        # Log parameters and artifacts to registry
        params = {
            "model_type": "linear_regression",
            "test_size": 0.2,
            "random_state": 42,
        }
        gateway.log_run_data(
            tags=tags,
            params=params,
            metrics={"rmse": float(rmse)},
            artifacts=[MODEL_PATH, MODEL_METADATA_PATH, METRICS_PATH],
        )
        gateway.register_and_promote(
            pipeline=pipeline,
            model_name="green-taxi-ride-duration",
            alias="production",
        )

    return metrics


if __name__ == "__main__":
    result_metrics = train_model()
    print(json.dumps(result_metrics, indent=2))
