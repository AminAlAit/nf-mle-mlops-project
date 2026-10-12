"""
Prediction API:
historical replay
trip_distance is the taximeter distance of a completed trip instead of an estimate available at pickup.
"""

import csv
import json
import os
import threading
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.preprocess import FEATURES, to_feature_dicts

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
# Overridable so tests don't pollute the real monitoring log.
REQUEST_LOG_PATH = Path(
    os.getenv("REQUEST_LOG_PATH", ROOT_DIR / "monitoring" / "requests.csv")
)

# --- Prometheus metrics (system + model behaviour) ---
PREDICTION_REQUESTS = Counter(
    "taxi_prediction_requests_total", "Number of successful prediction requests."
)
PREDICTION_LATENCY = Histogram(
    "taxi_prediction_latency_seconds", "Time spent computing a prediction."
)
PREDICTED_DURATION = Histogram(
    "taxi_predicted_duration_minutes",
    "Distribution of predicted trip durations.",
    buckets=[2, 5, 10, 15, 20, 30, 45, 60],
)

REQUEST_LOG_LOCK = threading.Lock()


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run `uv run dvc repro` first.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    return model, metadata


MODEL, METADATA = _load_model()

app = FastAPI(
    title="Green Taxi Ride Duration API",
    description=(
        "Predicts trip duration (minutes) for completed NYC green taxi trips. "
        "Historical replay: trip_distance is the measured taximeter distance."
    ),
    version="1.0.0",
)


class TripRequest(BaseModel):
    PULocationID: int = Field(..., ge=1, le=265, examples=[74])
    DOLocationID: int = Field(..., ge=1, le=265, examples=[75])
    trip_distance: float = Field(
        ..., ge=0.1, le=100, allow_inf_nan=False, examples=[1.8],
        description="Taximeter distance in miles (known only after the trip).",
    )


class PredictionResponse(BaseModel):
    predicted_duration_min: float
    model_version: str | None = None


def _log_request(payload: TripRequest, prediction: float) -> None:
    """Append inputs + prediction; used later by the Evidently report."""
    REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {**payload.model_dump(), "prediction": prediction}
    with REQUEST_LOG_LOCK:
        new_file = not REQUEST_LOG_PATH.exists()
        with REQUEST_LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": METADATA.get("model_version")}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TripRequest) -> PredictionResponse:
    start = perf_counter()
    X = pd.DataFrame([payload.model_dump()])[FEATURES]
    prediction = float(MODEL.predict(to_feature_dicts(X))[0])

    _log_request(payload, prediction)
    PREDICTION_REQUESTS.inc()
    PREDICTED_DURATION.observe(prediction)
    PREDICTION_LATENCY.observe(perf_counter() - start)

    return PredictionResponse(
        predicted_duration_min=round(prediction, 2),
        model_version=str(METADATA.get("model_version")),
    )