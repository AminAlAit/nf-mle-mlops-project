"""
Smoke test the prediction API without starting a server or touching the real request log.
"""

import asyncio
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from src import api

VALID_TRIP = {"PULocationID": 74, "DOLocationID": 75, "trip_distance": 1.8}


async def check_api() -> None:
    """Check health, a valid prediction, input validation, metrics and request logging."""
    with TemporaryDirectory() as directory:
        # Keep smoke-test requests out of the monitoring history.
        api.REQUEST_LOG_PATH = Path(directory) / "requests.csv"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            # 1. Health
            health = await client.get("/health")
            assert health.status_code == 200, health.text
            assert health.json()["status"] == "ok"

            # 2. Valid prediction
            response = await client.post("/predict", json=VALID_TRIP)
            assert response.status_code == 200, response.text
            prediction = response.json()["predicted_duration_min"]
            assert isinstance(prediction, float)
            assert 0 < prediction < 120, f"Implausible prediction: {prediction}"

            # 3. Invalid inputs are rejected (HTTP 422)
            negative = await client.post("/predict", json={**VALID_TRIP, "trip_distance": -1})
            assert negative.status_code == 422

            bad_zone = await client.post("/predict", json={**VALID_TRIP, "PULocationID": 999})
            assert bad_zone.status_code == 422

            missing = await client.post("/predict", json={"PULocationID": 74})
            assert missing.status_code == 422

            # 4. Prometheus metrics are exposed
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "taxi_prediction_requests_total" in metrics.text

        # 5. Exactly one (valid) request was logged
        with api.REQUEST_LOG_PATH.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1, f"Expected 1 logged request, got {len(rows)}"

    print(
        f"API checks passed: health, prediction ({prediction:.2f} min), "
        "3x HTTP 422, /metrics, request logging."
    )


if __name__ == "__main__":
    asyncio.run(check_api())