"""End-to-End (E2E) Test Suite for Day 30 MLOps Capstone Project.

Validates:
1. Local artifacts (model, metrics, metadata, MLflow SQLite DB, DVC pointers)
2. Docker container health (api, prometheus, grafana)
3. Live API endpoints (health, predictions, input validation, prometheus metrics increment)
4. Monitoring connectivity (Prometheus scraping api:8000, Grafana health)
5. CI/CD workflow structure & dataset endpoint reachability
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

API_BASE_URL = "http://127.0.0.1:8000"
PROMETHEUS_BASE_URL = "http://127.0.0.1:9090"
GRAFANA_BASE_URL = "http://127.0.0.1:3001"


def http_request(url: str, method: str = "GET", data: dict = None, headers: dict = None):
    """Utility to perform HTTP requests with urllib, following redirects."""
    req_headers = headers or {}
    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode("utf-8")
        if "Content-Type" not in req_headers:
            req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=req_data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
            return status_code, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return e.code, body
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to {url}: {e.reason}") from e


def are_docker_containers_up() -> bool:
    """Probe if docker containers are running."""
    try:
        res = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode != 0:
            return False
        names = res.stdout.lower()
        return "api" in names and "prometheus" in names
    except Exception:
        return False


def is_api_up() -> bool:
    """Probe if live API server is reachable."""
    try:
        req = urllib.request.Request(f"{API_BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.getcode() == 200
    except Exception:
        return False


def is_prometheus_up() -> bool:
    """Probe if Prometheus server is reachable."""
    try:
        req = urllib.request.Request(f"{PROMETHEUS_BASE_URL}/-/healthy")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.getcode() == 200
    except Exception:
        return False


class TestLocalArtifacts(unittest.TestCase):
    """Stage 1: Verify all trained model and data artifacts exist and are valid."""

    def test_model_artifact_loadable(self):
        import joblib
        from sklearn.pipeline import Pipeline

        model_path = ROOT_DIR / "models" / "model.joblib"
        self.assertTrue(model_path.exists(), f"Model file not found at {model_path}")
        self.assertGreater(model_path.stat().st_size, 1000, "Model file is suspiciously small")

        model = joblib.load(model_path)
        self.assertIsInstance(model, Pipeline, "Loaded model is not an sklearn Pipeline")
        step_names = [name for name, _ in model.steps]
        self.assertIn("dictvectorizer", step_names, "Missing DictVectorizer step in pipeline")
        self.assertIn("linearregression", step_names, "Missing LinearRegression step in pipeline")

    def test_model_metadata(self):
        meta_path = ROOT_DIR / "models" / "model_metadata.json"
        self.assertTrue(meta_path.exists(), f"Metadata file not found at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("target_column"), "duration")
        self.assertEqual(
            data.get("feature_columns"), ["PULocationID", "DOLocationID", "trip_distance"]
        )

    def test_train_metrics(self):
        metrics_path = ROOT_DIR / "reports" / "train_metrics.json"
        self.assertTrue(metrics_path.exists(), f"Metrics file not found at {metrics_path}")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertIn("rmse", metrics)
        self.assertIn("train_rows", metrics)
        self.assertIn("test_rows", metrics)
        self.assertIn("model_type", metrics)
        self.assertGreater(metrics["rmse"], 0.0)
        self.assertGreater(metrics["train_rows"], 1000)
        self.assertGreater(metrics["test_rows"], 500)
        self.assertEqual(metrics["model_type"], "linear_regression")

    def test_mlflow_database(self):
        db_path = ROOT_DIR / "mlflow.db"
        self.assertTrue(db_path.exists(), f"MLflow database not found at {db_path}")
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            self.assertIn("runs", tables, "MLflow table 'runs' missing")
            self.assertIn("metrics", tables, "MLflow table 'metrics' missing")

            runs = cursor.execute("SELECT run_uuid, status FROM runs").fetchall()
            self.assertGreater(len(runs), 0, "No runs recorded in MLflow database")
            self.assertTrue(any(status == "FINISHED" for _, status in runs), "No FINISHED run in MLflow database")

            rmse_metrics = cursor.execute("SELECT value FROM metrics WHERE key='rmse'").fetchall()
            self.assertGreater(len(rmse_metrics), 0, "No rmse metric recorded in MLflow database")
            self.assertGreater(rmse_metrics[0][0], 0.0)
        finally:
            conn.close()

    def test_dvc_pointers_and_cache(self):
        dvc_pointer = ROOT_DIR / "data" / "raw" / "green_tripdata_2024-01.parquet.dvc"
        self.assertTrue(dvc_pointer.exists(), f"DVC pointer file not found at {dvc_pointer}")
        content = dvc_pointer.read_text(encoding="utf-8")
        self.assertIn("33f3af0093a4bea95512fbd0df7f1b84", content)
        self.assertIn("1362284", content)

        # Check raw parquet exists
        parquet_file = ROOT_DIR / "data" / "raw" / "green_tripdata_2024-01.parquet"
        self.assertTrue(parquet_file.exists(), f"Raw parquet data file not found at {parquet_file}")
        self.assertEqual(parquet_file.stat().st_size, 1362284)

        # Check dvc.yaml
        dvc_yaml = ROOT_DIR / "dvc.yaml"
        self.assertTrue(dvc_yaml.exists(), f"dvc.yaml not found at {dvc_yaml}")
        dvc_text = dvc_yaml.read_text(encoding="utf-8")
        self.assertIn("train_model", dvc_text)
        self.assertIn("models/model.joblib", dvc_text)


@unittest.skipUnless(are_docker_containers_up(), "Docker compose stack is not currently running")
class TestDockerContainers(unittest.TestCase):
    """Stage 2: Verify Docker container status via docker ps."""

    def test_containers_running(self):
        res = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = res.stdout.strip().split("\n")
        container_map = {}
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                container_map[parts[0]] = parts[1]

        # Verify api, prometheus, grafana are running
        api_running = any("api" in name and "Up" in status for name, status in container_map.items())
        prom_running = any("prometheus" in name and "Up" in status for name, status in container_map.items())
        graf_running = any("grafana" in name and "Up" in status for name, status in container_map.items())

        self.assertTrue(api_running, f"API container is not Up. Running containers: {container_map}")
        self.assertTrue(prom_running, f"Prometheus container is not Up. Running containers: {container_map}")
        self.assertTrue(graf_running, f"Grafana container is not Up. Running containers: {container_map}")


@unittest.skipUnless(is_api_up(), "Live API server is not currently running")
class TestLiveApiEndpoints(unittest.TestCase):
    """Stage 3: Test live FastAPI endpoints (health, predict, validation, metrics)."""

    def test_health_endpoint(self):
        status, body = http_request(f"{API_BASE_URL}/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))

    def test_predict_valid_payload(self):
        payload = {"PULocationID": "43", "DOLocationID": "151", "trip_distance": 2.5}
        status, body = http_request(f"{API_BASE_URL}/predict", method="POST", data=payload)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn("duration_minutes", data)
        duration = data["duration_minutes"]
        self.assertIsInstance(duration, (int, float))
        self.assertGreater(duration, 0.0)
        # Expected duration for ~2.5 miles in Manhattan is typically 8-15 minutes
        self.assertTrue(5.0 <= duration <= 30.0, f"Unusual predicted duration: {duration}")

    def test_predict_negative_distance_returns_400(self):
        payload = {"PULocationID": "43", "DOLocationID": "151", "trip_distance": -2.5}
        status, body = http_request(f"{API_BASE_URL}/predict", method="POST", data=payload)
        self.assertEqual(status, 400, f"Expected 400 for negative distance, got {status}: {body}")
        data = json.loads(body)
        self.assertIn("positive", data.get("detail", "").lower())

    def test_predict_zero_distance_returns_400(self):
        payload = {"PULocationID": "43", "DOLocationID": "151", "trip_distance": 0.0}
        status, body = http_request(f"{API_BASE_URL}/predict", method="POST", data=payload)
        self.assertEqual(status, 400, f"Expected 400 for zero distance, got {status}: {body}")

    def test_predict_missing_fields_returns_422(self):
        payload = {"PULocationID": "43"}
        status, body = http_request(f"{API_BASE_URL}/predict", method="POST", data=payload)
        self.assertEqual(status, 422, f"Expected 422 for missing fields, got {status}: {body}")

    def test_predict_invalid_data_types_returns_422(self):
        payload = {"PULocationID": "43", "DOLocationID": "151", "trip_distance": "non_numeric"}
        status, body = http_request(f"{API_BASE_URL}/predict", method="POST", data=payload)
        self.assertEqual(status, 422, f"Expected 422 for invalid type, got {status}: {body}")

    def test_metrics_counter_increments(self):
        def get_prediction_counter():
            # Handle possible redirect
            url = f"{API_BASE_URL}/metrics/"
            status, body = http_request(url)
            self.assertEqual(status, 200)
            for line in body.splitlines():
                if line.startswith("taxi_predictions_total ") or line.startswith("taxi_predictions_total{"):
                    parts = line.split()
                    return float(parts[-1])
            return 0.0

        initial_count = get_prediction_counter()

        # Fire 2 predictions
        for _ in range(2):
            payload = {"PULocationID": "43", "DOLocationID": "151", "trip_distance": 3.0}
            status, _ = http_request(f"{API_BASE_URL}/predict", method="POST", data=payload)
            self.assertEqual(status, 200)

        new_count = get_prediction_counter()
        self.assertEqual(new_count, initial_count + 2.0, f"Counter did not increment by 2: {initial_count} -> {new_count}")


@unittest.skipUnless(is_prometheus_up(), "Prometheus server is not currently running")
class TestMonitoringConnectivity(unittest.TestCase):
    """Stage 4: Test Prometheus and Grafana health and scraping target status."""

    def test_prometheus_healthy(self):
        status, body = http_request(f"{PROMETHEUS_BASE_URL}/-/healthy")
        self.assertEqual(status, 200)
        self.assertIn("Prometheus Server is Healthy", body)

    def test_prometheus_scraping_api_target(self):
        status, body = http_request(f"{PROMETHEUS_BASE_URL}/api/v1/targets")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data.get("status"), "success")
        active_targets = data.get("data", {}).get("activeTargets", [])
        self.assertGreater(len(active_targets), 0, "No active targets found in Prometheus")

        api_target = None
        for t in active_targets:
            if t.get("labels", {}).get("job") == "reference-solution-api":
                api_target = t
                break

        self.assertIsNotNone(api_target, "Target 'reference-solution-api' not found in Prometheus active targets")
        self.assertEqual(api_target.get("health"), "up", f"API target health is not 'up': {api_target}")
        self.assertEqual(api_target.get("lastError"), "", f"Prometheus reported scrape error: {api_target.get('lastError')}")

    def test_grafana_health(self):
        status, body = http_request(f"{GRAFANA_BASE_URL}/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data.get("database"), "ok")


class TestCiCdSanity(unittest.TestCase):
    """Stage 5: Inspect GitHub Actions CI/CD workflow structure."""

    def test_workflow_file_validity(self):
        workflow_path = ROOT_DIR / ".github" / "workflows" / "taxi-pipeline.yaml"
        self.assertTrue(workflow_path.exists(), f"Workflow file not found at {workflow_path}")
        content = workflow_path.read_text(encoding="utf-8")

        # Verify key workflow requirements
        self.assertIn("name: Taxi Pipeline", content)
        self.assertIn("runs-on: ubuntu-latest", content)
        self.assertIn("astral-sh/setup-uv", content)
        self.assertIn("uv sync --locked", content)
        self.assertIn("green_tripdata_2024-01.parquet", content)
        self.assertIn("uv run python src/train.py", content)
        self.assertIn("test -f models/model.joblib", content)
        self.assertIn("test -f reports/train_metrics.json", content)

    def test_dataset_url_reachable(self):
        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.getcode(), 200)
            self.assertEqual(int(resp.headers.get("Content-Length", 0)), 1362284)


if __name__ == "__main__":
    unittest.main(verbosity=2)
