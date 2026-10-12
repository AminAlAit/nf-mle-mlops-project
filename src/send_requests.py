"""Replay real taxi trips against the prediction API.

Fills Prometheus/Grafana metrics and monitoring/requests.csv (used by Evidently).
Use --distance-factor to simulate a shifted batch for the drift comparison.
"""

import time

import click
import requests

from src.preprocess import download_data, load_data, preprocess


@click.command()
@click.option("--year", default=2024, show_default=True, type=int)
@click.option("--month", default=2, show_default=True, type=int,
              help="Month to replay (default: Feb, unseen by the model).")
@click.option("--n", "n_requests", default=300, show_default=True, type=int,
              help="Number of trips to send.")
@click.option("--url", default="http://127.0.0.1:8000/predict", show_default=True)
@click.option("--delay", default=0.2, show_default=True, type=float,
              help="Seconds between requests (spreads traffic so Grafana shows a curve).")
@click.option("--distance-factor", default=1.0, show_default=True, type=float,
              help="Multiply trip_distance to simulate drift (e.g. 3.0).")
@click.option("--seed", default=42, show_default=True, type=int)
def main(year, month, n_requests, url, delay, distance_factor, seed) -> None:
    """Sample trips from one month and POST them to the API."""
    path = download_data(year, month)
    trips = preprocess(load_data(path)).sample(n=n_requests, random_state=seed)

    ok, failed = 0, 0
    for _, trip in trips.iterrows():
        payload = {
            "PULocationID": int(trip["PULocationID"]),
            "DOLocationID": int(trip["DOLocationID"]),
            # Keep within the API's validation range (0.1–100 miles).
            "trip_distance": min(float(trip["trip_distance"]) * distance_factor, 100.0),
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            ok += 1
        except requests.RequestException as exc:
            failed += 1
            click.echo(f"Request failed: {exc}", err=True)
        time.sleep(delay)

    click.echo(f"Sent {ok + failed} requests: {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()