"""
Check data drift on logged requests and optionally trigger retraining.
"""

import json
from pathlib import Path

import click
from evidently import Report
from evidently.presets import DataDriftPreset

from src.generate_report import _as_dataset, build_reference, load_current
from src.preprocess import download_data
from src.train import run_training

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = ROOT_DIR / "monitoring" / "requests.csv"
SUMMARY_PATH = ROOT_DIR / "monitoring" / "reports" / "drift_summary.json"


def drift_summary(snapshot) -> dict:
    """Pull the drifted-columns count/share out of the Evidently snapshot."""
    for metric in snapshot.dict()["metrics"]:
        if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
            return {"drifted_columns": int(metric["value"]["count"]),
                    "drift_share": float(metric["value"]["share"])}
    raise RuntimeError("DriftedColumnsCount not found in the Evidently snapshot.")

@click.command()
@click.option("--current", "current_path", default=str(DEFAULT_CURRENT), show_default=True,
              type=click.Path(path_type=Path))
@click.option("--threshold", default=0.5, show_default=True, type=float,
              help="Retrain if at least this share of columns drifted.")
@click.option("--retrain", is_flag=True, help="Actually retrain when drift is detected.")
@click.option("--new-year", default=2024, show_default=True, type=int,
              help="Year of the fresh data to retrain on.")
@click.option("--new-month", default=2, show_default=True, type=int,
              help="Month of the fresh data to retrain on.")
def main(current_path, threshold, retrain, new_year, new_month) -> None:
    reference = _as_dataset(build_reference(2024, 1, 10_000, 42))
    current = _as_dataset(load_current(current_path))
    snapshot = Report(metrics=[DataDriftPreset()]).run(
        reference_data=reference, current_data=current
    )

    summary = drift_summary(snapshot)
    summary["threshold"] = threshold
    summary["drift_detected"] = summary["drift_share"] >= threshold

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    click.echo(json.dumps(summary, indent=2))

    if not summary["drift_detected"]:
        click.echo("No meaningful drift. No retraining needed.")
        return

    click.echo(f"Drift detected ({summary['drift_share']:.0%} of columns).")
    if not retrain:
        click.echo("Run again with --retrain to retrain the model.")
        return

    click.echo(f"Retraining on {new_year}-{new_month:02d} data...")
    download_data(new_year, new_month)
    metrics = run_training(new_year, new_month, "green", 0.2, 42, developer="Avi")
    click.echo(f"Retrained: {metrics}. Restart the API: docker compose restart api")


if __name__ == "__main__":
    main()