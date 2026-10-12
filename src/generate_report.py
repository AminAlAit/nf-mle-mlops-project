"""Generate an Evidently data drift report comparing reference data with live API requests."""

from pathlib import Path
import sys

ROOT_DIR            = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.utils.preprocess import FEATURE_COLUMNS, preprocess

REFERENCE_DATA_PATH = ROOT_DIR / "data" / "raw" / "green_tripdata_2024-01.parquet"
REQUEST_LOG_PATH    = ROOT_DIR / "monitoring" / "requests.csv"
REPORT_OUTPUT_PATH  = ROOT_DIR / "monitoring" / "reports" / "data_drift_report.html"

def generate_drift_report() -> Path:
    """Compare baseline reference features with live request logs."""
    if not REQUEST_LOG_PATH.exists():
        raise FileNotFoundError(
            "No request log found at monitoring/requests.csv. "
            "Send a few prediction requests to the API first."
        )

    # Load baseline reference data and process
    ref_df = pd.read_parquet(REFERENCE_DATA_PATH)
    ref_df = preprocess(ref_df)[FEATURE_COLUMNS].sample(n=min(1000, len(ref_df)), random_state=42)

    # Load live request history
    current_df = pd.read_csv(REQUEST_LOG_PATH)[FEATURE_COLUMNS]
    if len(current_df) < 5:
        raise ValueError(
            f"At least 5 requests needed to compute drift, found {len(current_df)}. "
            "Send more prediction requests to the API first."
        )

    # Run Evidently Data Drift Preset
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=current_df)

    # Save HTML snapshot
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_OUTPUT_PATH))
    return REPORT_OUTPUT_PATH

if __name__ == "__main__":
    report_file = generate_drift_report()
    print(f"Generated Evidently data drift report: {report_file}")
